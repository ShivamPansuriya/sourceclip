"""Main dubbing pipeline orchestrator."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from dub.align import align_segments_to_timeline, compute_total_duration_diff, match_durations
from dub.config import DubConfig
from dub.diarize import diarize
from dub.hardware import detect_hardware, select_models
from dub.lipsync import apply_lipsync
from dub.models import DubResult, ExitCode, Segment
from dub.subtitles import generate_subtitles
from dub.translate import translate_segments
from dub.tts import synthesize_segments
from dub.transcribe import transcribe
from dub.utils import console, format_duration, logger, print_result, write_json
from dub.video import extract_audio, extract_audio_segments, merge_dubbed_audio, probe_video

PIPELINE_STEPS = [
    "Probing video",
    "Detecting hardware",
    "Extracting audio",
    "Transcribing speech",
    "Detecting speakers",
    "Translating text",
    "Generating speech",
    "Matching durations",
    "Building subtitles",
    "Merging output",
    "Lip-sync (optional)",
    "Finalizing",
]


def run_pipeline(config: DubConfig) -> DubResult:
    """Execute the full dubbing pipeline."""
    start_time = time.time()
    config.ensure_dirs()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Dubbing pipeline", total=len(PIPELINE_STEPS))

        # Step 1: Probe video
        progress.update(task, description="Probing video...")
        video_info = probe_video(config.input_path)
        logger.info(
            "Input: %s (%dx%d, %.1f FPS, %s)",
            config.input_path.name, video_info.width, video_info.height,
            video_info.fps, format_duration(video_info.duration),
        )
        progress.advance(task)

        # Step 2: Hardware detection
        progress.update(task, description="Detecting hardware...")
        hw = detect_hardware()
        whisper_model = config.whisper_model if config.whisper_model != "auto" else hw.whisper_model
        hw = select_models(hw, config.whisper_model, config.tts_model)
        progress.advance(task)

        # Step 3: Extract audio
        progress.update(task, description="Extracting audio...")
        audio_path = config.work_dir / "audio" / "full.wav"
        extract_audio(config.input_path, audio_path)
        progress.advance(task)

        # Step 4: Transcribe
        progress.update(task, description="Transcribing speech...")
        src_lang = None if config.source_lang == "auto" else config.source_lang
        try:
            segments, detected_lang = transcribe(
                audio_path,
                model_size=hw.whisper_model,
                device=hw.device,
                compute_type=hw.whisper_compute,
                language=src_lang,
            )
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return _error_result(config, ExitCode.TRANSCRIPTION, str(e), start_time)

        if not segments:
            logger.warning("No speech detected in video")
            output = config.output_path or config.input_path.with_stem(config.input_path.stem + "_dubbed")
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config.input_path, output)
            return DubResult(
                status="no_speech",
                input_video=str(config.input_path),
                output_video=str(output),
                source_language=detected_lang,
                target_language=config.target_lang,
                duration_difference_ms=0.0,
                speakers=0,
                segments_processed=0,
                subtitles_generated=False,
                lipsync_applied=False,
                error="No speech detected in video",
            )

        source_lang = detected_lang
        progress.advance(task)

        # Step 5: Diarize
        progress.update(task, description="Detecting speakers...")
        try:
            segments, speaker_profiles = diarize(audio_path, segments)
        except Exception as e:
            logger.warning("Diarization failed, using single speaker: %s", e)
            for seg in segments:
                seg.speaker = "SPEAKER_00"
            speaker_profiles = {}
        progress.advance(task)

        # Step 6: Translate
        progress.update(task, description="Translating text...")
        try:
            segments = translate_segments(
                segments, source_lang, config.target_lang, hw.translation_backend
            )
        except Exception as e:
            logger.error("Translation failed: %s", e)
            return _error_result(config, ExitCode.TRANSLATION, str(e), start_time)
        progress.advance(task)

        # Step 7: TTS
        progress.update(task, description="Generating dubbed speech...")
        seg_audio_dir = config.work_dir / "segments"
        dubbed_dir = config.work_dir / "dubbed"
        try:
            # Extract original segments for voice cloning reference
            extract_audio_segments(config.input_path, seg_audio_dir, segments)

            segments = synthesize_segments(
                segments, speaker_profiles,
                hw.tts_model, hw.device,
                dubbed_dir,
                voice_clone=config.voice_clone and hw.can_voice_clone,
                target_lang=config.target_lang,
            )
        except Exception as e:
            logger.error("TTS failed: %s", e)
            return _error_result(config, ExitCode.TTS, str(e), start_time)
        progress.advance(task)

        # Step 8: Duration matching
        progress.update(task, description="Matching durations...")
        segments = match_durations(
            segments,
            max_diff_ms=config.max_duration_diff_ms,
            speed_range=config.tts_speed_range,
        )
        segments = align_segments_to_timeline(segments, video_info.duration)
        progress.advance(task)

        # Step 9: Subtitles
        if config.subtitles:
            progress.update(task, description="Generating subtitles...")
            generate_subtitles(
                segments, config.work_dir / "subtitles",
                fmt=config.subtitle_format, mode=config.subtitle_mode,
            )
        progress.advance(task)

        # Step 10: Merge audio
        progress.update(task, description="Merging dubbed audio...")
        output_path = config.output_path or config.input_path.with_stem(config.input_path.stem + "_dubbed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            merge_dubbed_audio(
                config.input_path, dubbed_dir, segments, output_path, video_info.duration,
            )
        except Exception as e:
            logger.error("Rendering failed: %s", e)
            return _error_result(config, ExitCode.RENDERING, str(e), start_time)
        progress.advance(task)

        # Step 11: Lip-sync (optional)
        lipsync_applied = False
        if config.lipsync and hw.can_lipsync:
            progress.update(task, description="Applying lip-sync...")
            lipsync_path = config.work_dir / "lipsync_output.mp4"
            result = apply_lipsync(output_path, config.work_dir / "_final_audio.wav", lipsync_path)
            if result != output_path:
                shutil.move(str(result), str(output_path))
                lipsync_applied = True
        progress.advance(task)

        # Step 12: Finalize
        progress.update(task, description="Finalizing...")
        progress.advance(task)

    # Build result
    dur_diff = compute_total_duration_diff(segments)
    elapsed = time.time() - start_time

    result = DubResult(
        status="completed",
        input_video=str(config.input_path),
        output_video=str(output_path),
        source_language=source_lang,
        target_language=config.target_lang,
        duration_difference_ms=dur_diff,
        speakers=len(speaker_profiles),
        segments_processed=len(segments),
        subtitles_generated=config.subtitles,
        lipsync_applied=lipsync_applied,
    )

    # Write machine-readable result
    result_json = config.work_dir / "result.json"
    write_json(result_json, result.to_dict())

    console.print()
    console.print("[bold green]Dubbing complete![/]")
    console.print(f"  Output:      {output_path}")
    console.print(f"  Duration diff: {dur_diff:.1f} ms")
    console.print(f"  Speakers:    {len(speaker_profiles)}")
    console.print(f"  Segments:    {len(segments)}")
    console.print(f"  Time:        {elapsed:.1f}s")

    # Phase 1: Branding
    if config.branding and (config.brand_channel or config.brand_intro_text or config.brand_outro_text or config.brand_watermark):
        try:
            from dub.branding import BrandingConfig, apply_branding
            progress.update(task, description="Applying branding...")
            brand_config = BrandingConfig(
                channel_name=config.brand_channel,
                intro_text=config.brand_intro_text or config.brand_channel,
                outro_text=config.brand_outro_text or "Thanks for watching!",
                watermark_text=config.brand_watermark,
                watermark_position=config.brand_watermark_pos,
            )
            branded_path = output_path.parent / (output_path.stem + "_branded.mp4")
            apply_branding(output_path, branded_path, brand_config)
            output_path = branded_path
            result.output_video = str(output_path)
            result.branded = True
            console.print(f"  Branded:     {output_path}")
        except Exception as e:
            logger.error("Branding failed: %s", e)

    # Phase 1: Shorts
    if config.shorts:
        try:
            from dub.shorts import ShortsConfig, cut_multiple_shorts
            progress.update(task, description="Generating Shorts...")
            shorts_config = ShortsConfig(max_duration=config.shorts_max_duration)
            shorts_dir = output_path.parent / (output_path.stem + "_shorts")
            shorts_paths = cut_multiple_shorts(output_path, shorts_dir, config.shorts_count, shorts_config)
            result.shorts_generated = [str(p) for p in shorts_paths]
            console.print(f"  Shorts:      {len(shorts_paths)} generated in {shorts_dir}")
        except Exception as e:
            logger.error("Shorts generation failed: %s", e)

    # Phase 1: YouTube upload
    if config.upload_youtube:
        try:
            from dub.upload import UploadConfig, upload_video
            progress.update(task, description="Uploading to YouTube...")
            upload_config = UploadConfig(
                title=config.upload_title,
                description=config.upload_description,
                tags=config.upload_tags,
                privacy_status=config.upload_privacy,
                category_id=config.upload_category,
                client_secrets_path=config.upload_client_secret,
            )
            upload_result = upload_video(output_path, upload_config)
            result.youtube_upload_url = upload_result.url
            console.print(f"  YouTube:     {upload_result.url}")
        except Exception as e:
            logger.error("YouTube upload failed: %s", e)

    if not config.keep_intermediate:
        shutil.rmtree(config.work_dir, ignore_errors=True)

    return result


def _error_result(config: DubConfig, code: ExitCode, error: str, start_time: float) -> DubResult:
    logger.error("Pipeline failed: %s", error)
    return DubResult(
        status="failed",
        input_video=str(config.input_path),
        output_video="",
        source_language=config.source_lang,
        target_language=config.target_lang,
        duration_difference_ms=0.0,
        speakers=0,
        segments_processed=0,
        subtitles_generated=False,
        lipsync_applied=False,
        exit_code=code,
        error=error,
    )
