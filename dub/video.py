"""Video processing — probe, extract audio, merge dubbed audio."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dub.models import VideoInfo
from dub.utils import run_cmd

logger = logging.getLogger("dub")


def probe_video(path: Path) -> VideoInfo:
    """Extract metadata from video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    result = run_cmd(cmd, capture=True)
    info = json.loads(result.stdout)

    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio_stream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    fmt = info["format"]

    fps = 0.0
    if "r_frame_rate" in video_stream:
        num, den = video_stream["r_frame_rate"].split("/")
        fps = int(num) / int(den) if int(den) else 0.0

    return VideoInfo(
        path=path,
        duration=float(fmt["duration"]),
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=fps,
        codec=video_stream["codec_name"],
        audio_codec=audio_stream["codec_name"] if audio_stream else "none",
        audio_sample_rate=int(audio_stream["sample_rate"]) if audio_stream else 0,
        audio_channels=int(audio_stream["channels"]) if audio_stream else 0,
        file_size=int(fmt["size"]),
    )


def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 24000) -> Path:
    """Extract audio track from video as WAV."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path),
    ]
    run_cmd(cmd)
    logger.info("Audio extracted → %s", output_path)
    return output_path


def extract_audio_segments(
    video_path: Path,
    segments_dir: Path,
    segments: list,
    sample_rate: int = 24000,
) -> list[Path]:
    """Extract individual audio segments for each speech segment."""
    paths = []
    for seg in segments:
        out = segments_dir / f"seg_{seg.id:04d}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", str(seg.start),
            "-t", str(seg.end - seg.start),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            str(out),
        ]
        run_cmd(cmd)
        paths.append(out)
    logger.info("Extracted %d audio segments", len(paths))
    return paths


def merge_dubbed_audio(
    video_path: Path,
    dubbed_segments_dir: Path,
    segments: list,
    output_path: Path,
    total_duration: float,
) -> Path:
    """Replace original audio with dubbed segments, preserving silence gaps."""
    # Build a silence-padded audio track from dubbed segments
    # Strategy: create a silent base, overlay each dubbed segment at its timestamp
    work_dir = output_path.parent
    base_silence = work_dir / "_silence.wav"
    merged_audio = work_dir / "_merged.wav"

    # Create silence of total duration
    cmd_silence = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r=24000:cl=mono",
        "-t", str(total_duration),
        "-acodec", "pcm_s16le",
        str(base_silence),
    ]
    run_cmd(cmd_silence)

    # Build complex filter to overlay segments
    inputs = ["-i", str(base_silence)]
    filter_parts = []

    for seg in segments:
        if seg.dubbed_audio_path and seg.dubbed_audio_path.exists():
            inputs.extend(["-i", str(seg.dubbed_audio_path)])

    if len(inputs) <= 2:
        # No dubbed audio, just use silence
        cmd_copy = [
            "ffmpeg", "-y",
            "-i", str(base_silence),
            "-acodec", "pcm_s16le",
            str(merged_audio),
        ]
        run_cmd(cmd_copy)
    else:
        # Build filter complex: [0:a] is silence, [1:] are segments
        n_segments = len([s for s in segments if s.dubbed_audio_path and s.dubbed_audio_path.exists()])
        filter_complex = "[0:a]aformat=sample_fmts=fltp:sample_rates=24000:channel_layouts=mono[base];"

        delayed = []
        idx = 1
        for seg in segments:
            if seg.dubbed_audio_path and seg.dubbed_audio_path.exists():
                delay_ms = int(seg.start * 1000)
                filter_complex += f"[{idx}:a]aformat=sample_fmts=fltp:sample_rates=24000:channel_layouts=mono,adelay={delay_ms}|{delay_ms}[d{idx}];"
                delayed.append(f"[d{idx}]")
                idx += 1

        mix_inputs = "".join(delayed) + "[base]"
        out_label = f"[out]"
        filter_complex += f"{mix_inputs}amix=inputs={len(delayed) + 1}:duration=longest:dropout_transition=0:normalize=0{out_label}"

        cmd_merge = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", out_label,
            "-acodec", "pcm_s16le",
            str(merged_audio),
        ]
        run_cmd(cmd_merge)

    # Mux merged audio with original video (no re-encode)
    cmd_mux = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(merged_audio),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]
    run_cmd(cmd_mux)
    logger.info("Muxed dubbed audio → %s", output_path)

    # Cleanup temp files
    base_silence.unlink(missing_ok=True)
    merged_audio.unlink(missing_ok=True)

    return output_path
