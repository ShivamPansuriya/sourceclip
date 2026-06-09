"""Speaker diarization with gender detection.

Uses pyannote when available (requires CUDA), falls back to energy-based VAD.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from dub.models import Segment, SpeakerProfile

logger = logging.getLogger("dub")

F0_MALE_MAX = 165.0
F0_FEMALE_MIN = 165.0


def diarize(
    audio_path: Path,
    segments: list[Segment],
    num_speakers: int | None = None,
) -> tuple[list[Segment], dict[str, SpeakerProfile]]:
    """Assign speaker labels to transcription segments with gender detection."""
    speaker_timeline = _run_diarization(audio_path, num_speakers)

    speaker_profiles: dict[str, SpeakerProfile] = {}

    for seg in segments:
        if not speaker_timeline:
            seg.speaker = "SPEAKER_00"
        else:
            seg.speaker = _find_best_speaker(seg, speaker_timeline)

        if seg.speaker not in speaker_profiles:
            speaker_profiles[seg.speaker] = SpeakerProfile(speaker_id=seg.speaker)
        speaker_profiles[seg.speaker].segments.append(seg)

    _detect_speaker_genders(audio_path, speaker_profiles)

    n_speakers = len(speaker_profiles)
    logger.info("Detected %d speakers", n_speakers)
    for sp_id, sp in speaker_profiles.items():
        logger.info(
            "  %s: %d segments, gender=%s (%.0f%% confidence)",
            sp_id, len(sp.segments), sp.gender, sp.gender_confidence,
        )

    return segments, speaker_profiles


def _run_diarization(audio_path: Path, num_speakers: int | None = None) -> list[tuple[float, float, str]]:
    """Try pyannote, fall back to energy-based diarization."""
    result = _run_pyannote(audio_path, num_speakers)
    if result is not None:
        return result
    return _run_energy_diarization(audio_path)


def _run_pyannote(audio_path: Path, num_speakers: int | None = None) -> list[tuple[float, float, str]] | None:
    """Try pyannote diarization. Returns None on any failure."""
    try:
        import os
        import numpy as np
        if not hasattr(np, "NaN"):
            np.NaN = np.nan

        from pyannote.audio import Pipeline as PyannotePipeline
        import torch
        import torchaudio

        logger.info("Loading pyannote diarization pipeline...")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        diarize_kwargs = {}
        if num_speakers is not None:
            diarize_kwargs["num_speakers"] = num_speakers

        logger.info("Running speaker diarization...")
        waveform, sample_rate = torchaudio.load(str(audio_path))
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}
        diarization = pipeline(audio_input, **diarize_kwargs)

        speaker_timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_timeline.append((turn.start, turn.end, speaker))

        return speaker_timeline

    except Exception as e:
        logger.warning("Diarization failed (%s)", e)
        return None


def _run_energy_diarization(audio_path: Path) -> list[tuple[float, float, str]]:
    """Energy-based diarization: RMS envelope → speech regions → speaker clustering."""
    logger.info("Using energy-based diarization fallback")

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_path)],
        capture_output=True, text=True, check=True,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])

    result = subprocess.run(
        ["ffmpeg", "-i", str(audio_path), "-ar", "16000", "-ac", "1",
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True, check=True,
    )

    import numpy as np
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    frame_size = 1600  # 100ms at 16kHz
    n_frames = len(samples) // frame_size

    rms = np.zeros(n_frames)
    for i in range(n_frames):
        chunk = samples[i * frame_size : (i + 1) * frame_size]
        rms[i] = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

    threshold = np.percentile(rms, 30) * 1.5
    if threshold < 100:
        threshold = 100

    is_speech = rms > threshold
    speech_regions = []
    in_speech = False
    start = 0

    for i in range(n_frames):
        if is_speech[i] and not in_speech:
            start = i
            in_speech = True
        elif not is_speech[i] and in_speech:
            speech_regions.append((start * frame_size / 16000, i * frame_size / 16000))
            in_speech = False
    if in_speech:
        speech_regions.append((start * frame_size / 16000, n_frames * frame_size / 16000))

    merged = []
    for region in speech_regions:
        if merged and region[0] - merged[-1][1] < 0.2:
            merged[-1] = (merged[-1][0], region[1])
        else:
            merged.append(region)

    if not merged:
        return []

    speakers = []
    for start, end in merged:
        sf = int(start * 16000 / frame_size)
        ef = int(end * 16000 / frame_size)
        region_energy = np.mean(rms[sf:ef])
        speaker = "SPEAKER_00" if region_energy >= np.median(rms) else "SPEAKER_01"
        speakers.append((start, end, speaker))

    n_actual = len(set(s for _, _, s in speakers))
    logger.info("Energy diarization: %d regions, %d speakers", len(speakers), n_actual)
    return speakers


def _find_best_speaker(seg: Segment, speaker_timeline: list[tuple[float, float, str]]) -> str:
    """Find the speaker with most temporal overlap for a segment."""
    best_speaker = "SPEAKER_00"
    best_overlap = 0.0

    for s_start, s_end, s_id in speaker_timeline:
        overlap_start = max(seg.start, s_start)
        overlap_end = min(seg.end, s_end)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = s_id

    return best_speaker


def _detect_speaker_genders(
    audio_path: Path,
    speaker_profiles: dict[str, SpeakerProfile],
) -> None:
    """Detect gender for each speaker using pitch (F0) analysis."""
    if not speaker_profiles:
        return

    logger.info("Detecting speaker genders via pitch analysis...")

    for sp_id, profile in speaker_profiles.items():
        longest = profile.longest_segment
        if not longest or longest.original_duration < 0.5:
            continue

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(longest.start),
                    "-t", str(longest.original_duration),
                    "-i", str(audio_path),
                    "-ar", "16000", "-ac", "1",
                    "-acodec", "pcm_s16le",
                    str(tmp_path),
                ],
                capture_output=True,
                check=True,
            )

            gender, confidence = _analyze_pitch(tmp_path)
            profile.gender = gender
            profile.gender_confidence = confidence

            logger.debug(
                "  %s: gender=%s (%.0f%%) from segment %.1f-%.1fs",
                sp_id, gender, confidence, longest.start, longest.end,
            )

        except Exception as e:
            logger.warning("Gender detection failed for %s: %s", sp_id, e)
            profile.gender = "unknown"
            profile.gender_confidence = 0.0

        finally:
            tmp_path.unlink(missing_ok=True)


def _analyze_pitch(audio_path: Path) -> tuple[str, float]:
    """Analyze fundamental frequency (F0) to determine gender."""
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(audio_path), sr=22050)
        f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        voiced_f0 = f0[~np.isnan(f0)]

        if len(voiced_f0) < 5:
            return ("unknown", 0.0)

        mean_f0 = float(np.median(voiced_f0))

        if mean_f0 < F0_MALE_MAX:
            confidence = min(95, 70 + (F0_MALE_MAX - mean_f0) * 0.5)
            return ("male", confidence)
        else:
            confidence = min(95, 70 + (mean_f0 - F0_FEMALE_MIN) * 0.3)
            return ("female", confidence)

    except ImportError:
        logger.warning("librosa not available for pitch analysis")
        return ("unknown", 0.0)
    except Exception as e:
        logger.warning("Pitch analysis failed: %s", e)
        return ("unknown", 0.0)
