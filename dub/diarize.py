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
    """Assign speaker labels to transcription segments with gender detection.

    Strategy:
    1. Try pyannote (requires CUDA) — best accuracy
    2. Fall back to pause-based diarization using transcription segment gaps
    """
    speaker_timeline = _run_pyannote(audio_path, num_speakers)

    speaker_profiles: dict[str, SpeakerProfile] = {}

    if speaker_timeline:
        # Pyannote gave us a timeline — map segments to speakers
        for seg in segments:
            seg.speaker = _find_best_speaker(seg, speaker_timeline)
            if seg.speaker not in speaker_profiles:
                speaker_profiles[seg.speaker] = SpeakerProfile(speaker_id=seg.speaker)
            speaker_profiles[seg.speaker].segments.append(seg)
    else:
        # No pyannote — use pause-based speaker change detection
        _assign_speakers_by_pauses(segments)

        for seg in segments:
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


def _assign_speakers_by_pauses(segments: list[Segment], pause_threshold: float = 0.3) -> None:
    """Detect speaker changes from gaps between transcription segments.

    When there's a significant pause between segments, it's likely a speaker change.
    Assigns alternating SPEAKER_00, SPEAKER_01, etc.
    """
    if len(segments) <= 1:
        for seg in segments:
            seg.speaker = "SPEAKER_00"
        return

    speaker_id = 0
    segments[0].speaker = f"SPEAKER_{speaker_id:02d}"

    for i in range(1, len(segments)):
        gap = segments[i].start - segments[i - 1].end
        if gap >= pause_threshold:
            speaker_id = 1 - speaker_id  # Alternate between 0 and 1
            logger.debug(
                "Speaker change at %.2fs (gap=%.2fs): -> SPEAKER_%02d",
                segments[i].start, gap, speaker_id,
            )
        segments[i].speaker = f"SPEAKER_{speaker_id:02d}"


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
