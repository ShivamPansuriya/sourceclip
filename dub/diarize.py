"""Speaker diarization using pyannote.audio with gender detection."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from dub.models import Segment, SpeakerProfile

logger = logging.getLogger("dub")

# F0 (fundamental frequency) thresholds for gender detection
F0_MALE_MAX = 165.0  # Hz — below this is likely male
F0_FEMALE_MIN = 165.0  # Hz — above this is likely female


def diarize(
    audio_path: Path,
    segments: list[Segment],
    num_speakers: int | None = None,
) -> tuple[list[Segment], dict[str, SpeakerProfile]]:
    """Assign speaker labels to transcription segments with gender detection.

    Pipeline:
    1. Run pyannote speaker diarization → who spoke when
    2. Map speakers onto transcription segments by temporal overlap
    3. Detect gender per speaker using pitch (F0) analysis
    """
    speaker_timeline = _run_diarization(audio_path, num_speakers)

    # Map speakers to segments
    speaker_profiles: dict[str, SpeakerProfile] = {}

    for seg in segments:
        if not speaker_timeline:
            seg.speaker = "SPEAKER_00"
        else:
            seg.speaker = _find_best_speaker(seg, speaker_timeline)

        if seg.speaker not in speaker_profiles:
            speaker_profiles[seg.speaker] = SpeakerProfile(speaker_id=seg.speaker)
        speaker_profiles[seg.speaker].segments.append(seg)

    # Detect gender per speaker
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
    """Run pyannote diarization and return speaker timeline."""
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
        import torch

        logger.info("Loading pyannote diarization pipeline...")
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=None,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        diarize_kwargs = {}
        if num_speakers is not None:
            diarize_kwargs["num_speakers"] = num_speakers

        logger.info("Running speaker diarization...")
        diarization = pipeline(str(audio_path), **diarize_kwargs)

        speaker_timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_timeline.append((turn.start, turn.end, speaker))

        return speaker_timeline

    except Exception as e:
        logger.warning("Diarization failed (%s), using single speaker", e)
        return []


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
    """Detect gender for each speaker using pitch (F0) analysis.

    Extracts each speaker's longest utterance, analyzes fundamental frequency,
    and classifies as male (<165Hz) or female (>165Hz).
    """
    if not speaker_profiles:
        return

    logger.info("Detecting speaker genders via pitch analysis...")

    for sp_id, profile in speaker_profiles.items():
        longest = profile.longest_segment
        if not longest or longest.original_duration < 0.5:
            continue

        # Extract this speaker's audio segment as temp file
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
    """Analyze fundamental frequency (F0) to determine gender.

    Returns (gender, confidence) tuple.
    """
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(audio_path), sr=22050)

        # Extract pitch using pyin (probabilistic YIN)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=65, fmax=400, sr=sr,
        )

        # Use only voiced frames (where pitch was detected)
        voiced_f0 = f0[~np.isnan(f0)]

        if len(voiced_f0) < 5:
            return ("unknown", 0.0)

        mean_f0 = float(np.median(voiced_f0))  # Median is more robust than mean

        # Classify gender based on F0
        if mean_f0 < F0_MALE_MAX:
            # Male: how confident?
            confidence = min(95, 70 + (F0_MALE_MAX - mean_f0) * 0.5)
            return ("male", confidence)
        else:
            # Female: how confident?
            confidence = min(95, 70 + (mean_f0 - F0_FEMALE_MIN) * 0.3)
            return ("female", confidence)

    except ImportError:
        logger.warning("librosa not available for pitch analysis")
        return ("unknown", 0.0)
    except Exception as e:
        logger.warning("Pitch analysis failed: %s", e)
        return ("unknown", 0.0)
