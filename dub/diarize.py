"""Speaker diarization with gender detection.

Uses pyannote (works on CPU), falls back to pause-based VAD.
Gender detection uses multi-segment F0 + spectral analysis (physics-based, cross-lingual).
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from dub.models import Segment, SpeakerProfile

logger = logging.getLogger("dub")


def diarize(
    audio_path: Path,
    segments: list[Segment],
    num_speakers: int | None = None,
) -> tuple[list[Segment], dict[str, SpeakerProfile]]:
    """Assign speaker labels to transcription segments with gender detection.

    Strategy:
    1. Try pyannote — best accuracy (CPU-compatible)
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
            " %s: %d segments, gender=%s (%.0f%% confidence)",
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
            speaker_id = 1 - speaker_id
            logger.debug(
                "Speaker change at %.2fs (gap=%.2fs): -> SPEAKER_%02d",
                segments[i].start, gap, speaker_id,
            )
        segments[i].speaker = f"SPEAKER_{speaker_id:02d}"


def _run_pyannote(audio_path: Path, num_speakers: int | None = None) -> list[tuple[float, float, str]] | None:
    """Run pyannote speaker diarization. Returns None on any failure."""
    try:
        import os

        np.NaN = np.nan

        from pyannote.audio import Pipeline as PyannotePipeline
        import torch

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
        import soundfile as sf
        waveform_np, sample_rate = sf.read(str(audio_path), dtype="float32")
        if waveform_np.ndim == 1:
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        else:
            waveform = torch.from_numpy(waveform_np.T)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}
        diarization_output = pipeline(audio_input, **diarize_kwargs)

        speaker_timeline: list[tuple[float, float, str]] = []
        for turn, _, speaker in diarization_output.speaker_diarization.itertracks(yield_label=True):
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
    """Detect gender for each speaker using multi-segment F0 + spectral analysis.

    Uses physics-based analysis (vocal cord characteristics) that works cross-lingually.
    Analyzes multiple segments per speaker and takes majority vote for robustness.
    """
    if not speaker_profiles:
        return

    logger.info("Detecting speaker genders via F0 + spectral analysis...")

    for sp_id, profile in speaker_profiles.items():
        if not profile.segments:
            continue

        all_features: list[dict[str, float]] = []
        for seg in profile.segments:
            if seg.original_duration < 0.5:
                continue

            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-ss", str(seg.start),
                        "-t", str(min(seg.original_duration, 10.0)),
                        "-i", str(audio_path),
                        "-ar", "16000", "-ac", "1",
                        "-acodec", "pcm_s16le",
                        str(tmp_path),
                    ],
                    capture_output=True,
                    check=True,
                )

                features = _extract_voice_features(tmp_path)
                if features and features["f0"] > 0:
                    all_features.append(features)

            except Exception as e:
                logger.debug("Segment analysis failed: %s", e)
            finally:
                tmp_path.unlink(missing_ok=True)

        if all_features:
            avg_features = {
                "f0": np.mean([f["f0"] for f in all_features]),
                "centroid": np.mean([f["centroid"] for f in all_features]),
                "rms": np.mean([f["rms"] for f in all_features]),
                "zcr": np.mean([f["zcr"] for f in all_features]),
            }
            profile._voice_features = avg_features
        else:
            profile._voice_features = None

        _resolve_gender_by_comparison(speaker_profiles)


def _extract_median_f0(audio_path: Path) -> float | None:
    """Extract median F0 from audio file."""
    try:
        import librosa

        y, sr = librosa.load(str(audio_path), sr=22050)
        f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        voiced_f0 = f0[~np.isnan(f0)]

        if len(voiced_f0) < 5:
            return None

        return float(np.median(voiced_f0))

    except Exception:
        return None


def _extract_voice_features(audio_path: Path) -> dict[str, float] | None:
    """Extract multiple voice features for gender analysis."""
    try:
        import librosa

        y, sr = librosa.load(str(audio_path), sr=22050)

        f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        voiced_f0 = f0[~np.isnan(f0)]
        median_f0 = float(np.median(voiced_f0)) if len(voiced_f0) >= 5 else 0.0

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_centroid = float(np.mean(spectral_centroid))

        rms = librosa.feature.rms(y=y)
        mean_rms = float(np.mean(rms))

        zcr = librosa.feature.zero_crossing_rate(y)
        mean_zcr = float(np.mean(zcr))

        return {
            "f0": median_f0,
            "centroid": mean_centroid,
            "rms": mean_rms,
            "zcr": mean_zcr,
        }

    except Exception:
        return None


def _resolve_gender_by_comparison(speaker_profiles: dict[str, SpeakerProfile]) -> None:
    """Resolve gender using multi-feature comparison between speakers."""
    if len(speaker_profiles) < 2:
        return

    features_data: dict[str, dict[str, float]] = {}
    for sp_id, profile in speaker_profiles.items():
        if hasattr(profile, "_voice_features") and profile._voice_features is not None:
            features_data[sp_id] = profile._voice_features

    if len(features_data) < 2:
        return

    scores: dict[str, float] = {}
    for sp_id, features in features_data.items():
        score = 0.0

        f0 = features["f0"]
        if f0 > 200:
            score += 2.0
        elif f0 > 180:
            score += 1.0
        elif f0 < 140:
            score -= 2.0
        elif f0 < 160:
            score -= 1.0

        centroid = features["centroid"]
        if centroid > 3000:
            score += 1.5
        elif centroid > 2500:
            score += 0.5
        elif centroid < 2000:
            score -= 1.5
        elif centroid < 2500:
            score -= 0.5

        rms = features["rms"]
        if rms < 0.02:
            score += 1.0
        elif rms < 0.04:
            score += 0.5
        elif rms > 0.06:
            score -= 1.0
        elif rms > 0.04:
            score -= 0.5

        zcr = features["zcr"]
        if zcr > 0.1:
            score += 0.5
        elif zcr < 0.05:
            score -= 0.5

        scores[sp_id] = score

    sorted_speakers = sorted(scores.items(), key=lambda x: x[1])
    lowest_id, lowest_score = sorted_speakers[0]
    highest_id, highest_score = sorted_speakers[-1]
    score_diff = highest_score - lowest_score

    if score_diff > 1.0:
        logger.info("Using multi-feature comparison: scores %s",
                    {sp_id: f"{s:.1f}" for sp_id, s in scores.items()})

        for sp_id, profile in speaker_profiles.items():
            if sp_id == lowest_id:
                profile.gender = "male"
                profile.gender_confidence = min(95, 70 + score_diff * 10)
            elif sp_id == highest_id:
                profile.gender = "female"
                profile.gender_confidence = min(95, 70 + score_diff * 10)


def _analyze_voice(audio_path: Path) -> tuple[str, float]:
    """Analyze voice using F0 + spectral features for gender classification."""
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(str(audio_path), sr=22050)

        f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        voiced_f0 = f0[~np.isnan(f0)]

        if len(voiced_f0) < 5:
            return ("unknown", 0.0)

        median_f0 = float(np.median(voiced_f0))
        mean_f0 = float(np.mean(voiced_f0))

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_centroid = float(np.mean(spectral_centroid))

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        mean_rolloff = float(np.mean(spectral_rolloff))

        f0_score = 0.0
        if median_f0 < 140:
            f0_score = 1.0
        elif median_f0 < 160:
            f0_score = 0.7
        elif median_f0 < 180:
            f0_score = 0.5
        elif median_f0 < 200:
            f0_score = 0.3
        else:
            f0_score = 0.0

        centroid_score = 0.0
        if mean_centroid < 2000:
            centroid_score = 1.0
        elif mean_centroid < 2500:
            centroid_score = 0.7
        elif mean_centroid < 3000:
            centroid_score = 0.5
        elif mean_centroid < 3500:
            centroid_score = 0.3
        else:
            centroid_score = 0.0

        combined = (f0_score * 0.7) + (centroid_score * 0.3)

        if combined < 0.5:
            gender = "male"
            confidence = (1.0 - combined) * 100
        else:
            gender = "female"
            confidence = combined * 100

        confidence = max(50, min(99, confidence))

        return (gender, confidence)

    except ImportError:
        logger.warning("librosa not available for voice analysis")
        return ("unknown", 0.0)
    except Exception as e:
        logger.warning("Voice analysis failed: %s", e)
        return ("unknown", 0.0)
