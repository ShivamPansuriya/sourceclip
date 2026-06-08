"""Speaker diarization using pyannote.audio."""

from __future__ import annotations

import logging
from pathlib import Path

from dub.models import Segment, SpeakerProfile

logger = logging.getLogger("dub")


def diarize(
    audio_path: Path,
    segments: list[Segment],
    num_speakers: int | None = None,
) -> tuple[list[Segment], dict[str, SpeakerProfile]]:
    """Assign speaker labels to transcription segments.

    Uses pyannote.audio for diarization, then maps speaker labels
    onto pre-existing transcription segments by temporal overlap.
    """
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
        import torch

        logger.info("Loading pyannote diarization pipeline...")
        pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=None,  # Set HF_TOKEN env var if needed
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        diarize_kwargs = {}
        if num_speakers is not None:
            diarize_kwargs["num_speakers"] = num_speakers

        logger.info("Running speaker diarization...")
        diarization = pipeline(str(audio_path), **diarize_kwargs)

        # Build speaker timeline
        speaker_timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_timeline.append((turn.start, turn.end, speaker))

    except Exception as e:
        logger.warning("Diarization failed (%s), assigning single speaker to all segments", e)
        speaker_timeline = []

    # Map speakers to segments
    speaker_profiles: dict[str, SpeakerProfile] = {}

    for seg in segments:
        if not speaker_timeline:
            seg.speaker = "SPEAKER_00"
        else:
            # Find the speaker with most overlap
            seg_mid = (seg.start + seg.end) / 2
            best_speaker = "SPEAKER_00"
            best_overlap = 0.0
            for s_start, s_end, s_id in speaker_timeline:
                overlap_start = max(seg.start, s_start)
                overlap_end = min(seg.end, s_end)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = s_id
            seg.speaker = best_speaker

        if seg.speaker not in speaker_profiles:
            speaker_profiles[seg.speaker] = SpeakerProfile(speaker_id=seg.speaker)
        speaker_profiles[seg.speaker].segments.append(seg)

    n_speakers = len(speaker_profiles)
    logger.info("Detected %d speakers", n_speakers)
    for sp_id, sp in speaker_profiles.items():
        logger.info("  %s: %d segments", sp_id, len(sp.segments))

    return segments, speaker_profiles
