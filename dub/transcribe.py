"""Transcription using faster-whisper."""

from __future__ import annotations

import logging
from pathlib import Path

from dub.models import Segment

logger = logging.getLogger("dub")


def transcribe(
    audio_path: Path,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
) -> tuple[list[Segment], str]:
    """Transcribe audio file and return segments + detected language.

    Returns:
        (segments, detected_language)
    """
    from faster_whisper import WhisperModel

    logger.info("Loading Whisper model: %s (device=%s, compute=%s)", model_size, device, compute_type)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    kwargs = {}
    if language and language != "auto":
        kwargs["language"] = language

    raw_segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        **kwargs,
    )

    detected_lang = info.language
    logger.info("Detected language: %s (prob=%.2f)", detected_lang, info.language_probability)

    segments = []
    for i, seg in enumerate(raw_segments):
        segments.append(Segment(
            id=i,
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
        ))

    logger.info("Transcribed %d segments", len(segments))
    return segments, detected_lang
