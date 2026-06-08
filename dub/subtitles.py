"""Subtitle generation — SRT, VTT, ASS."""

from __future__ import annotations

import logging
from pathlib import Path

from dub.models import Segment

logger = logging.getLogger("dub")


def generate_subtitles(
    segments: list[Segment],
    output_dir: Path,
    fmt: str = "srt",
    mode: str = "translated",
) -> Path:
    """Generate subtitle file from segments.

    Args:
        segments: Processed segments with text/translated_text.
        output_dir: Directory to write subtitle file.
        fmt: Format — 'srt', 'vtt', or 'ass'.
        mode: 'original', 'translated', or 'bilingual'.

    Returns:
        Path to generated subtitle file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"subtitles.{fmt}"

    if fmt == "srt":
        _write_srt(segments, out_path, mode)
    elif fmt == "vtt":
        _write_vtt(segments, out_path, mode)
    elif fmt == "ass":
        _write_ass(segments, out_path, mode)
    else:
        raise ValueError(f"Unsupported subtitle format: {fmt}")

    logger.info("Subtitles generated → %s (format=%s, mode=%s)", out_path, fmt, mode)
    return out_path


def _get_text(seg: Segment, mode: str) -> str:
    if mode == "original":
        return seg.text
    elif mode == "bilingual":
        return f"{seg.text}\n{seg.translated_text}"
    else:
        return seg.translated_text


def _format_time_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_time_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _format_time_ass(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _write_srt(segments: list[Segment], path: Path, mode: str) -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        text = _get_text(seg, mode)
        lines.append(str(i))
        lines.append(f"{_format_time_srt(seg.start)} --> {_format_time_srt(seg.end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_vtt(segments: list[Segment], path: Path, mode: str) -> None:
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        text = _get_text(seg, mode)
        lines.append(str(i))
        lines.append(f"{_format_time_vtt(seg.start)} --> {_format_time_vtt(seg.end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ass(segments: list[Segment], path: Path, mode: str) -> None:
    header = """[Script Info]
Title: Dubbed Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for seg in segments:
        text = _get_text(seg, mode).replace("\n", "\\N")
        events.append(
            f"Dialogue: 0,{_format_time_ass(seg.start)},{_format_time_ass(seg.end)},"
            f"Default,,0,0,0,,{text}"
        )

    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
