"""Duration matching and audio alignment."""

from __future__ import annotations

import logging
import subprocess
import wave
from pathlib import Path

from dub.models import Segment

logger = logging.getLogger("dub")


def match_durations(
    segments: list[Segment],
    max_diff_ms: float = 100.0,
    speed_range: tuple[float, float] = (0.5, 2.0),
) -> list[Segment]:
    """Adjust dubbed audio durations to match original durations.

    Pipeline:
    1. Check if duration already within tolerance
    2. Speed up or slow down via atempo
    3. If still out of tolerance, truncate or pad with silence
    """
    for seg in segments:
        if not seg.dubbed_audio_path or not seg.dubbed_audio_path.exists():
            continue

        orig_dur = seg.original_duration
        dubbed_dur = seg.dubbed_duration
        diff_ms = abs(orig_dur - dubbed_dur) * 1000

        if diff_ms <= max_diff_ms:
            seg.duration_match_ok = True
            logger.debug("Segment %d: within tolerance (%.1f ms)", seg.id, diff_ms)
            continue

        # Step 1: Speed adjustment
        if dubbed_dur > 0 and orig_dur > 0:
            speed_factor = dubbed_dur / orig_dur
            speed_factor = max(speed_range[0], min(speed_range[1], speed_factor))

            adjusted_path = seg.dubbed_audio_path.parent / f"dubbed_{seg.id:04d}_adj.wav"
            _adjust_speed(seg.dubbed_audio_path, adjusted_path, speed_factor)
            seg.dubbed_audio_path = adjusted_path

            with wave.open(str(adjusted_path), "rb") as wf:
                seg.dubbed_duration = wf.getnframes() / wf.getframerate()

        # Step 2: If still out of tolerance, truncate or pad
        new_diff_ms = abs(orig_dur - seg.dubbed_duration) * 1000
        if new_diff_ms > max_diff_ms:
            final_path = seg.dubbed_audio_path.parent / f"dubbed_{seg.id:04d}_final.wav"
            _truncate_or_pad(seg.dubbed_audio_path, final_path, orig_dur)
            seg.dubbed_audio_path = final_path

            with wave.open(str(final_path), "rb") as wf:
                seg.dubbed_duration = wf.getnframes() / wf.getframerate()

        new_diff_ms = abs(orig_dur - seg.dubbed_duration) * 1000
        seg.duration_match_ok = new_diff_ms <= max_diff_ms
        logger.info(
            "Segment %d: final diff %.1f ms (was %.1f ms)",
            seg.id, new_diff_ms, diff_ms,
        )

    return segments


def _adjust_speed(input_path: Path, output_path: Path, speed_factor: float) -> None:
    """Adjust audio speed using FFmpeg atempo filter."""
    remaining = speed_factor
    filters = []
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")

    filter_str = ",".join(filters)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter:a", filter_str,
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _truncate_or_pad(input_path: Path, output_path: Path, target_duration: float) -> None:
    """Truncate or pad audio to exact target duration using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", f"apad=whole_dur={target_duration}",
        "-t", str(target_duration),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def align_segments_to_timeline(
    segments: list[Segment],
    total_duration: float,
) -> list[Segment]:
    """Ensure segments don't overlap and fit within total duration."""
    for i, seg in enumerate(segments):
        if seg.end > total_duration:
            seg.end = total_duration
        if i > 0 and seg.start < segments[i - 1].end:
            seg.start = segments[i - 1].end
    return segments


def compute_total_duration_diff(segments: list[Segment]) -> float:
    """Compute max absolute duration difference across all segments in ms."""
    if not segments:
        return 0.0
    diffs = [abs(s.original_duration - s.dubbed_duration) * 1000 for s in segments]
    return max(diffs) if diffs else 0.0
