"""Auto-cut videos to vertical Shorts format (9:16, ≤60s).

Uses FFmpeg for cropping/scaling and scene detection for finding the best segment.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dub.utils import logger


@dataclass
class ShortsConfig:
    """Configuration for Shorts generation."""

    max_duration: float = 60.0  # seconds
    target_width: int = 1080
    target_height: int = 1920
    crop_mode: str = "center"  # center | top | smart
    min_scene_activity: float = 0.3  # 0-1, higher = more scene changes needed


def find_best_segment(
    video_path: Path,
    target_duration: float = 60.0,
) -> tuple[float, float]:
    """Find the most visually active segment in a video.

    Uses FFmpeg scene detection to find the segment with the most
    visual activity (scene changes), which is typically the most engaging.

    Args:
        video_path: Path to the video.
        target_duration: Desired segment duration.

    Returns:
        Tuple of (start_time, end_time) in seconds.
    """
    # Get total duration
    from dub.video import probe_video
    info = probe_video(video_path)
    total = info.duration

    if total <= target_duration:
        return (0.0, total)

    # Use FFmpeg scene detection
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", "select='gt(scene,0.3)',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # Parse scene change timestamps from showinfo output
    scene_times = []
    for line in result.stderr.split("\n"):
        if "showinfo" in line and "pts_time:" in line:
            try:
                pts_part = line.split("pts_time:")[1].split()[0]
                t = float(pts_part)
                scene_times.append(t)
            except (IndexError, ValueError):
                continue

    if not scene_times:
        # No scene changes detected, use middle segment
        start = max(0, (total - target_duration) / 2)
        return (start, start + target_duration)

    # Find the window with most scene changes
    best_start = 0.0
    best_count = 0

    for i, t in enumerate(scene_times):
        window_end = t + target_duration
        count = sum(1 for st in scene_times if t <= st <= window_end)
        if count > best_count:
            best_count = count
            best_start = t

    # Clamp to valid range
    best_start = max(0, min(best_start, total - target_duration))
    return (best_start, best_start + target_duration)


def cut_shorts(
    video_path: Path,
    output_path: Path,
    config: ShortsConfig | None = None,
) -> Path:
    """Cut a video into vertical Shorts format.

    Args:
        video_path: Input video (any aspect ratio).
        output_path: Output path for the Shorts video.
        config: Shorts configuration.

    Returns:
        Path to the generated Shorts video.
    """
    config = config or ShortsConfig()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find best segment
    start, end = find_best_segment(video_path, config.max_duration)
    duration = end - start
    logger.info("Best segment: %.1fs - %.1fs (%.1fs)", start, end, duration)

    # Build FFmpeg filter for 9:16 conversion
    if config.crop_mode == "top":
        # Crop from top (good for talking heads)
        vf = (
            f"crop=ih*{config.target_width}/{config.target_height}:ih:0:0,"
            f"scale={config.target_width}:{config.target_height}"
        )
    elif config.crop_mode == "smart":
        # Center crop (most common)
        vf = (
            f"crop=ih*{config.target_width}/{config.target_height}:ih,"
            f"scale={config.target_width}:{config.target_height}"
        )
    else:
        # Center crop (default)
        vf = (
            f"crop=ih*{config.target_width}/{config.target_height}:ih,"
            f"scale={config.target_width}:{config.target_height}"
        )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info("Generated Shorts: %s (%.1fs)", output_path, duration)
    return output_path


def cut_multiple_shorts(
    video_path: Path,
    output_dir: Path,
    count: int = 3,
    config: ShortsConfig | None = None,
) -> list[Path]:
    """Generate multiple Shorts from different segments of a video.

    Args:
        video_path: Input video.
        output_dir: Directory for output Shorts.
        count: Number of Shorts to generate.
        config: Shorts configuration.

    Returns:
        List of paths to generated Shorts.
    """
    config = config or ShortsConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    from dub.video import probe_video
    info = probe_video(video_path)
    total = info.duration

    if total <= config.max_duration:
        # Video is already short enough
        out = output_dir / f"short_1.mp4"
        cut_shorts(video_path, out, config)
        return [out]

    # Find multiple non-overlapping segments
    segments = _find_multiple_segments(video_path, count, config.max_duration)

    shorts = []
    for i, (start, end) in enumerate(segments, 1):
        out = output_dir / f"short_{i}.mp4"
        duration = end - start

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", str(video_path),
            "-vf", (
                f"crop=ih*{config.target_width}/{config.target_height}:ih,"
                f"scale={config.target_width}:{config.target_height}"
            ),
            "-c:v", "libx264", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out),
        ]

        _run_ffmpeg(cmd)
        shorts.append(out)
        logger.info("Generated short_%d: %.1fs - %.1fs", i, start, end)

    return shorts


def _find_multiple_segments(
    video_path: Path,
    count: int,
    segment_duration: float,
) -> list[tuple[float, float]]:
    """Find multiple non-overlapping active segments."""
    from dub.video import probe_video
    info = probe_video(video_path)
    total = info.duration

    if total <= segment_duration:
        return [(0.0, total)]

    # Scene detection
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", "select='gt(scene,0.3)',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    scene_times = []
    for line in result.stderr.split("\n"):
        if "showinfo" in line and "pts_time:" in line:
            try:
                pts_part = line.split("pts_time:")[1].split()[0]
                t = float(pts_part)
                scene_times.append(t)
            except (IndexError, ValueError):
                continue

    if not scene_times:
        # Divide evenly
        step = total / (count + 1)
        return [(step * i, min(step * i + segment_duration, total)) for i in range(1, count + 1)]

    # Score windows and pick top non-overlapping
    scored = []
    for t in scene_times:
        window_end = min(t + segment_duration, total)
        actual_dur = window_end - t
        if actual_dur < segment_duration * 0.5:
            continue
        activity = sum(1 for st in scene_times if t <= st <= window_end)
        scored.append((activity, t, window_end))

    scored.sort(reverse=True)

    segments = []
    for _, start, end in scored:
        if len(segments) >= count:
            break
        # Check overlap with existing segments
        overlap = False
        for es, ee in segments:
            if not (end <= es or start >= ee):
                overlap = True
                break
        if not overlap:
            segments.append((start, end))

    segments.sort(key=lambda x: x[0])
    return segments


def _run_ffmpeg(cmd: list[str]) -> None:
    """Run an FFmpeg command and handle errors."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr[-1000:] if result.stderr else "unknown error"
        raise RuntimeError(f"FFmpeg failed: {stderr}")
