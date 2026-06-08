"""Auto-generate intro/outro branding overlays using FFmpeg.

Creates text-based intro/outro clips and concatenates them with the main video.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dub.utils import logger


@dataclass
class BrandingConfig:
    """Configuration for video branding."""

    channel_name: str = ""
    intro_text: str = ""
    outro_text: str = ""
    intro_duration: float = 3.0  # seconds
    outro_duration: float = 4.0  # seconds
    font_size: int = 60
    font_color: str = "white"
    bg_color: str = "black"
    fade_duration: float = 0.5
    # Overlay settings (applied to main video)
    watermark_text: str = ""
    watermark_position: str = "bottom-right"  # top-left, top-right, bottom-left, bottom-right
    watermark_opacity: float = 0.7


def generate_intro(
    output_path: Path,
    config: BrandingConfig,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
) -> Path:
    """Generate an intro clip with channel name.

    Args:
        output_path: Where to save the intro clip.
        config: Branding configuration.
        width: Video width.
        height: Video height.
        fps: Frames per second.

    Returns:
        Path to the generated intro clip.
    """
    text = config.intro_text or config.channel_name or "Welcome"
    duration = config.intro_duration

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-r", str(fps),
        "-i", f"color=c={config.bg_color}:s={width}x{height}:d={duration}",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
        "-vf", (
            f"drawtext=text='{_escape_text(text)}':"
            f"fontsize={config.font_size}:"
            f"fontcolor={config.font_color}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"alpha='if(lt(t,{config.fade_duration}),t/{config.fade_duration},"
            f"if(gt(t,{duration - config.fade_duration}),(({duration}-t)/{config.fade_duration}),1))'"
        ),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(duration),
        "-shortest",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info("Generated intro: %s", output_path)
    return output_path


def generate_outro(
    output_path: Path,
    config: BrandingConfig,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
) -> Path:
    """Generate an outro clip with call-to-action text."""
    text = config.outro_text or "Thanks for watching!"
    duration = config.outro_duration

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-r", str(fps),
        "-i", f"color=c={config.bg_color}:s={width}x{height}:d={duration}",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
        "-vf", (
            f"drawtext=text='{_escape_text(text)}':"
            f"fontsize={config.font_size}:"
            f"fontcolor={config.font_color}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"alpha='if(lt(t,{config.fade_duration}),t/{config.fade_duration},"
            f"if(gt(t,{duration - config.fade_duration}),(({duration}-t)/{config.fade_duration}),1))'"
        ),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(duration),
        "-shortest",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info("Generated outro: %s", output_path)
    return output_path


def concat_videos(
    parts: list[Path],
    output_path: Path,
) -> Path:
    """Concatenate video parts (intro + main + outro) using filter_complex.

    Handles mismatched codecs by re-encoding.

    Args:
        parts: List of video file paths in order.
        output_path: Where to save the concatenated video.

    Returns:
        Path to the concatenated video.
    """
    if len(parts) == 1:
        import shutil
        shutil.copy2(parts[0], output_path)
        return output_path

    inputs = []
    filter_parts = []
    for i, part in enumerate(parts):
        inputs.extend(["-i", str(part)])
        filter_parts.append(f"[{i}:v][{i}:a]")

    n = len(parts)
    filter_complex = "".join(filter_parts) + f"concat=n={n}:v=1:a=1[v][a]"

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info("Concatenated %d parts → %s", n, output_path)
    return output_path


def apply_watermark(
    input_path: Path,
    output_path: Path,
    config: BrandingConfig,
) -> Path:
    """Apply text watermark overlay to a video.

    Args:
        input_path: Input video.
        output_path: Output video with watermark.
        config: Branding config with watermark settings.

    Returns:
        Path to watermarked video.
    """
    if not config.watermark_text:
        return input_path

    pos_map = {
        "top-left": "10:10",
        "top-right": "main_w-overlay_w-10:10",
        "bottom-left": "10:main_h-overlay_h-10",
        "bottom-right": "main_w-overlay_w-10:main_h-overlay_h-10",
    }
    pos = pos_map.get(config.watermark_position, pos_map["bottom-right"])

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", (
            f"drawtext=text='{_escape_text(config.watermark_text)}':"
            f"fontsize=24:fontcolor=white@{config.watermark_opacity}:"
            f"x={pos.split(':')[0]}:y={pos.split(':')[1]}"
        ),
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "copy",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info("Applied watermark: %s", output_path)
    return output_path


def apply_branding(
    input_path: Path,
    output_path: Path,
    config: BrandingConfig,
) -> Path:
    """Apply full branding pipeline: intro + watermark + outro.

    Args:
        input_path: Original dubbed video.
        output_path: Final branded video.
        config: Branding configuration.

    Returns:
        Path to the final branded video.
    """
    from dub.video import probe_video

    info = probe_video(input_path)
    w, h, fps = info.width, info.height, info.fps

    work_dir = output_path.parent / ".branding_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    parts = [input_path]

    # Generate intro
    if config.intro_text or config.channel_name:
        intro_path = work_dir / "intro.mp4"
        generate_intro(intro_path, config, w, h, fps)
        parts.insert(0, intro_path)

    # Generate outro
    if config.outro_text:
        outro_path = work_dir / "outro.mp4"
        generate_outro(outro_path, config, w, h, fps)
        parts.append(outro_path)

    # Concatenate
    if len(parts) > 1:
        concat_path = work_dir / "concat.mp4"
        concat_videos(parts, concat_path)
    else:
        concat_path = input_path

    # Apply watermark
    if config.watermark_text:
        apply_watermark(concat_path, output_path, config)
    else:
        import shutil
        shutil.copy2(concat_path, output_path)

    # Cleanup
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("Branding applied: %s", output_path)
    return output_path


def _escape_text(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter."""
    return text.replace("'", "'\\''").replace(":", "\\:").replace("\\", "\\\\")


def _run_ffmpeg(cmd: list[str]) -> None:
    """Run an FFmpeg command and handle errors."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr[-1000:] if result.stderr else "unknown error"
        raise RuntimeError(f"FFmpeg failed: {stderr}")
