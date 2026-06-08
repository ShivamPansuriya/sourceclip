"""Auto-generate text overlays and fact cards using FFmpeg.

Supports timed text popups, lower thirds, fact cards, and animated text.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from dub.utils import logger


@dataclass
class TextOverlay:
    """A single text overlay with timing and style."""

    text: str
    start: float  # seconds
    end: float  # seconds
    position: str = "center"  # center | top | bottom | top-left | top-right | bottom-left | bottom-right
    font_size: int = 48
    font_color: str = "white"
    bg_color: str = ""  # empty = no background
    bg_opacity: float = 0.7
    animation: str = "fade"  # fade | none
    fade_duration: float = 0.3


@dataclass
class FactCard:
    """A fact card overlay (text box with background)."""

    title: str
    fact: str
    start: float
    end: float
    position: str = "bottom"  # bottom | center
    width_ratio: float = 0.8  # width as ratio of video width
    font_size: int = 36
    title_color: str = "#FFD700"
    text_color: str = "white"
    bg_color: str = "#000000CC"


@dataclass
class OverlaysConfig:
    """Configuration for overlay generation."""

    overlays: list[TextOverlay] = field(default_factory=list)
    fact_cards: list[FactCard] = field(default_factory=list)
    subtitle_overlay: bool = False  # Burn subtitles into video
    subtitle_path: Path | None = None


def apply_text_overlays(
    input_path: Path,
    output_path: Path,
    overlays: list[TextOverlay],
) -> Path:
    """Apply text overlays to a video using FFmpeg drawtext filter.

    Args:
        input_path: Input video.
        output_path: Output video with overlays.
        overlays: List of TextOverlay objects.

    Returns:
        Path to output video.
    """
    if not overlays:
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build drawtext filter chain
    filters = []
    for ov in overlays:
        pos = _get_position(ov.position)
        alpha_expr = _get_fade_alpha(ov.start, ov.end, ov.fade_duration)

        filter_parts = [
            f"drawtext=text='{_escape(ov.text)}'",
            f"fontsize={ov.font_size}",
            f"fontcolor={ov.font_color}",
            f"x={pos[0]}",
            f"y={pos[1]}",
            f"alpha='{alpha_expr}'",
        ]

        if ov.bg_color:
            filter_parts.append(f"box=1")
            filter_parts.append(f"boxcolor={ov.bg_color}@{ov.bg_opacity}")
            filter_parts.append(f"boxborderw=10")

        filters.append(":".join(filter_parts))

    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", filter_str,
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "copy",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info("Applied %d text overlays → %s", len(overlays), output_path)
    return output_path


def apply_fact_cards(
    input_path: Path,
    output_path: Path,
    cards: list[FactCard],
    video_width: int = 1920,
    video_height: int = 1080,
) -> Path:
    """Apply fact card overlays to a video.

    Fact cards appear as styled text boxes with title + description.

    Args:
        input_path: Input video.
        output_path: Output video with fact cards.
        cards: List of FactCard objects.
        video_width: Video width for positioning.
        video_height: Video height for positioning.

    Returns:
        Path to output video.
    """
    if not cards:
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    filters = []
    for card in cards:
        # Position calculation
        if card.position == "bottom":
            y_pos = f"main_h-{int(video_height * 0.25)}"
        else:
            y_pos = f"(main_h-text_h)/2"

        x_pos = f"(main_w-text_w)/2"

        # Title overlay
        title_alpha = _get_fade_alpha(card.start, card.end, 0.3)
        filters.append(
            f"drawtext=text='{_escape(card.title)}':"
            f"fontsize={card.font_size + 8}:"
            f"fontcolor={card.title_color}:"
            f"x={x_pos}:y={y_pos}:"
            f"alpha='{title_alpha}':"
            f"box=1:boxcolor={card.bg_color}:boxborderw=15"
        )

        # Fact text overlay (below title)
        fact_y = f"{y_pos}+{card.font_size + 20}"
        filters.append(
            f"drawtext=text='{_escape(card.fact)}':"
            f"fontsize={card.font_size}:"
            f"fontcolor={card.text_color}:"
            f"x={x_pos}:y={fact_y}:"
            f"alpha='{title_alpha}':"
            f"box=1:boxcolor={card.bg_color}:boxborderw=15"
        )

    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", filter_str,
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "copy",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info("Applied %d fact cards → %s", len(cards), output_path)
    return output_path


def burn_subtitles(
    input_path: Path,
    output_path: Path,
    subtitle_path: Path,
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
    outline_width: int = 2,
) -> Path:
    """Burn subtitles into video using FFmpeg subtitles filter.

    Args:
        input_path: Input video.
        output_path: Output video with burned subtitles.
        subtitle_path: Path to SRT/ASS subtitle file.
        font_size: Subtitle font size.
        font_color: Subtitle font color.
        outline_color: Text outline color.
        outline_width: Text outline width.

    Returns:
        Path to output video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Escape path for FFmpeg filter
    sub_path_escaped = str(subtitle_path).replace("'", "'\\''").replace(":", "\\:")

    style = f"FontSize={font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline={outline_width}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"subtitles='{sub_path_escaped}':force_style='{style}'",
        "-c:v", "libx264", "-crf", "23",
        "-c:a", "copy",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info("Burned subtitles → %s", output_path)
    return output_path


def apply_all_overlays(
    input_path: Path,
    output_path: Path,
    config: OverlaysConfig,
    video_width: int = 1920,
    video_height: int = 1080,
) -> Path:
    """Apply all overlay types in sequence.

    Args:
        input_path: Input video.
        output_path: Final output.
        config: Overlays configuration.
        video_width: Video width.
        video_height: Video height.

    Returns:
        Path to final video.
    """
    import shutil
    from dub.branding import _run_ffmpeg

    work_dir = output_path.parent / ".overlays_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    current = input_path

    # Apply fact cards
    if config.fact_cards:
        cards_path = work_dir / "fact_cards.mp4"
        apply_fact_cards(current, cards_path, config.fact_cards, video_width, video_height)
        current = cards_path

    # Apply text overlays
    if config.overlays:
        overlays_path = work_dir / "overlays.mp4"
        apply_text_overlays(current, overlays_path, config.overlays)
        current = overlays_path

    # Burn subtitles
    if config.subtitle_overlay and config.subtitle_path:
        subs_path = work_dir / "subtitled.mp4"
        burn_subtitles(current, subs_path, config.subtitle_path)
        current = subs_path

    # Copy final result
    if current != output_path:
        shutil.copy2(current, output_path)

    # Cleanup
    shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("All overlays applied → %s", output_path)
    return output_path


def _get_position(position: str) -> tuple[str, str]:
    """Get x, y coordinates for a position string."""
    mapping = {
        "center": ("(w-text_w)/2", "(h-text_h)/2"),
        "top": ("(w-text_w)/2", "50"),
        "bottom": ("(w-text_w)/2", "h-text_h-50"),
        "top-left": ("50", "50"),
        "top-right": ("w-text_w-50", "50"),
        "bottom-left": ("50", "h-text_h-50"),
        "bottom-right": ("w-text_w-50", "h-text_h-50"),
    }
    return mapping.get(position, mapping["center"])


def _get_fade_alpha(start: float, end: float, fade_duration: float) -> str:
    """Generate FFmpeg alpha expression for fade in/out."""
    return (
        f"if(lt(t,{start}),0,"
        f"if(lt(t,{start + fade_duration}),(t-{start})/{fade_duration},"
        f"if(lt(t,{end - fade_duration}),1,"
        f"if(lt(t,{end}),({end}-t)/{fade_duration},"
        f"0))))"
    )


def _escape(text: str) -> str:
    """Escape special characters for FFmpeg drawtext."""
    return text.replace("'", "'\\''").replace(":", "\\:").replace("\\", "\\\\")


def _run_ffmpeg(cmd: list[str]) -> None:
    """Run FFmpeg and handle errors."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr[-1000:] if result.stderr else "unknown error"
        raise RuntimeError(f"FFmpeg failed: {stderr}")
