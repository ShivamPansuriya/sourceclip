"""Background music mixing with ducking and volume control.

Mixes royalty-free background music with dubbed audio, automatically
ducking music volume when speech is present.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from dub.utils import logger


@dataclass
class MusicConfig:
    """Configuration for background music mixing."""

    music_path: Path | None = None  # Path to music file
    music_volume: float = 0.15  # Background music volume (0.0 - 1.0)
    duck_volume: float = 0.05  # Volume when speech is present
    fade_in: float = 2.0  # Fade in duration (seconds)
    fade_out: float = 3.0  # Fade out duration (seconds)
    loop: bool = True  # Loop music if shorter than video
    normalize: bool = True  # Normalize music volume


def mix_background_music(
    input_path: Path,
    output_path: Path,
    config: MusicConfig,
    video_duration: float | None = None,
) -> Path:
    """Mix background music with a video's audio track.

    Uses FFmpeg's sidechaincompress for automatic ducking —
    music volume drops when speech is detected.

    Args:
        input_path: Input video with speech audio.
        output_path: Output video with mixed music.
        config: Music configuration.
        video_duration: Video duration in seconds (for loop calculation).

    Returns:
        Path to mixed video.
    """
    if not config.music_path or not config.music_path.exists():
        import shutil
        shutil.copy2(input_path, output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get video duration if not provided
    if video_duration is None:
        from dub.video import probe_video
        info = probe_video(input_path)
        video_duration = info.duration

    # Build FFmpeg command with sidechain compression for ducking
    # Music is input[1], speech is input[0]
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),  # Video + speech audio
        "-i", str(config.music_path),  # Background music
    ]

    # Build filter complex
    filters = []

    # Loop music if needed
    if config.loop:
        loops = int(video_duration / _get_audio_duration(config.music_path)) + 1
        filters.append(f"[1:a]aloop=loop={loops}:size=2e9[looped]")
        music_label = "looped"
    else:
        music_label = "1:a"

    # Apply volume to music
    filters.append(f"[{music_label}]volume={config.music_volume}[quiet]")

    # Apply fade in/out to music
    fade_start = max(0, video_duration - config.fade_out)
    filters.append(
        f"[quiet]afade=t=in:st=0:d={config.fade_in},"
        f"afade=t=out:st={fade_start}:d={config.fade_out}[music]"
    )

    # Sidechain compression: duck music when speech is present
    filters.append(
        f"[music][0:a]sidechaincompress=threshold=0.1:ratio=4:attack=50:release=200[ducked]"
    )

    # Mix speech (louder) with ducked music
    filters.append(
        f"[0:a][ducked]amix=inputs=2:duration=first:weights=1 0.3[out]"
    )

    filter_complex = ";".join(filters)

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v",  # Keep video from input
        "-map", "[out]",  # Use mixed audio
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    ])

    _run_ffmpeg(cmd)
    logger.info("Mixed background music → %s (vol=%.0f%%)", output_path, config.music_volume * 100)
    return output_path


def mix_music_simple(
    input_path: Path,
    output_path: Path,
    music_path: Path,
    music_volume: float = 0.15,
) -> Path:
    """Simple music mixing without sidechain (lighter weight).

    Just overlays music at a fixed volume, trimmed to video length.

    Args:
        input_path: Input video.
        output_path: Output video with music.
        music_path: Path to music file.
        music_volume: Music volume (0.0 - 1.0).

    Returns:
        Path to mixed video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-i", str(music_path),
        "-filter_complex",
        f"[1:a]volume={music_volume},afade=t=in:d=2,afade=t=out:st=999:d=3[music];"
        f"[0:a][music]amix=inputs=2:duration=first:weights=1 0.3[out]",
        "-map", "0:v",
        "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info("Simple music mix → %s", output_path)
    return output_path


def add_intro_music(
    video_path: Path,
    music_path: Path,
    output_path: Path,
    music_duration: float = 3.0,
    music_volume: float = 0.3,
) -> Path:
    """Add music only to the first N seconds of a video (for intros).

    Args:
        video_path: Input video.
        music_path: Music file.
        output_path: Output video.
        music_duration: How many seconds of music to overlay.
        music_volume: Music volume.

    Returns:
        Path to output video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(music_path),
        "-filter_complex",
        f"[1:a]volume={music_volume},atrim=0:{music_duration},afade=t=out:st={music_duration-1}:d=1[music];"
        f"[0:a][music]amix=inputs=2:duration=first:weights=1 0.5[out]",
        "-map", "0:v",
        "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ]

    _run_ffmpeg(cmd)
    logger.info("Added intro music → %s", output_path)
    return output_path


def _get_audio_duration(audio_path: Path) -> float:
    """Get duration of an audio file in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 180.0  # Default 3 minutes


def _run_ffmpeg(cmd: list[str]) -> None:
    """Run FFmpeg and handle errors."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr[-1000:] if result.stderr else "unknown error"
        raise RuntimeError(f"FFmpeg failed: {stderr}")
