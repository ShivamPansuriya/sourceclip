"""Lip-sync enhancement — optional post-processing with Wav2Lip or MuseTalk."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from dub.models import Segment

logger = logging.getLogger("dub")


def apply_lipsync(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    model: str = "wav2lip",
) -> Path:
    """Apply lip-sync to video using the generated dubbed audio.

    This is a best-effort feature. If the lip-sync model is not
    available or fails, returns the original video path unchanged.

    Args:
        video_path: Path to the dubbed video.
        audio_path: Path to the full dubbed audio track.
        output_path: Path for the lip-synced output.
        model: 'wav2lip' or 'musetalk'.

    Returns:
        Path to lip-synced video, or input video_path on failure.
    """
    if model == "wav2lip":
        return _wav2lip(video_path, audio_path, output_path)
    elif model == "musetalk":
        return _musetalk(video_path, audio_path, output_path)
    else:
        logger.warning("Unknown lip-sync model: %s", model)
        return video_path


def _wav2lip(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    """Apply Wav2Lip for lip-sync."""
    try:
        # Check if Wav2Lip is available
        wav2lip_dir = Path.home() / "Wav2Lip"
        if not wav2lip_dir.exists():
            # Try common locations
            for candidate in [Path("/opt/Wav2Lip"), Path("./Wav2Lip")]:
                if candidate.exists():
                    wav2lip_dir = candidate
                    break
            else:
                logger.warning("Wav2Lip not found. Skipping lip-sync.")
                return video_path

        checkpoint = wav2lip_dir / "checkpoints" / "wav2lip_gan.pth"
        if not checkpoint.exists():
            logger.warning("Wav2Lip checkpoint not found at %s", checkpoint)
            return video_path

        cmd = [
            "python", str(wav2lip_dir / "inference.py"),
            "--checkpoint_path", str(checkpoint),
            "--face", str(video_path),
            "--audio", str(audio_path),
            "--outfile", str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, cwd=str(wav2lip_dir))
        logger.info("Wav2Lip lip-sync applied → %s", output_path)
        return output_path

    except Exception as e:
        logger.warning("Wav2Lip failed: %s — returning original video", e)
        return video_path


def _musetalk(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    """Apply MuseTalk for lip-sync."""
    logger.warning("MuseTalk integration not yet implemented. Skipping lip-sync.")
    return video_path
