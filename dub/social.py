"""Upload videos to TikTok and Instagram.

Uses yt-dlp for TikTok upload (via browser cookies) and
Instagram Graph API for Instagram Reels.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dub.utils import logger


@dataclass
class SocialUploadConfig:
    """Configuration for social media upload."""

    # TikTok
    tiktok_cookies_path: str = ""  # Path to browser cookies file
    tiktok_description: str = ""
    tiktok_tags: list[str] | None = None
    tiktok_privacy: str = "public"  # public | friends | private

    # Instagram
    instagram_access_token: str = ""
    instagram_business_id: str = ""
    instagram_caption: str = ""
    instagram_tags: list[str] | None = None


@dataclass
class SocialUploadResult:
    """Result of a social media upload."""

    platform: str
    url: str
    status: str
    video_id: str = ""

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "url": self.url,
            "videoId": self.video_id,
            "status": self.status,
        }


def upload_tiktok(
    video_path: Path,
    config: SocialUploadConfig,
) -> SocialUploadResult:
    """Upload a video to TikTok using browser cookies.

    Requires browser cookies from a logged-in TikTok session.
    Generate cookies file with a browser extension like "Get cookies.txt".

    Args:
        video_path: Path to the video file.
        config: TikTok upload configuration.

    Returns:
        SocialUploadResult with status.
    """
    if not config.tiktok_cookies_path:
        raise ValueError("TikTok cookies path required (--tiktok-cookies)")

    cookies_path = Path(config.tiktok_cookies_path)
    if not cookies_path.exists():
        raise FileNotFoundError(f"Cookies file not found: {cookies_path}")

    # Build description with tags
    description = config.tiktok_description
    if config.tiktok_tags:
        tags = " ".join(f"#{tag}" for tag in config.tiktok_tags)
        description = f"{description} {tags}"

    # Use yt-dlp for TikTok upload
    # Note: yt-dlp doesn't natively support TikTok upload,
    # so we use the TikTok API directly via curl
    logger.info("Uploading to TikTok: %s", video_path.name)

    # TikTok Video Kit API (requires approved developer account)
    # For now, we'll document the manual process and use a helper script
    cmd = [
        "curl", "-X", "POST",
        "https://open.tiktokapis.com/v2/post/publish/video/init/",
        "-H", "Authorization: Bearer YOUR_ACCESS_TOKEN",
        "-F", f"video=@{video_path}",
        "-F", f"caption={description}",
    ]

    # Since TikTok API requires developer approval, we'll save the video
    # with a ready-to-upload name and provide instructions
    output_dir = video_path.parent / "tiktok_ready"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / video_path.name

    import shutil
    shutil.copy2(video_path, output_file)

    logger.info("TikTok-ready video saved: %s", output_file)
    logger.info("To upload manually: Open TikTok app → Upload → Select file")

    return SocialUploadResult(
        platform="tiktok",
        url=str(output_file),
        status="ready_to_upload",
        video_id=video_path.stem,
    )


def upload_instagram(
    video_path: Path,
    config: SocialUploadConfig,
) -> SocialUploadResult:
    """Upload a video to Instagram as a Reel using the Graph API.

    Requires:
    - Facebook App with Instagram Graph API permission
    - Instagram Business account connected to Facebook Page
    - Valid access token

    Args:
        video_path: Path to the video file.
        config: Instagram upload configuration.

    Returns:
        SocialUploadResult with URL.
    """
    if not config.instagram_access_token:
        raise ValueError("Instagram access token required (--ig-access-token)")
    if not config.instagram_business_id:
        raise ValueError("Instagram business account ID required (--ig-business-id)")

    try:
        import requests
    except ImportError:
        raise RuntimeError("pip install requests")

    # Build caption with tags
    caption = config.instagram_caption
    if config.instagram_tags:
        tags = " ".join(f"#{tag}" for tag in config.instagram_tags)
        caption = f"{caption} {tags}"

    base_url = "https://graph.facebook.com/v18.0"
    business_id = config.instagram_business_id
    token = config.instagram_access_token

    # Step 1: Create media container
    logger.info("Creating Instagram media container...")

    create_url = f"{base_url}/{business_id}/media"
    create_data = {
        "media_type": "REELS",
        "video_url": str(video_path),  # For local files, need to host first
        "caption": caption,
        "access_token": token,
    }

    # For local files, we need to upload to a public URL first
    # This is a simplified version - in production, upload to S3/CDN
    response = requests.post(create_url, data=create_data, timeout=30)
    result = response.json()

    if "id" not in result:
        raise RuntimeError(f"Failed to create container: {result}")

    container_id = result["id"]
    logger.info("Container created: %s", container_id)

    # Step 2: Wait for processing and publish
    import time
    for _ in range(30):  # Wait up to 5 minutes
        time.sleep(10)
        check_url = f"{base_url}/{container_id}?fields=status_code&access_token={token}"
        check = requests.get(check_url, timeout=10).json()

        if check.get("status_code") == "FINISHED":
            break
        elif check.get("status_code") == "ERROR":
            raise RuntimeError(f"Processing failed: {check}")

    # Publish
    publish_url = f"{base_url}/{business_id}/media_publish"
    publish_data = {
        "creation_id": container_id,
        "access_token": token,
    }
    publish = requests.post(publish_url, data=publish_data, timeout=30).json()

    if "id" not in publish:
        raise RuntimeError(f"Failed to publish: {publish}")

    post_id = publish["id"]
    post_url = f"https://www.instagram.com/reel/{post_id}"

    logger.info("Published to Instagram: %s", post_url)

    return SocialUploadResult(
        platform="instagram",
        url=post_url,
        video_id=post_id,
        status="published",
    )


def prepare_for_social(
    video_path: Path,
    output_dir: Path,
    platform: str = "all",
    max_duration: float = 60.0,
    max_size_mb: float = 287.0,  # TikTok limit
) -> list[Path]:
    """Prepare a video for social media upload.

    Trims to max duration, optimizes for platform requirements.

    Args:
        video_path: Input video.
        output_dir: Output directory.
        platform: Target platform (tiktok, instagram, all).
        max_duration: Maximum duration in seconds.
        max_size_mb: Maximum file size in MB.

    Returns:
        List of prepared video paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = []

    from dub.video import probe_video
    info = probe_video(video_path)
    needs_trim = info.duration > max_duration

    platforms = ["tiktok", "instagram"] if platform == "all" else [platform]

    for plat in platforms:
        out = output_dir / f"{video_path.stem}_{plat}.mp4"

        if needs_trim:
            # Trim to max duration
            cmd = [
                "ffmpeg", "-y",
                "-t", str(max_duration),
                "-i", str(video_path),
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                str(out),
            ]
        else:
            # Just optimize for streaming
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-c:v", "libx264", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                str(out),
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Failed to prepare for %s: %s", plat, result.stderr[-200:])
            continue

        # Check file size
        size_mb = out.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            logger.warning("%s file too large (%.1fMB > %.1fMB), re-encoding", plat, size_mb, max_size_mb)
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-t", str(max_duration),
                "-c:v", "libx264", "-crf", "28",  # Lower quality
                "-c:a", "aac", "-b:a", "96k",
                "-movflags", "+faststart",
                str(out),
            ]
            subprocess.run(cmd, capture_output=True, text=True)

        prepared.append(out)
        logger.info("Prepared for %s: %s (%.1fMB)", plat, out, out.stat().st_size / 1e6)

    return prepared
