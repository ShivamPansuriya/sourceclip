"""Source CC-licensed videos from YouTube.

Uses YouTube Data API v3 for search + yt-dlp for download.
Requires YOUTUBE_API_KEY env var or --api-key flag.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dub.utils import logger


@dataclass
class CCVideo:
    """A Creative Commons licensed video found on YouTube."""

    video_id: str
    title: str
    channel: str
    duration: int  # seconds
    description: str
    thumbnail: str
    url: str

    def to_dict(self) -> dict:
        return {
            "videoId": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "duration": self.duration,
            "url": self.url,
        }


def search_cc_videos(
    query: str,
    api_key: str,
    max_results: int = 10,
    max_duration: int = 600,
    min_duration: int = 30,
    order: str = "relevance",
    region_code: str = "US",
) -> list[CCVideo]:
    """Search YouTube for Creative Commons licensed videos.

    Args:
        query: Search query string.
        api_key: YouTube Data API v3 key.
        max_results: Max videos to return (1-50).
        max_duration: Max video duration in seconds.
        min_duration: Min video duration in seconds.
        order: Sort order (relevance, date, viewCount, rating).
        region_code: ISO 3166-1 alpha-2 country code.

    Returns:
        List of CCVideo objects.
    """
    try:
        from googleapiclient.discovery import build
    except ImportError:
        raise RuntimeError("pip install google-api-python-client")

    youtube = build("youtube", "v3", developerKey=api_key)

    # Search with CC license filter
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        videoLicense="creativeCommon",
        maxResults=min(max_results, 50),
        order=order,
        regionCode=region_code,
    )
    response = request.execute()

    video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
    if not video_ids:
        logger.warning("No CC videos found for query: %s", query)
        return []

    # Get detailed info (duration, etc.)
    details_req = youtube.videos().list(
        part="contentDetails,snippet",
        id=",".join(video_ids),
    )
    details = details_req.execute()

    videos = []
    for item in details.get("items", []):
        vid = item["id"]
        snippet = item["snippet"]
        content = item["contentDetails"]

        # Parse ISO 8601 duration (PT1H2M3S → seconds)
        duration = _parse_iso_duration(content.get("duration", "PT0S"))

        if duration < min_duration or duration > max_duration:
            continue

        videos.append(
            CCVideo(
                video_id=vid,
                title=snippet.get("title", ""),
                channel=snippet.get("channelTitle", ""),
                duration=duration,
                description=snippet.get("description", "")[:500],
                thumbnail=snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                url=f"https://www.youtube.com/watch?v={vid}",
            )
        )

    logger.info("Found %d CC videos (filtered from %d results)", len(videos), len(video_ids))
    return videos[:max_results]


def download_video(
    video: CCVideo | str,
    output_dir: Path,
    quality: str = "best",
) -> Path:
    """Download a YouTube video using yt-dlp.

    Args:
        video: CCVideo object or video URL/ID string.
        output_dir: Directory to save the video.
        quality: Video quality (best, 720p, 480p).

    Returns:
        Path to downloaded video file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(video, CCVideo):
        url = video.url
        filename = f"{video.video_id}.mp4"
    else:
        url = video if "youtube.com" in video or "youtu.be" in video else f"https://www.youtube.com/watch?v={video}"
        filename = "%(id)s.%(ext)s"

    output_template = str(output_dir / filename)

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--format", _quality_to_format(quality),
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-playlist",
        "--no-overwrites",
        "--print", "after_move:filepath",
        url,
    ]

    logger.info("Downloading: %s", url)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[-500:]}")

    # Parse output path from yt-dlp
    filepath = result.stdout.strip().split("\n")[-1]
    if filepath and Path(filepath).exists():
        logger.info("Downloaded: %s", filepath)
        return Path(filepath)

    # Fallback: find the file
    for f in output_dir.iterdir():
        if f.suffix == ".mp4" and f.stat().st_size > 0:
            return f

    raise FileNotFoundError(f"Download succeeded but file not found in {output_dir}")


def download_batch(
    videos: list[CCVideo],
    output_dir: Path,
    quality: str = "best",
) -> list[Path]:
    """Download multiple CC videos.

    Returns:
        List of paths to downloaded files.
    """
    downloaded = []
    for i, video in enumerate(videos, 1):
        try:
            logger.info("[%d/%d] Downloading: %s", i, len(videos), video.title)
            path = download_video(video, output_dir, quality)
            downloaded.append(path)
        except Exception as e:
            logger.error("Failed to download %s: %s", video.video_id, e)
    return downloaded


def _parse_iso_duration(iso: str) -> int:
    """Parse ISO 8601 duration (PT1H2M3S) to seconds."""
    import re

    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _quality_to_format(quality: str) -> str:
    """Convert quality string to yt-dlp format selector."""
    mapping = {
        "best": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "720p": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
        "480p": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]",
    }
    return mapping.get(quality, mapping["best"])
