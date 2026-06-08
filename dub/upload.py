"""Upload videos to YouTube via OAuth2.

Requires Google Cloud project with YouTube Data API v3 enabled
and OAuth2 credentials (client_secret.json).
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from dub.utils import logger


@dataclass
class UploadConfig:
    """Configuration for YouTube upload."""

    title: str = ""
    description: str = ""
    tags: list[str] | None = None
    category_id: str = "22"  # 22 = People & Blogs
    privacy_status: str = "private"  # private | unlisted | public
    client_secrets_path: str = "client_secret.json"
    credentials_path: str = ".yt_credentials.pickle"
    default_language: str = "en"
    embeddable: bool = True
    public_stats_viewable: bool = True


@dataclass
class UploadResult:
    """Result of a YouTube upload."""

    video_id: str
    url: str
    title: str
    status: str

    def to_dict(self) -> dict:
        return {
            "videoId": self.video_id,
            "url": self.url,
            "title": self.title,
            "status": self.status,
        }


def get_youtube_client(config: UploadConfig):
    """Authenticate and return a YouTube API client.

    Uses OAuth2 with offline access. First run opens browser for auth,
    subsequent runs use cached credentials.
    """
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise RuntimeError("pip install google-api-python-client google-auth-oauthlib")

    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    creds = None
    credentials_path = Path(config.credentials_path)

    # Load existing credentials
    if credentials_path.exists():
        try:
            with open(credentials_path, "rb") as f:
                creds = pickle.load(f)
        except Exception:
            creds = None

    # Refresh or get new credentials
    if creds and creds.valid:
        pass
    elif creds and creds.expired and creds.refresh_token:
        try:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
        except Exception:
            creds = None

    if not creds or not creds.valid:
        secrets_path = Path(config.client_secrets_path)
        if not secrets_path.exists():
            raise FileNotFoundError(
                f"OAuth client secrets not found: {secrets_path}\n"
                "Download from Google Cloud Console → APIs & Services → Credentials"
            )

        flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
        creds = flow.run_local_server(port=0)

        # Cache credentials
        with open(credentials_path, "wb") as f:
            pickle.dump(creds, f)
        logger.info("Credentials cached to %s", credentials_path)

    return build("youtube", "v3", credentials=creds)


def upload_video(
    video_path: Path,
    config: UploadConfig | None = None,
) -> UploadResult:
    """Upload a video to YouTube.

    Args:
        video_path: Path to the video file.
        config: Upload configuration.

    Returns:
        UploadResult with video ID and URL.
    """
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        raise RuntimeError("pip install google-api-python-client")

    config = config or UploadConfig()
    youtube = get_youtube_client(config)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Build request body
    body = {
        "snippet": {
            "title": config.title or video_path.stem,
            "description": config.description,
            "tags": config.tags or [],
            "categoryId": config.category_id,
            "defaultLanguage": config.default_language,
        },
        "status": {
            "privacyStatus": config.privacy_status,
            "selfDeclaredMadeForKids": False,
            "embeddable": config.embeddable,
            "publicStatsViewable": config.public_stats_viewable,
        },
    }

    # Upload with resumable upload
    file_size = video_path.stat().st_size
    media = MediaFileUpload(
        str(video_path),
        chunksize=10 * 1024 * 1024,  # 10MB chunks
        resumable=True,
    )

    logger.info("Uploading %s (%.1f MB)...", video_path.name, file_size / 1e6)

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            logger.info("Upload progress: %d%%", pct)

    video_id = response["id"]
    url = f"https://www.youtube.com/watch?v={video_id}"

    logger.info("Upload complete: %s", url)

    return UploadResult(
        video_id=video_id,
        url=url,
        title=body["snippet"]["title"],
        status="uploaded",
    )


def upload_batch(
    video_paths: list[Path],
    config: UploadConfig | None = None,
) -> list[UploadResult]:
    """Upload multiple videos to YouTube.

    Returns:
        List of UploadResult objects.
    """
    results = []
    for i, path in enumerate(video_paths, 1):
        try:
            logger.info("[%d/%d] Uploading: %s", i, len(video_paths), path.name)
            result = upload_video(path, config)
            results.append(result)
        except Exception as e:
            logger.error("Failed to upload %s: %s", path.name, e)
            results.append(UploadResult(
                video_id="",
                url="",
                title=path.stem,
                status=f"failed: {e}",
            ))
    return results
