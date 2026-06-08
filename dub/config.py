"""Configuration and CLI options."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DubConfig:
    """All configuration for a dubbing run."""

    input_path: Path
    output_path: Path | None = None
    source_lang: str = "auto"
    target_lang: str = "en"

    # Feature flags
    subtitles: bool = False
    subtitle_format: str = "srt"
    subtitle_mode: str = "translated"  # original | translated | bilingual
    voice_clone: bool = True
    lipsync: bool = False

    # Pipeline tuning
    whisper_model: str = "auto"  # auto | tiny | base | small | medium | large-v3
    tts_model: str = "auto"  # auto | xtts | cosyvoice | f5
    translation_model: str = "auto"  # auto | nllb | m2m100 | google
    max_duration_diff_ms: float = 100.0
    tts_speed_range: tuple[float, float] = (0.5, 2.0)

    # Hardware
    device: str = "auto"  # auto | cpu | cuda
    compute_type: str = "auto"  # auto | int8 | float16 | float32

    # Output
    keep_intermediate: bool = False
    verbose: bool = False
    json_output: bool = False

    # Batch mode
    recursive: bool = False

    # Phase 1: Content sourcing
    source_cc: bool = False  # Find and download CC videos from YouTube
    source_query: str = ""  # Search query for CC videos
    source_api_key: str = ""  # YouTube API key
    source_max_results: int = 5  # Max CC videos to download
    source_max_duration: int = 600  # Max video duration in seconds
    source_download_dir: Path | None = None  # Where to save downloads

    # Phase 1: Branding
    branding: bool = False  # Apply intro/outro branding
    brand_channel: str = ""  # Channel name for intro
    brand_intro_text: str = ""  # Custom intro text
    brand_outro_text: str = ""  # Custom outro text
    brand_watermark: str = ""  # Watermark text overlay
    brand_watermark_pos: str = "bottom-right"

    # Phase 1: Shorts
    shorts: bool = False  # Generate vertical Shorts
    shorts_count: int = 1  # Number of Shorts to generate
    shorts_max_duration: float = 60.0  # Max Shorts duration

    # Phase 1: YouTube upload
    upload_youtube: bool = False  # Upload to YouTube
    upload_title: str = ""
    upload_description: str = ""
    upload_tags: list[str] = field(default_factory=list)
    upload_privacy: str = "private"  # private | unlisted | public
    upload_category: str = "22"
    upload_client_secret: str = "client_secret.json"

    # Phase 2: Overlays
    overlays: bool = False  # Apply text overlays
    overlay_texts: list[str] = field(default_factory=list)  # "text@start-end" format
    fact_cards: list[str] = field(default_factory=list)  # "title|fact@start-end" format
    burn_subtitles: bool = False  # Burn subtitles into video

    # Phase 2: Background music
    background_music: bool = False  # Mix background music
    music_path: str = ""  # Path to music file
    music_volume: float = 0.15  # Music volume (0.0-1.0)
    music_duck: bool = True  # Duck music during speech

    # Phase 2: Social media upload
    upload_tiktok: bool = False  # Upload to TikTok
    tiktok_cookies: str = ""  # Path to browser cookies
    tiktok_description: str = ""
    tiktok_tags: list[str] = field(default_factory=list)
    upload_instagram: bool = False  # Upload to Instagram
    ig_access_token: str = ""  # Instagram access token
    ig_business_id: str = ""  # Instagram business account ID
    ig_caption: str = ""
    ig_tags: list[str] = field(default_factory=list)

    @property
    def work_dir(self) -> Path:
        return self.output_path.parent / f".{self.output_path.stem}_dub_work" if self.output_path else Path("/tmp/dub_work")

    def ensure_dirs(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "audio").mkdir(exist_ok=True)
        (self.work_dir / "segments").mkdir(exist_ok=True)
        (self.work_dir / "dubbed").mkdir(exist_ok=True)
        (self.work_dir / "subtitles").mkdir(exist_ok=True)
