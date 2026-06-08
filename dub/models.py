"""Data models for the dubbing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PipelineStage(str, Enum):
    SOURCE = "source"
    EXTRACT_AUDIO = "extract_audio"
    TRANSCRIBE = "transcribe"
    DIARIZE = "diarize"
    TRANSLATE = "translate"
    TTS = "tts"
    ALIGN = "align"
    MERGE = "merge"
    SUBTITLE = "subtitle"
    LIPSYNC = "lipsync"
    BRANDING = "branding"
    SHORTS = "shorts"
    UPLOAD = "upload"


class ExitCode(int, Enum):
    SUCCESS = 0
    VALIDATION = 1
    TRANSCRIPTION = 2
    TRANSLATION = 3
    TTS = 4
    RENDERING = 5


@dataclass
class Segment:
    """A single speech segment with timing and speaker info."""

    id: int
    start: float  # seconds
    end: float  # seconds
    text: str
    translated_text: str = ""
    speaker: str = "SPEAKER_00"
    original_duration: float = 0.0
    dubbed_duration: float = 0.0
    dubbed_audio_path: Path | None = None
    duration_match_ok: bool = False

    def __post_init__(self):
        self.original_duration = self.end - self.start


@dataclass
class SpeakerProfile:
    """Voice profile for a detected speaker."""

    speaker_id: str
    segments: list[Segment] = field(default_factory=list)
    voice_embedding: list[float] = field(default_factory=list)
    gender: str = "unknown"
    reference_audio_path: Path | None = None


@dataclass
class VideoInfo:
    """Metadata extracted from input video."""

    path: Path
    duration: float  # seconds
    width: int
    height: int
    fps: float
    codec: str
    audio_codec: str
    audio_sample_rate: int
    audio_channels: int
    file_size: int  # bytes


@dataclass
class DubResult:
    """Final result of a single dubbing run (one video, one language)."""

    status: str
    input_video: str
    output_video: str
    source_language: str
    target_language: str
    duration_difference_ms: float
    speakers: int
    segments_processed: int
    subtitles_generated: bool
    lipsync_applied: bool
    exit_code: ExitCode = ExitCode.SUCCESS
    error: str = ""

    # Phase 1 fields
    source_video_id: str = ""
    source_video_title: str = ""
    branded: bool = False
    shorts_generated: list[str] = field(default_factory=list)
    youtube_upload_url: str = ""

    # Phase 2 fields
    overlays_applied: bool = False
    music_mixed: bool = False
    tiktok_url: str = ""
    instagram_url: str = ""

    # Phase 3 fields
    public_domain_sourced: bool = False
    attribution_file: str = ""
    metadata_file: str = ""

    def to_dict(self) -> dict:
        d = {
            "status": self.status,
            "inputVideo": self.input_video,
            "outputVideo": self.output_video,
            "sourceLanguage": self.source_language,
            "targetLanguage": self.target_language,
            "durationDifferenceMs": round(self.duration_difference_ms, 2),
            "speakers": self.speakers,
            "segmentsProcessed": self.segments_processed,
            "subtitlesGenerated": self.subtitles_generated,
            "lipsyncApplied": self.lipsync_applied,
            "error": self.error,
        }
        if self.source_video_id:
            d["sourceVideoId"] = self.source_video_id
            d["sourceVideoTitle"] = self.source_video_title
        if self.branded:
            d["branded"] = True
        if self.shorts_generated:
            d["shortsGenerated"] = self.shorts_generated
        if self.youtube_upload_url:
            d["youtubeUploadUrl"] = self.youtube_upload_url
        if self.overlays_applied:
            d["overlaysApplied"] = True
        if self.music_mixed:
            d["musicMixed"] = True
        if self.tiktok_url:
            d["tiktokUrl"] = self.tiktok_url
        if self.instagram_url:
            d["instagramUrl"] = self.instagram_url
        if self.public_domain_sourced:
            d["publicDomainSourced"] = True
        if self.attribution_file:
            d["attributionFile"] = self.attribution_file
        if self.metadata_file:
            d["metadataFile"] = self.metadata_file
        return d


@dataclass
class MultiDubResult:
    """Aggregated result across all videos and languages."""

    input_video: str
    source_language: str
    target_languages: list[str]
    results: list[DubResult] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.results)

    @property
    def successful(self) -> int:
        return sum(1 for r in self.results if r.exit_code == ExitCode.SUCCESS)

    @property
    def failed(self) -> int:
        return self.total_files - self.successful

    @property
    def output_files(self) -> list[str]:
        return [r.output_video for r in self.results if r.exit_code == ExitCode.SUCCESS]

    def to_dict(self) -> dict:
        return {
            "inputVideo": self.input_video,
            "sourceLanguage": self.source_language,
            "targetLanguages": self.target_languages,
            "totalFiles": self.total_files,
            "successful": self.successful,
            "failed": self.failed,
            "outputFiles": self.output_files,
            "results": [r.to_dict() for r in self.results],
        }
