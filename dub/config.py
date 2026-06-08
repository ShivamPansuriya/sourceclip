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

    @property
    def work_dir(self) -> Path:
        return self.output_path.parent / f".{self.output_path.stem}_dub_work" if self.output_path else Path("/tmp/dub_work")

    def ensure_dirs(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "audio").mkdir(exist_ok=True)
        (self.work_dir / "segments").mkdir(exist_ok=True)
        (self.work_dir / "dubbed").mkdir(exist_ok=True)
        (self.work_dir / "subtitles").mkdir(exist_ok=True)
