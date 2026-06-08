"""Utility helpers."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logger = logging.getLogger("dub")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def run_cmd(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess with logging."""
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def ffmpeg_available() -> bool:
    try:
        run_cmd(["ffmpeg", "-version"], capture=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def format_duration(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{int(m):02d}:{s:05.2f}"


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def print_result(result: dict) -> None:
    console.print_json(json.dumps(result, default=str))


def get_video_files(path: Path, recursive: bool = False) -> list[Path]:
    exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
    if path.is_file():
        return [path] if path.suffix.lower() in exts else []
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in path.glob(pattern) if p.suffix.lower() in exts)
