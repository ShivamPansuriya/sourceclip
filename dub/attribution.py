"""Auto-generate CC/public domain attribution text and credits.

Creates proper attribution for sourced content (Creative Commons, public domain).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dub.utils import logger


@dataclass
class AttributionEntry:
    """A single attribution entry for a sourced work."""

    title: str
    creator: str
    source: str  # URL or platform
    license: str  # CC-BY, CC-BY-SA, CC0, public_domain, etc.
    year: str = ""
    modifications: str = "Translated and dubbed"  # What we did
    url: str = ""

    def to_text(self) -> str:
        """Generate attribution text in CC-BY format."""
        parts = [f'"{self.title}"']

        if self.creator:
            parts.append(f"by {self.creator}")

        if self.year:
            parts.append(f"({self.year})")

        if self.url:
            parts.append(f"Source: {self.url}")
        elif self.source:
            parts.append(f"Source: {self.source}")

        parts.append(f"License: {self.license}")

        if self.modifications:
            parts.append(f"Modifications: {self.modifications}")

        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "creator": self.creator,
            "source": self.source,
            "license": self.license,
            "year": self.year,
            "modifications": self.modifications,
            "url": self.url,
        }


@dataclass
class AttributionDocument:
    """Complete attribution document for a video project."""

    project_title: str
    entries: list[AttributionEntry]
    generated_date: str = ""
    generator: str = "VideoDubber"

    def __post_init__(self):
        if not self.generated_date:
            self.generated_date = datetime.now().strftime("%Y-%m-%d")

    def to_text(self) -> str:
        """Generate human-readable attribution document."""
        lines = [
            f"ATTRIBUTION — {self.project_title}",
            f"Generated: {self.generated_date} by {self.generator}",
            "=" * 60,
            "",
        ]

        for i, entry in enumerate(self.entries, 1):
            lines.append(f"{i}. {entry.to_text()}")
            lines.append("")

        lines.extend([
            "=" * 60,
            "",
            "This content uses materials that are either:",
            "  • In the public domain",
            "  • Licensed under Creative Commons terms",
            "  • Used with permission from the copyright holder",
            "",
            "Modifications have been made as noted above.",
        ])

        return "\n".join(lines)

    def to_srt_footer(self) -> str:
        """Generate a short attribution for SRT subtitle footer."""
        parts = []
        for entry in self.entries:
            if entry.license.startswith("CC"):
                parts.append(f'"{entry.title}" by {entry.creator} ({entry.license})')
        return " | ".join(parts) if parts else ""

    def to_description_footer(self) -> str:
        """Generate attribution for video description."""
        lines = ["---", "Credits & Attribution:"]
        for entry in self.entries:
            lines.append(f'• "{entry.title}" by {entry.creator} — {entry.license}')
            if entry.url:
                lines.append(f"  {entry.url}")
        return "\n".join(lines)

    def save(self, output_path: Path) -> Path:
        """Save attribution document as text and JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save text version
        txt_path = output_path.with_suffix(".txt")
        txt_path.write_text(self.to_text(), encoding="utf-8")

        # Save JSON version
        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

        logger.info("Attribution saved: %s", txt_path)
        return txt_path

    def to_dict(self) -> dict:
        return {
            "projectTitle": self.project_title,
            "generatedDate": self.generated_date,
            "generator": self.generator,
            "entries": [e.to_dict() for e in self.entries],
        }


def create_attribution(
    project_title: str,
    video_sources: list[dict] | None = None,
    cc_sources: list[dict] | None = None,
    music_sources: list[dict] | None = None,
) -> AttributionDocument:
    """Create an attribution document from multiple source types.

    Args:
        project_title: Title of the video project.
        video_sources: List of video source dicts (from public_domain.py).
        cc_sources: List of CC video source dicts (from source.py).
        music_sources: List of music source dicts.

    Returns:
        AttributionDocument with all entries.
    """
    entries = []

    # Add video sources
    for src in (video_sources or []):
        entries.append(AttributionEntry(
            title=src.get("title", "Untitled"),
            creator=src.get("creator", "Unknown"),
            source=src.get("source", ""),
            license=src.get("license", "public_domain"),
            year=src.get("year", ""),
            url=src.get("url", ""),
            modifications="Translated, dubbed, and reformatted",
        ))

    # Add CC sources
    for src in (cc_sources or []):
        entries.append(AttributionEntry(
            title=src.get("title", "Untitled"),
            creator=src.get("channel", "Unknown"),
            source="YouTube",
            license="CC-BY",
            url=src.get("url", ""),
            modifications="Translated and dubbed",
        ))

    # Add music sources
    for src in (music_sources or []):
        entries.append(AttributionEntry(
            title=src.get("title", "Untitled"),
            creator=src.get("artist", "Unknown"),
            source=src.get("source", ""),
            license=src.get("license", "CC-BY"),
            url=src.get("url", ""),
            modifications="Mixed as background music",
        ))

    return AttributionDocument(
        project_title=project_title,
        entries=entries,
    )


def auto_attributed_description(
    base_description: str,
    attribution: AttributionDocument,
) -> str:
    """Append attribution to a video description.

    Args:
        base_description: Original description.
        attribution: Attribution document.

    Returns:
        Description with attribution footer.
    """
    footer = attribution.to_description_footer()
    return f"{base_description}\n\n{footer}" if base_description else footer
