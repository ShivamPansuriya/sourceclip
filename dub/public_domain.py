"""Source public domain content from Internet Archive and Project Gutenberg.

Fetches free, copyright-safe videos, audio, and text for content creation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dub.utils import logger


@dataclass
class PublicDomainItem:
    """A public domain item from Internet Archive or Gutenberg."""

    item_id: str
    title: str
    source: str  # internet_archive | gutenberg | librivox
    media_type: str  # video | audio | text
    url: str
    download_url: str = ""
    license: str = "public_domain"
    creator: str = ""
    year: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.item_id,
            "title": self.title,
            "source": self.source,
            "mediaType": self.media_type,
            "url": self.url,
            "license": self.license,
            "creator": self.creator,
            "year": self.year,
        }


def search_internet_archive(
    query: str,
    media_type: str = "movies",
    max_results: int = 10,
    license_filter: str = "publicdomain",
) -> list[PublicDomainItem]:
    """Search Internet Archive for public domain content.

    Uses the Archive.org Advanced Search API.

    Args:
        query: Search query.
        media_type: Media type (movies, audio, texts).
        max_results: Max results to return.
        license_filter: License filter (publicdomain, all).

    Returns:
        List of PublicDomainItem objects.
    """
    import urllib.request
    import urllib.parse

    # Archive.org search API
    search_query = f"mediatype:{media_type} AND ({query})"
    if license_filter == "publicdomain":
        search_query += " AND licenseurl:publicdomain"

    params = {
        "q": search_query,
        "fl[]": "identifier,title,creator,date,description,licenseurl",
        "sort[]": "downloads desc",
        "rows": str(max_results),
        "output": "json",
    }

    url = f"https://archive.org/advancedsearch.php?{'&'.join(f'{k}={urllib.parse.quote(str(v))}' for k, v in params.items())}"

    logger.info("Searching Internet Archive: %s", query)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VideoDubber/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.error("Archive.org search failed: %s", e)
        return []

    items = []
    for doc in data.get("response", {}).get("docs", []):
        identifier = doc.get("identifier", "")
        title = doc.get("title", "Untitled")
        creator = doc.get("creator", "")
        year = doc.get("date", "")[:4] if doc.get("date") else ""
        desc = doc.get("description", "")
        if isinstance(desc, list):
            desc = desc[0] if desc else ""

        # Get download URL
        download_url = f"https://archive.org/download/{identifier}"

        # Determine media type from identifier metadata
        item_media = "video" if media_type == "movies" else media_type.rstrip("s")

        items.append(PublicDomainItem(
            item_id=identifier,
            title=title,
            source="internet_archive",
            media_type=item_media,
            url=f"https://archive.org/details/{identifier}",
            download_url=download_url,
            license="public_domain",
            creator=creator if isinstance(creator, str) else ", ".join(creator) if creator else "",
            year=year,
            description=str(desc)[:500],
        ))

    logger.info("Found %d items on Internet Archive", len(items))
    return items


def search_gutenberg(
    query: str,
    max_results: int = 10,
) -> list[PublicDomainItem]:
    """Search Project Gutenberg for public domain texts.

    Useful for sourcing narration scripts from classic literature.

    Args:
        query: Search query.
        max_results: Max results.

    Returns:
        List of PublicDomainItem objects.
    """
    import urllib.request
    import urllib.parse

    # Gutenberg search API
    url = f"https://www.gutenberg.org/ebooks/search/?query={urllib.parse.quote(query)}&submit_search=Go%21"

    logger.info("Searching Project Gutenberg: %s", query)

    # Use the Gutenberg catalog API
    api_url = f"http://www.gutenberg.org/ebooks/search.json?query={urllib.parse.quote(query)}&limit={max_results}"

    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "VideoDubber/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception:
        # Fallback: parse the HTML search results
        return _search_gutenberg_html(query, max_results)

    items = []
    for work in data.get("works", [])[:max_results]:
        title = work.get("title", "Untitled")
        authors = work.get("authors", [])
        author = authors[0].get("name", "Unknown") if authors else "Unknown"
        book_id = work.get("id", "")

        items.append(PublicDomainItem(
            item_id=f"gutenberg-{book_id}",
            title=title,
            source="gutenberg",
            media_type="text",
            url=f"https://www.gutenberg.org/ebooks/{book_id}",
            download_url=f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
            license="public_domain",
            creator=author,
        ))

    logger.info("Found %d items on Gutenberg", len(items))
    return items


def search_librivox(
    query: str,
    max_results: int = 10,
) -> list[PublicDomainItem]:
    """Search LibriVox for public domain audiobooks.

    LibriVox audiobooks are free and can be used as narration reference.

    Args:
        query: Search query.
        max_results: Max results.

    Returns:
        List of PublicDomainItem objects.
    """
    import urllib.request
    import urllib.parse

    # LibriVox uses the Internet Archive API
    search_query = f"mediatype:audio AND collection:librivox AND ({query})"

    params = {
        "q": search_query,
        "fl[]": "identifier,title,creator,date",
        "sort[]": "downloads desc",
        "rows": str(max_results),
        "output": "json",
    }

    url = f"https://archive.org/advancedsearch.php?{'&'.join(f'{k}={urllib.parse.quote(str(v))}' for k, v in params.items())}"

    logger.info("Searching LibriVox: %s", query)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VideoDubber/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        logger.error("LibriVox search failed: %s", e)
        return []

    items = []
    for doc in data.get("response", {}).get("docs", []):
        identifier = doc.get("identifier", "")
        title = doc.get("title", "Untitled")
        creator = doc.get("creator", "")

        items.append(PublicDomainItem(
            item_id=identifier,
            title=title,
            source="librivox",
            media_type="audio",
            url=f"https://archive.org/details/{identifier}",
            download_url=f"https://archive.org/download/{identifier}",
            license="public_domain",
            creator=creator if isinstance(creator, str) else "",
        ))

    logger.info("Found %d LibriVox items", len(items))
    return items


def download_item(
    item: PublicDomainItem,
    output_dir: Path,
    filename: str | None = None,
) -> Path:
    """Download a public domain item.

    Args:
        item: PublicDomainItem to download.
        output_dir: Directory to save to.
        filename: Custom filename (auto-detected if None).

    Returns:
        Path to downloaded file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if item.source == "gutenberg":
        # Download text file directly
        url = item.download_url
        ext = ".txt"
    else:
        # Internet Archive: find best file
        url = _find_best_file(item)
        ext = Path(url).suffix if Path(url).suffix else ".mp4"

    if not filename:
        filename = f"{item.item_id}{ext}"

    output_path = output_dir / filename

    logger.info("Downloading: %s → %s", item.title, output_path)

    if item.source == "gutenberg" or ext == ".txt":
        # Simple HTTP download for text
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "VideoDubber/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(output_path, "wb") as f:
                f.write(resp.read())
    else:
        # Use yt-dlp for media files
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--format", "best[ext=mp4]/best",
            "--output", str(output_path),
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback to wget
            cmd = ["wget", "-q", "-O", str(output_path), url]
            subprocess.run(cmd, capture_output=True, text=True)

    if output_path.exists():
        logger.info("Downloaded: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
        return output_path

    raise FileNotFoundError(f"Failed to download: {item.title}")


def search_all_sources(
    query: str,
    sources: list[str] | None = None,
    max_results: int = 5,
) -> list[PublicDomainItem]:
    """Search all public domain sources.

    Args:
        query: Search query.
        sources: Sources to search (internet_archive, gutenberg, librivox).
        max_results: Max results per source.

    Returns:
        Combined list of items from all sources.
    """
    sources = sources or ["internet_archive", "gutenberg", "librivox"]
    all_items = []

    for source in sources:
        try:
            if source == "internet_archive":
                items = search_internet_archive(query, max_results=max_results)
            elif source == "gutenberg":
                items = search_gutenberg(query, max_results=max_results)
            elif source == "librivox":
                items = search_librivox(query, max_results=max_results)
            else:
                continue
            all_items.extend(items)
        except Exception as e:
            logger.error("Search failed for %s: %s", source, e)

    return all_items


def _find_best_file(item: PublicDomainItem) -> str:
    """Find the best downloadable file for an Archive.org item."""
    import urllib.request

    metadata_url = f"https://archive.org/metadata/{item.item_id}/files"
    try:
        req = urllib.request.Request(metadata_url, headers={"User-Agent": "VideoDubber/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        # Prefer MP4, then other video formats
        for f in data.get("result", []):
            name = f.get("name", "")
            fmt = f.get("format", "")
            if name.endswith(".mp4") and "Video" in fmt:
                return f"https://archive.org/download/{item.item_id}/{name}"

        # Fallback to any video file
        for f in data.get("result", []):
            name = f.get("name", "")
            if any(name.endswith(ext) for ext in [".mp4", ".ogv", ".webm"]):
                return f"https://archive.org/download/{item.item_id}/{name}"

    except Exception:
        pass

    return item.download_url


def _search_gutenberg_html(query: str, max_results: int) -> list[PublicDomainItem]:
    """Fallback HTML scraping for Gutenberg search."""
    import urllib.request
    import urllib.parse
    import re

    url = f"https://www.gutenberg.org/ebooks/search/?query={urllib.parse.quote(query)}&submit_search=Go%21"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VideoDubber/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        items = []
        # Parse book links
        for match in re.finditer(r'href="/ebooks/(\d+)"[^>]*>([^<]+)', html):
            book_id = match.group(1)
            title = match.group(2).strip()
            items.append(PublicDomainItem(
                item_id=f"gutenberg-{book_id}",
                title=title,
                source="gutenberg",
                media_type="text",
                url=f"https://www.gutenberg.org/ebooks/{book_id}",
                download_url=f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
                license="public_domain",
            ))
            if len(items) >= max_results:
                break

        return items
    except Exception:
        return []
