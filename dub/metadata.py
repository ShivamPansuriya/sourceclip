"""Auto-generate SEO-optimized metadata for YouTube/TikTok/Instagram.

Creates titles, descriptions, tags, and hashtags optimized for each platform.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from dub.utils import logger


@dataclass
class VideoMetadata:
    """SEO-optimized metadata for a video."""

    title: str
    description: str
    tags: list[str]
    hashtags: list[str]
    category: str = "22"  # YouTube category ID
    language: str = "en"

    # Platform-specific
    youtube_title: str = ""
    youtube_description: str = ""
    youtube_tags: list[str] = field(default_factory=list)
    tiktok_caption: str = ""
    tiktok_tags: list[str] = field(default_factory=list)
    instagram_caption: str = ""
    instagram_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "tags": self.tags,
            "hashtags": self.hashtags,
            "category": self.category,
            "language": self.language,
            "youtube": {
                "title": self.youtube_title or self.title,
                "description": self.youtube_description or self.description,
                "tags": self.youtube_tags or self.tags,
            },
            "tiktok": {
                "caption": self.tiktok_caption or self.title,
                "tags": self.tiktok_tags or self.hashtags,
            },
            "instagram": {
                "caption": self.instagram_caption or self.description,
                "tags": self.instagram_tags or self.hashtags,
            },
        }


# Language-specific keyword mappings
LANGUAGE_KEYWORDS = {
    "hi": {
        "dubbed": ["hindi dubbed", "हिंदी में", "hindi version", "hindi dub"],
        "explained": ["समझाया", "hindi mein samjhao", "explained in hindi"],
    },
    "es": {
        "dubbed": ["doblaje español", "en español", "versión en español"],
        "explained": ["explicado", "explicado en español"],
    },
    "fr": {
        "dubbed": ["doublage français", "en français", "version française"],
        "explained": ["expliqué", "expliqué en français"],
    },
    "de": {
        "dubbed": ["deutscher sync", "auf Deutsch", "deutsche version"],
        "explained": ["erklärt", "auf Deutsch erklärt"],
    },
    "ja": {
        "dubbed": ["日本語吹き替え", "日本語版"],
        "explained": ["解説", "説明"],
    },
    "ko": {
        "dubbed": ["한국어 더빙", "한국어 버전"],
        "explained": ["설명", "해설"],
    },
    "pt": {
        "dubbed": ["dublado em português", "em português"],
        "explained": ["explicado", "explicado em português"],
    },
    "ar": {
        "dubbed": ["مدبلج عربي", "باللغة العربية"],
        "explained": ["مفسر", "شرح بالعربي"],
    },
}

# Trending topic keywords by category
TRENDING_TOPICS = {
    "science": ["science", "physics", "chemistry", "biology", "space", "nasa", "quantum"],
    "tech": ["technology", "ai", "artificial intelligence", "coding", "programming", "tech"],
    "history": ["history", "ancient", "civilization", "war", "empire", "historical"],
    "nature": ["nature", "animals", "wildlife", "ocean", "forest", "earth"],
    "space": ["space", "universe", "galaxy", "planet", "star", "cosmos"],
}


def generate_metadata(
    original_title: str,
    target_language: str,
    category: str = "general",
    description_addon: str = "",
    tags_addon: list[str] | None = None,
) -> VideoMetadata:
    """Generate SEO-optimized metadata for a dubbed video.

    Creates platform-specific titles, descriptions, and tags
    optimized for search and discovery in the target language.

    Args:
        original_title: Original video title.
        target_language: Target language code.
        category: Content category.
        description_addon: Additional description text.
        tags_addon: Additional tags.

    Returns:
        VideoMetadata with all optimized fields.
    """
    lang_name = _get_language_name(target_language)
    lang_keywords = LANGUAGE_KEYWORDS.get(target_language, {})

    # Generate base title
    title = _generate_title(original_title, lang_name, lang_keywords)

    # Generate description
    description = _generate_description(
        original_title, lang_name, description_addon, lang_keywords
    )

    # Generate tags
    tags = _generate_tags(original_title, target_language, category, tags_addon)

    # Generate hashtags
    hashtags = _generate_hashtags(original_title, target_language, category)

    # Platform-specific
    youtube_title = _optimize_youtube_title(title, target_language)
    youtube_description = _optimize_youtube_description(description, hashtags, target_language)
    youtube_tags = _optimize_youtube_tags(tags, target_language)

    tiktok_caption = _optimize_tiktok_caption(title, hashtags)
    tiktok_tags = _optimize_tiktok_tags(tags, target_language)

    instagram_caption = _optimize_instagram_caption(description, hashtags)
    instagram_tags = _optimize_instagram_tags(tags, target_language)

    return VideoMetadata(
        title=title,
        description=description,
        tags=tags,
        hashtags=hashtags,
        category=category,
        language=target_language,
        youtube_title=youtube_title,
        youtube_description=youtube_description,
        youtube_tags=youtube_tags,
        tiktok_caption=tiktok_caption,
        tiktok_tags=tiktok_tags,
        instagram_caption=instagram_caption,
        instagram_tags=instagram_tags,
    )


def save_metadata(metadata: VideoMetadata, output_path: Path) -> Path:
    """Save metadata to JSON file.

    Args:
        metadata: VideoMetadata to save.
        output_path: Output path (without extension).

    Returns:
        Path to saved JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")

    json_path.write_text(json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metadata saved: %s", json_path)
    return json_path


def _generate_title(
    original_title: str,
    lang_name: str,
    lang_keywords: dict,
) -> str:
    """Generate a compelling title in the target language."""
    # Clean original title
    clean_title = re.sub(r'\s*\([^)]*\)\s*', '', original_title)
    clean_title = re.sub(r'\s*\[[^\]]*\]\s*', '', clean_title).strip()

    # Add language indicator
    dubbed_keywords = lang_keywords.get("dubbed", [f"{lang_name.lower()} dubbed"])

    # Format: "Original Title (Hindi Dubbed)" or similar
    if len(clean_title) + len(dubbed_keywords[0]) + 5 < 100:
        return f"{clean_title} ({dubbed_keywords[0].title()})"
    else:
        return f"{clean_title[:80]}... ({lang_name})"


def _generate_description(
    original_title: str,
    lang_name: str,
    addon: str,
    lang_keywords: dict,
) -> str:
    """Generate SEO-optimized description."""
    explained = lang_keywords.get("explained", ["explained"])[0]

    lines = [
        f"Watch '{original_title}' {explained} in {lang_name}!",
        "",
        f"This video has been professionally dubbed into {lang_name} with accurate "
        f"translations and natural-sounding speech.",
        "",
    ]

    if addon:
        lines.append(addon)
        lines.append("")

    lines.extend([
        "LIKE & SUBSCRIBE for more content in your language!",
        "",
        "━━━━━━━━━━━━━━━━━━━━━",
        "Tags:",
    ])

    return "\n".join(lines)


def _generate_tags(
    original_title: str,
    target_language: str,
    category: str,
    addon_tags: list[str] | None,
) -> list[str]:
    """Generate SEO tags."""
    tags = []

    # Extract keywords from title
    words = re.findall(r'\w+', original_title.lower())
    tags.extend([w for w in words if len(w) > 3][:5])

    # Language tags
    lang_name = _get_language_name(target_language).lower()
    tags.extend([f"{lang_name} dubbed", f"{lang_name} version", lang_name])

    # Category tags
    if category in TRENDING_TOPICS:
        tags.extend(TRENDING_TOPICS[category][:3])

    # General tags
    tags.extend(["explained", "educational", "documentary"])

    # Add custom tags
    if addon_tags:
        tags.extend(addon_tags)

    # Deduplicate and limit
    seen = set()
    unique = []
    for t in tags:
        t_lower = t.lower().strip()
        if t_lower not in seen and len(t_lower) > 2:
            seen.add(t_lower)
            unique.append(t)

    return unique[:30]  # YouTube allows 30 tags max


def _generate_hashtags(
    original_title: str,
    target_language: str,
    category: str,
) -> list[str]:
    """Generate hashtags for social media."""
    hashtags = []

    # Language hashtag
    lang_name = _get_language_name(target_language)
    hashtags.append(f"{lang_name.lower()}dubbed")

    # Category hashtags
    if category in TRENDING_TOPICS:
        for topic in TRENDING_TOPICS[category][:2]:
            hashtags.append(topic.replace(" ", ""))

    # Title-derived hashtags
    words = re.findall(r'\w+', original_title.lower())
    for w in words:
        if len(w) > 4 and w not in {"the", "and", "for", "with", "this", "that"}:
            hashtags.append(w)
        if len(hashtags) >= 8:
            break

    return hashtags[:10]


def _optimize_youtube_title(title: str, lang: str) -> str:
    """Optimize title for YouTube (max 100 chars)."""
    if len(title) > 100:
        return title[:97] + "..."
    return title


def _optimize_youtube_description(desc: str, hashtags: list[str], lang: str) -> str:
    """Optimize description for YouTube (first 150 chars shown in search)."""
    tag_line = " ".join(f"#{h}" for h in hashtags[:5])
    return f"{desc}\n\n{tag_line}"


def _optimize_youtube_tags(tags: list[str], lang: str) -> list[str]:
    """Optimize tags for YouTube."""
    return tags[:30]


def _optimize_tiktok_caption(title: str, hashtags: list[str]) -> str:
    """Optimize caption for TikTok (max 300 chars, 3-5 hashtags ideal)."""
    tag_line = " ".join(f"#{h}" for h in hashtags[:5])
    caption = f"{title}\n\n{tag_line}"
    if len(caption) > 300:
        caption = caption[:297] + "..."
    return caption


def _optimize_tiktok_tags(tags: list[str], lang: str) -> list[str]:
    """Optimize tags for TikTok."""
    return tags[:10]


def _optimize_instagram_caption(desc: str, hashtags: list[str]) -> str:
    """Optimize caption for Instagram (max 2200 chars)."""
    tag_line = " ".join(f"#{h}" for h in hashtags[:10])
    caption = f"{desc}\n\n{tag_line}"
    if len(caption) > 2200:
        caption = caption[:2197] + "..."
    return caption


def _optimize_instagram_tags(tags: list[str], lang: str) -> list[str]:
    """Optimize tags for Instagram."""
    return tags[:30]


def _get_language_name(code: str) -> str:
    """Get human-readable language name from code."""
    names = {
        "hi": "Hindi", "en": "English", "es": "Spanish", "fr": "French",
        "de": "German", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
        "ar": "Arabic", "pt": "Portuguese", "ru": "Russian", "it": "Italian",
        "bn": "Bengali", "gu": "Gujarati", "ta": "Tamil", "te": "Telugu",
        "mr": "Marathi", "pa": "Punjabi", "ur": "Urdu", "tr": "Turkish",
        "nl": "Dutch", "pl": "Polish", "th": "Thai", "vi": "Vietnamese",
        "id": "Indonesian",
    }
    return names.get(code, code.upper())
