"""CLI interface using Typer."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dub.config import DubConfig
from dub.models import ExitCode, MultiDubResult
from dub.pipeline import run_pipeline
from dub.utils import get_video_files, setup_logging

app = typer.Typer(
    name="dub",
    help="High-precision AI video dubbing platform",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# Language code → human name
LANG_NAMES = {
    "hi": "Hindi", "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ar": "Arabic", "pt": "Portuguese", "ru": "Russian", "it": "Italian",
    "bn": "Bengali", "gu": "Gujarati", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "pa": "Punjabi", "ur": "Urdu", "tr": "Turkish",
    "nl": "Dutch", "pl": "Polish", "th": "Thai", "vi": "Vietnamese",
    "id": "Indonesian",
}


def _parse_targets(target_str: str) -> list[str]:
    """Parse comma-separated language codes."""
    return [t.strip().lower() for t in target_str.split(",") if t.strip()]


def _compute_output_path(video_path: Path, lang: str, output: Path | None) -> Path:
    """Compute output path: <output_dir>/<lang>/<original_filename>"""
    if output:
        if output.suffix:
            return output
        return output / lang / video_path.name
    return video_path.parent / "dubbed" / lang / video_path.name


@app.command()
def video(
    input_path: Path = typer.Argument(..., help="Input video file or directory"),
    source: str = typer.Option("auto", "--source", "-s", help="Source language code (auto for detection)"),
    target: str = typer.Option("en", "--target", "-t", help="Target language(s), comma-separated (e.g. hi,es,fr)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory (or file for single video+lang)"),
    subtitles: bool = typer.Option(False, "--subtitles", help="Generate subtitle file"),
    subtitle_format: str = typer.Option("srt", "--subtitle-format", help="Subtitle format: srt, vtt, ass"),
    subtitle_mode: str = typer.Option("translated", "--subtitle-mode", help="original, translated, or bilingual"),
    voice_clone: bool = typer.Option(True, "--voice-clone/--no-voice-clone", help="Enable voice cloning"),
    lipsync: bool = typer.Option(False, "--lipsync", help="Enable lip-sync enhancement"),
    whisper_model: str = typer.Option("auto", "--whisper-model", help="Whisper model: auto, tiny, base, small, medium, large-v3"),
    tts_model: str = typer.Option("auto", "--tts-model", help="TTS model: auto, edge-tts"),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda"),
    max_diff: float = typer.Option(100.0, "--max-diff", help="Max allowed duration difference in ms"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process directories recursively"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    json_out: bool = typer.Option(False, "--json", help="Print machine-readable JSON result"),
    keep: bool = typer.Option(False, "--keep", help="Keep intermediate files"),
    # Phase 1: Content sourcing
    source_cc: bool = typer.Option(False, "--source-cc", help="Find and download CC-licensed videos from YouTube"),
    source_query: str = typer.Option("", "--source-query", help="Search query for CC videos"),
    source_api_key: str = typer.Option("", "--source-api-key", envvar="YOUTUBE_API_KEY", help="YouTube Data API key"),
    source_max_results: int = typer.Option(5, "--source-max-results", help="Max CC videos to download"),
    source_max_duration: int = typer.Option(600, "--source-max-duration", help="Max video duration in seconds"),
    # Phase 1: Branding
    branding: bool = typer.Option(False, "--branding", help="Apply intro/outro branding"),
    brand_channel: str = typer.Option("", "--brand-channel", help="Channel name for intro text"),
    brand_intro_text: str = typer.Option("", "--brand-intro", help="Custom intro text"),
    brand_outro_text: str = typer.Option("", "--brand-outro", help="Custom outro text"),
    brand_watermark: str = typer.Option("", "--brand-watermark", help="Watermark text overlay"),
    # Phase 1: Shorts
    shorts: bool = typer.Option(False, "--shorts", help="Generate vertical Shorts (9:16, ≤60s)"),
    shorts_count: int = typer.Option(1, "--shorts-count", help="Number of Shorts to generate"),
    shorts_max_duration: float = typer.Option(60.0, "--shorts-max-duration", help="Max Shorts duration in seconds"),
    # Phase 1: YouTube upload
    upload_youtube: bool = typer.Option(False, "--upload-youtube", help="Upload dubbed video to YouTube"),
    upload_title: str = typer.Option("", "--upload-title", help="YouTube video title"),
    upload_description: str = typer.Option("", "--upload-desc", help="YouTube video description"),
    upload_tags: str = typer.Option("", "--upload-tags", help="YouTube tags, comma-separated"),
    upload_privacy: str = typer.Option("private", "--upload-privacy", help="YouTube privacy: private, unlisted, public"),
    upload_client_secret: str = typer.Option("client_secret.json", "--upload-client-secret", help="Path to OAuth client_secret.json"),
    # Phase 2: Overlays
    overlays: bool = typer.Option(False, "--overlays", help="Apply text overlays to video"),
    overlay_texts: str = typer.Option("", "--overlay-text", help="Text overlay: 'text@start-end' (repeatable)"),
    fact_cards: str = typer.Option("", "--fact-card", help="Fact card: 'title|fact@start-end' (repeatable)"),
    burn_subs: bool = typer.Option(False, "--burn-subs", help="Burn subtitles into video"),
    # Phase 2: Background music
    background_music: bool = typer.Option(False, "--music", help="Mix background music"),
    music_path: str = typer.Option("", "--music-path", help="Path to music file"),
    music_volume: float = typer.Option(0.15, "--music-volume", help="Music volume (0.0-1.0)"),
    music_duck: bool = typer.Option(True, "--music-duck/--no-music-duck", help="Duck music during speech"),
    # Phase 2: Social media
    upload_tiktok: bool = typer.Option(False, "--upload-tiktok", help="Upload to TikTok"),
    tiktok_cookies: str = typer.Option("", "--tiktok-cookies", help="Path to TikTok browser cookies file"),
    tiktok_desc: str = typer.Option("", "--tiktok-desc", help="TikTok video description"),
    tiktok_tags: str = typer.Option("", "--tiktok-tags", help="TikTok tags, comma-separated"),
    upload_instagram: bool = typer.Option(False, "--upload-instagram", help="Upload to Instagram Reels"),
    ig_token: str = typer.Option("", "--ig-access-token", envvar="IG_ACCESS_TOKEN", help="Instagram access token"),
    ig_business: str = typer.Option("", "--ig-business-id", help="Instagram business account ID"),
    ig_caption: str = typer.Option("", "--ig-caption", help="Instagram caption"),
    ig_tags: str = typer.Option("", "--ig-tags", help="Instagram tags, comma-separated"),
) -> None:
    """Dub video(s) into one or more target languages.

    Examples:

        dub video.mp4 --target hi

        dub video.mp4 --target hi,es,fr

        dub ./videos/ --target hi,es --recursive

        dub video.mp4 --target hi,es --output ./dubbed/ --json

        dub video.mp4 --source-cc --source-query "science documentary" --target hi

        dub video.mp4 --target hi --branding --brand-channel "My Channel" --shorts

        dub video.mp4 --target hi --upload-youtube --upload-privacy public
    """
    setup_logging(verbose)

    # Handle source-cc mode: find and download CC videos, then process them
    if source_cc:
        if not source_query:
            console.print("[red]Error: --source-query is required when using --source-cc[/]")
            raise typer.Exit(code=ExitCode.VALIDATION)
        if not source_api_key:
            console.print("[red]Error: --source-api-key or YOUTUBE_API_KEY env var required[/]")
            raise typer.Exit(code=ExitCode.VALIDATION)

        from dub.source import search_cc_videos, download_batch

        console.print(f"[bold]Searching YouTube for CC videos: {source_query}[/]")
        cc_videos = search_cc_videos(
            source_query, source_api_key,
            max_results=source_max_results,
            max_duration=source_max_duration,
        )
        if not cc_videos:
            console.print("[red]No CC videos found[/]")
            raise typer.Exit(code=ExitCode.VALIDATION)

        console.print(f"[green]Found {len(cc_videos)} CC videos:[/]")
        for v in cc_videos:
            console.print(f"  {v.title} ({v.duration}s) — {v.url}")

        download_dir = Path(source_download_dir or "./cc_videos")
        console.print(f"[bold]Downloading to {download_dir}...[/]")
        downloaded = download_batch(cc_videos, download_dir)
        console.print(f"[green]Downloaded {len(downloaded)} videos[/]")

        # Set input_path to downloaded videos directory
        input_path = download_dir
        source_cc = False  # Don't re-trigger in the loop

    input_path = Path(input_path)
    if not input_path.exists():
        console.print(f"[red]Error: {input_path} does not exist[/]")
        raise typer.Exit(code=ExitCode.VALIDATION)

    videos = get_video_files(input_path, recursive)
    if not videos:
        console.print(f"[red]Error: No video files found in {input_path}[/]")
        raise typer.Exit(code=ExitCode.VALIDATION)

    languages = _parse_targets(target)
    lang_display = ", ".join(f"{l} ({LANG_NAMES.get(l, l)})" for l in languages)
    console.print(f"[bold]Processing {len(videos)} video(s) × {len(languages)} language(s): {lang_display}[/]")

    multi_results: list[MultiDubResult] = []

    for vid in videos:
        multi = MultiDubResult(
            input_video=str(vid),
            source_language=source,
            target_languages=languages,
        )

        for lang in languages:
            out_path = _compute_output_path(vid, lang, output)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            console.print(f"\n[bold cyan]{vid.name}[/] → [bold]{lang} ({LANG_NAMES.get(lang, lang)})[/] → {out_path}")

            config = DubConfig(
                input_path=vid,
                output_path=out_path,
                source_lang=source,
                target_lang=lang,
                subtitles=subtitles,
                subtitle_format=subtitle_format,
                subtitle_mode=subtitle_mode,
                voice_clone=voice_clone,
                lipsync=lipsync,
                whisper_model=whisper_model,
                tts_model=tts_model,
                device=device,
                max_duration_diff_ms=max_diff,
                recursive=recursive,
                verbose=verbose,
                json_output=json_out,
                keep_intermediate=keep,
                # Phase 1: Branding
                branding=branding,
                brand_channel=brand_channel,
                brand_intro_text=brand_intro_text,
                brand_outro_text=brand_outro_text,
                brand_watermark=brand_watermark,
                brand_watermark_pos="bottom-right",
                # Phase 1: Shorts
                shorts=shorts,
                shorts_count=shorts_count,
                shorts_max_duration=shorts_max_duration,
                # Phase 1: Upload
                upload_youtube=upload_youtube,
                upload_title=upload_title or vid.stem,
                upload_description=upload_description,
                upload_tags=[t.strip() for t in upload_tags.split(",") if t.strip()],
                upload_privacy=upload_privacy,
                upload_category="22",
                upload_client_secret=upload_client_secret,
                # Phase 2: Overlays
                overlays=overlays,
                overlay_texts=[t.strip() for t in overlay_texts.split("|") if t.strip()],
                fact_cards=[c.strip() for c in fact_cards.split("|") if c.strip()],
                burn_subtitles=burn_subs,
                # Phase 2: Music
                background_music=background_music,
                music_path=music_path,
                music_volume=music_volume,
                music_duck=music_duck,
                # Phase 2: Social
                upload_tiktok=upload_tiktok,
                tiktok_cookies=tiktok_cookies,
                tiktok_description=tiktok_desc,
                tiktok_tags=[t.strip() for t in tiktok_tags.split(",") if t.strip()],
                upload_instagram=upload_instagram,
                ig_access_token=ig_token,
                ig_business_id=ig_business,
                ig_caption=ig_caption,
                ig_tags=[t.strip() for t in ig_tags.split(",") if t.strip()],
            )

            result = run_pipeline(config)
            multi.results.append(result)

            if result.exit_code != ExitCode.SUCCESS:
                console.print(f"  [red]Failed: {result.error}[/]")

        multi_results.append(multi)

    console.print()
    _print_summary(multi_results)

    if json_out:
        all_results = [m.to_dict() for m in multi_results]
        output_json = {
            "command": "dub",
            "totalInputVideos": len(videos),
            "targetLanguages": languages,
            "totalOutputFiles": sum(m.successful for m in multi_results),
            "videos": all_results,
        }
        console.print("\n[bold]JSON Result:[/]")
        print(json.dumps(output_json, indent=2, default=str))


@app.command()
def agent_info() -> None:
    """Output machine-readable tool documentation for LLM agents.

    Prints JSON with all commands, options, examples, and capabilities
    so an LLM can understand and invoke the tool correctly.
    """
    info = {
        "tool": "dub",
        "version": "0.1.0",
        "description": "AI video dubbing platform. Translates speech in videos to other languages and generates dubbed output matching original timing.",
        "requires": ["ffmpeg", "python3.12+"],
        "install": "pip install -e '.[cpu]'",
        "commands": {
            "video": {
                "description": "Dub one or more video files into target language(s)",
                "usage": "dub video <input_path> [OPTIONS]",
                "arguments": {
                    "input_path": {
                        "type": "path",
                        "required": True,
                        "description": "Video file or directory containing videos"
                    }
                },
                "options": {
                    "--target": {"short": "-t", "type": "str", "default": "en", "description": "Target language(s), comma-separated. E.g. hi,es,fr"},
                    "--source": {"short": "-s", "type": "str", "default": "auto", "description": "Source language code. 'auto' for auto-detection"},
                    "--output": {"short": "-o", "type": "path", "default": None, "description": "Output directory. Default: ./dubbed/<lang>/filename.mp4"},
                    "--subtitles": {"type": "bool", "default": False, "description": "Generate subtitle file alongside video"},
                    "--subtitle-format": {"type": "str", "default": "srt", "choices": ["srt", "vtt", "ass"]},
                    "--subtitle-mode": {"type": "str", "default": "translated", "choices": ["original", "translated", "bilingual"]},
                    "--voice-clone": {"type": "bool", "default": True, "description": "Use voice cloning to match original speaker"},
                    "--lipsync": {"type": "bool", "default": False, "description": "Apply lip-sync enhancement (requires GPU)"},
                    "--whisper-model": {"type": "str", "default": "auto", "choices": ["auto", "tiny", "base", "small", "medium", "large-v3"]},
                    "--tts-model": {"type": "str", "default": "auto", "choices": ["auto", "edge-tts"]},
                    "--device": {"type": "str", "default": "auto", "choices": ["auto", "cpu", "cuda"]},
                    "--max-diff": {"type": "float", "default": 100.0, "description": "Max allowed duration difference in ms"},
                    "--recursive": {"short": "-r", "type": "bool", "default": False, "description": "Process directories recursively"},
                    "--verbose": {"short": "-v", "type": "bool", "default": False},
                    "--json": {"type": "bool", "default": False, "description": "Output machine-readable JSON"},
                    "--keep": {"type": "bool", "default": False, "description": "Keep intermediate work files"},
                },
                "examples": [
                    "dub video.mp4 --target hi",
                    "dub video.mp4 --target hi,es,fr",
                    "dub video.mp4 --target es --subtitles --output ./output/",
                    "dub ./videos/ --target hi --recursive --json",
                    "dub video.mp4 --target hi --verbose --keep",
                ],
            },
            "detect": {
                "description": "Detect hardware and show optimal model selections",
                "usage": "dub detect",
                "options": {},
                "examples": ["dub detect"],
            },
            "batch": {
                "description": "Batch dub all videos in a directory",
                "usage": "dub batch <input_dir> [OPTIONS]",
                "arguments": {
                    "input_dir": {"type": "path", "required": True, "description": "Directory containing video files"}
                },
                "options": {
                    "--target": {"short": "-t", "type": "str", "default": "en"},
                    "--source": {"short": "-s", "type": "str", "default": "auto"},
                    "--output-dir": {"short": "-o", "type": "path", "description": "Output root directory"},
                    "--recursive": {"short": "-r", "type": "bool", "default": False},
                    "--verbose": {"short": "-v", "type": "bool", "default": False},
                },
                "examples": [
                    "dub batch ./videos/ --target hi,es",
                    "dub batch ./videos/ --target hi --output-dir ./dubbed/",
                ],
            },
            "agent-info": {
                "description": "Output this machine-readable documentation",
                "usage": "dub agent-info",
                "options": {},
                "examples": ["dub agent-info"],
            },
        },
        "supportedLanguages": {
            code: name for code, name in LANG_NAMES.items()
        },
        "outputStructure": {
            "single_video": "dubbed/<lang>/<original_filename>.mp4",
            "multiple_videos": "dubbed/<lang>/<video1>.mp4, dubbed/<lang>/<video2>.mp4",
            "multiple_languages": "dubbed/hi/video.mp4, dubbed/es/video.mp4, dubbed/fr/video.mp4",
        },
        "jsonOutputFormat": {
            "command": "dub",
            "totalInputVideos": "int",
            "targetLanguages": ["lang_code"],
            "totalOutputFiles": "int",
            "videos": [
                {
                    "inputVideo": "absolute_path",
                    "sourceLanguage": "en",
                    "targetLanguages": ["hi", "es"],
                    "totalFiles": 2,
                    "successful": 2,
                    "failed": 0,
                    "outputFiles": ["dubbed/hi/video.mp4"],
                    "results": [
                        {
                            "status": "completed|failed|no_speech",
                            "inputVideo": "path",
                            "outputVideo": "path",
                            "sourceLanguage": "en",
                            "targetLanguage": "hi",
                            "durationDifferenceMs": 22.2,
                            "speakers": 1,
                            "segmentsProcessed": 7,
                            "subtitlesGenerated": False,
                            "lipsyncApplied": False,
                            "error": "",
                        }
                    ],
                }
            ],
        },
        "exitCodes": {
            "0": "success",
            "1": "validation failure (bad input, file not found)",
            "2": "transcription failure",
            "3": "translation failure",
            "4": "TTS failure",
            "5": "rendering failure",
        },
        "notes": [
            "Video stream is copied without re-encoding (no quality loss)",
            "Audio is replaced with dubbed speech",
            "Duration matching keeps output within 100ms of original",
            "Hardware is auto-detected and models are selected accordingly",
            "Diarization assigns speaker labels for multi-speaker videos",
            "Edge TTS (free, neural) is used for speech synthesis",
            "Translation uses Google Translate (free, no API key)",
            "Intermediate files are cleaned up unless --keep is used",
        ],
    }

    print(json.dumps(info, indent=2))


@app.command()
def detect() -> None:
    """Detect hardware and show optimal model selections."""
    from dub.hardware import detect_hardware, select_models

    hw = detect_hardware()
    hw = select_models(hw)

    console.print("\n[bold]Hardware Detection Results[/]")
    console.print(f"  CPU cores:    {hw.cpu_cores}")
    console.print(f"  RAM:          {hw.ram_gb:.1f} GB")
    console.print(f"  GPU:          {hw.gpu_name or 'None'}")
    console.print(f"  GPU Memory:   {hw.gpu_memory_gb:.1f} GB")
    console.print(f"  Disk Free:    {hw.disk_free_gb:.1f} GB")
    console.print(f"\n[bold]Selected Models[/]")
    console.print(f"  Whisper:      {hw.whisper_model} ({hw.whisper_compute})")
    console.print(f"  Device:       {hw.device}")
    console.print(f"  TTS:          {hw.tts_model}")
    console.print(f"  Voice Clone:  {'Yes' if hw.can_voice_clone else 'No'}")
    console.print(f"  Lip-Sync:     {'Yes' if hw.can_lipsync else 'No'}")
    console.print()


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Directory containing videos"),
    target: str = typer.Option("en", "--target", "-t", help="Target language(s), comma-separated"),
    source: str = typer.Option("auto", "--source", "-s", help="Source language"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="Output root directory"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process recursively"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Batch dub all videos in a directory into multiple languages.

    Output structure:

        <output_dir>/hi/video1.mp4

        <output_dir>/es/video1.mp4
    """
    setup_logging(verbose)

    if not input_dir.is_dir():
        console.print(f"[red]Error: {input_dir} is not a directory[/]")
        raise typer.Exit(code=ExitCode.VALIDATION)

    videos = get_video_files(input_dir, recursive)
    if not videos:
        console.print(f"[red]No video files found in {input_dir}[/]")
        raise typer.Exit(code=ExitCode.VALIDATION)

    languages = _parse_targets(target)
    out_root = output_dir or input_dir / "dubbed"
    out_root.mkdir(parents=True, exist_ok=True)

    lang_display = ", ".join(f"{l} ({LANG_NAMES.get(l, l)})" for l in languages)
    console.print(f"[bold]Batch dubbing {len(videos)} video(s) × {len(languages)} language(s): {lang_display}[/]")
    console.print(f"[bold]Output: {out_root}[/]")

    multi_results: list[MultiDubResult] = []

    for vid in videos:
        multi = MultiDubResult(
            input_video=str(vid),
            source_language=source,
            target_languages=languages,
        )

        for lang in languages:
            out_path = out_root / lang / vid.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            console.print(f"\n[bold cyan]{vid.name}[/] → [bold]{lang}[/]")

            config = DubConfig(
                input_path=vid,
                output_path=out_path,
                source_lang=source,
                target_lang=lang,
                verbose=verbose,
            )
            result = run_pipeline(config)
            multi.results.append(result)

        multi_results.append(multi)

    console.print()
    _print_summary(multi_results)


def _print_summary(multi_results: list[MultiDubResult]) -> None:
    """Print a rich summary table of all dubbing results."""
    table = Table(title="Dubbing Results", show_lines=True)
    table.add_column("Input Video", style="cyan", max_width=40)
    table.add_column("Language", style="bold")
    table.add_column("Status")
    table.add_column("Duration Diff", justify="right")
    table.add_column("Output Path", max_width=60)

    total_ok = 0
    total_fail = 0

    for multi in multi_results:
        for r in multi.results:
            lang_name = LANG_NAMES.get(r.target_language, r.target_language)
            if r.exit_code == ExitCode.SUCCESS:
                table.add_row(
                    Path(r.input_video).name,
                    f"{r.target_language} ({lang_name})",
                    "[green]OK[/]",
                    f"{r.duration_difference_ms:.1f}ms",
                    r.output_video,
                )
                total_ok += 1
            else:
                table.add_row(
                    Path(r.input_video).name,
                    f"{r.target_language} ({lang_name})",
                    "[red]FAIL[/]",
                    "—",
                    r.error[:50] if r.error else "unknown",
                )
                total_fail += 1

    console.print(table)
    console.print(f"\n[bold]Total: {total_ok + total_fail} files | [green]{total_ok} succeeded[/] | [red]{total_fail} failed[/][/]")


if __name__ == "__main__":
    app()
