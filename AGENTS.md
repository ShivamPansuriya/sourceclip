# AGENTS.md — Video Dubber Tool Reference

This file documents the `dub` CLI tool for LLM agents.

## Quick Start

```bash
# Single video, single language
dub video input.mp4 --target hi

# Single video, multiple languages
dub video input.mp4 --target hi,es,fr

# Directory of videos, multiple languages
dub video ./videos/ --target hi,es --recursive

# Get machine-readable JSON output
dub video input.mp4 --target hi --json

# Detect hardware
dub detect

# Get full tool documentation as JSON
dub agent-info
```

## Commands

| Command | Description |
|---------|-------------|
| `dub video <path>` | Dub video file(s) into target language(s) |
| `dub detect` | Show hardware and optimal model selections |
| `dub batch <dir>` | Batch dub all videos in a directory |
| `dub agent-info` | Output full tool documentation as JSON |

## Options (video command)

| Flag | Default | Description |
|------|---------|-------------|
| `--target, -t` | `en` | Target language(s), comma-separated (e.g. `hi,es,fr`) |
| `--source, -s` | `auto` | Source language code (`auto` = detect automatically) |
| `--output, -o` | `./dubbed/<lang>/` | Output directory |
| `--subtitles` | `false` | Generate subtitle file |
| `--subtitle-format` | `srt` | `srt`, `vtt`, or `ass` |
| `--subtitle-mode` | `translated` | `original`, `translated`, or `bilingual` |
| `--voice-clone` | `true` | Clone original speaker's voice |
| `--lipsync` | `false` | Apply lip-sync (needs GPU) |
| `--whisper-model` | `auto` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--device` | `auto` | `cpu` or `cuda` |
| `--max-diff` | `100` | Max duration difference in ms |
| `--recursive, -r` | `false` | Process subdirectories |
| `--verbose, -v` | `false` | Verbose logging |
| `--json` | `false` | Machine-readable JSON output |
| `--keep` | `false` | Keep intermediate work files |

## Output Structure

```
dubbed/
├── hi/
│   └── video.mp4
├── es/
│   └── video.mp4
└── fr/
    └── video.mp4
```

## Supported Languages

hi (Hindi), en (English), es (Spanish), fr (French), de (German),
ja (Japanese), ko (Korean), zh (Chinese), ar (Arabic), pt (Portuguese),
ru (Russian), it (Italian), bn (Bengali), gu (Gujarati), ta (Tamil),
te (Telugu), mr (Marathi), pa (Punjabi), ur (Urdu), tr (Turkish),
nl (Dutch), pl (Polish), th (Thai), vi (Vietnamese), id (Indonesian)

## JSON Output Format

```json
{
  "command": "dub",
  "totalInputVideos": 1,
  "targetLanguages": ["hi", "es"],
  "totalOutputFiles": 2,
  "videos": [
    {
      "inputVideo": "/path/to/video.mp4",
      "sourceLanguage": "en",
      "targetLanguages": ["hi", "es"],
      "totalFiles": 2,
      "successful": 2,
      "failed": 0,
      "outputFiles": ["dubbed/hi/video.mp4", "dubbed/es/video.mp4"],
      "results": [
        {
          "status": "completed",
          "outputVideo": "dubbed/hi/video.mp4",
          "targetLanguage": "hi",
          "durationDifferenceMs": 22.2,
          "segmentsProcessed": 7,
          "error": ""
        }
      ]
    }
  ]
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Validation failure |
| 2 | Transcription failure |
| 3 | Translation failure |
| 4 | TTS failure |
| 5 | Rendering failure |

## How It Works

1. **Extract audio** from video (FFmpeg, no re-encode)
2. **Transcribe** speech (faster-whisper)
3. **Detect speakers** (pyannote.audio, falls back to single speaker)
4. **Translate** text (Google Translate via deep-translator)
5. **Generate speech** (Microsoft Edge TTS — free, neural quality)
6. **Match durations** (speed adjustment + truncate/pad to ≤100ms tolerance)
7. **Merge** dubbed audio with original video (FFmpeg, video stream copied)

## Requirements

- Python 3.12+
- ffmpeg (system package)
- ~2GB disk for model downloads (first run)
