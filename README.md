# Video Dubber

High-precision AI video dubbing platform. Translates spoken content from one language to another and generates a dubbed video matching the original's timing, pacing, and structure.

## Quick Start

```bash
# Install
pip install -e ".[cpu]"

# Single video
dub video.mp4 --target hi --output dubbed.mp4

# With subtitles and voice cloning
dub video.mp4 --target es --subtitles --voice-clone --output dubbed.mp4

# Batch
dub ./videos/ --target en --recursive

# Hardware detection
dub detect
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Interface                            │
│                    dub/video, dub/batch, dub/detect             │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    Pipeline Orchestrator                         │
│  (dub/pipeline.py) — sequences all stages, handles errors       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                                                                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐     │
│  │  Video   │  │  Hardware │  │Subtitle  │  │  LipSync  │     │
│  │ Processor│  │ Detector  │  │Generator │  │ (Optional)│     │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘     │
│                                                                 │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐     │
│  │Transcribe│  │  Diarize  │  │Translate │  │    TTS    │     │
│  │Whisper   │  │ Pyannote  │  │NLLB/GTrans│ │  XTTS     │     │
│  └──────────┘  └───────────┘  └──────────┘  └───────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Duration Matching & Alignment Module              │  │
│  │    (atempo chains, semantic compression, padding)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Input Video (.mp4/.mkv/.avi)
    │
    ▼
[1] Probe Video ──→ VideoInfo (resolution, fps, duration, codec)
    │
    ▼
[2] Extract Audio ──→ WAV (24kHz mono)
    │
    ▼
[3] Transcribe ──→ Segments[{id, start, end, text}]
    │
    ▼
[4] Diarize ──→ Segments[{..., speaker_id}]
    │
    ▼
[5] Translate ──→ Segments[{..., translated_text}]
    │
    ▼
[6] TTS ──→ Segments[{..., dubbed_audio_path}]
    │
    ▼
[7] Duration Match ──→ Speed-adjusted audio segments
    │
    ▼
[8] Merge ──→ Video + Dubbed Audio (copy video, replace audio)
    │
    ▼
[9] (Optional) Subtitles, Lip-Sync
    │
    ▼
Output Video (.mp4) + result.json
```

## Tech Stack

| Component | Technology | When |
|-----------|-----------|------|
| Video processing | FFmpeg | Always |
| Transcription | faster-whisper | Always |
| Speaker diarization | pyannote.audio 3.1 | Always |
| Translation | deep-translator (Google) | Default |
| Translation | Meta NLLB-200 | High-end GPU |
| TTS / Voice cloning | Coqui XTTS v2 | GPU ≥6GB |
| TTS (fallback) | Tacotron2-DDC | CPU / low-resource |
| Lip-sync | Wav2Lip | GPU ≥12GB |
| CLI | Typer + Rich | Always |

## Hardware-Aware Model Selection

| Hardware | Whisper | TTS | Voice Clone | Lip-Sync |
|----------|---------|-----|-------------|----------|
| RTX 4090 (24GB) | large-v3 (fp16) | XTTS v2 | Yes | Yes |
| RTX 3060 (12GB) | medium (fp16) | XTTS v2 | Yes | No |
| CPU (8GB+) | small (int8) | Tacotron2 | No | No |
| CPU (4GB+) | tiny (int8) | Tacotron2 | No | No |

## CLI Reference

```
dub video INPUT [OPTIONS]

Options:
  --source, -s TEXT          Source language (auto)        [default: auto]
  --target, -t TEXT          Target language               [default: en]
  --output, -o PATH          Output video path
  --subtitles                Generate subtitles           [default: false]
  --subtitle-format TEXT     srt, vtt, or ass             [default: srt]
  --subtitle-mode TEXT       original/translated/bilingual [default: translated]
  --voice-clone/--no-voice-clone                          [default: voice-clone]
  --lipsync                  Enable lip-sync              [default: false]
  --whisper-model TEXT       auto/tiny/base/small/medium/large-v3
  --tts-model TEXT           auto/xtts/cosyvoice
  --device TEXT              auto/cpu/cuda
  --max-diff FLOAT           Max duration diff in ms      [default: 100]
  --verbose, -v              Verbose logging
  --json                     Machine-readable JSON output
  --keep                     Keep intermediate files

dub detect                   Show hardware and model selections
dub batch DIR [OPTIONS]      Batch process all videos in directory
```

## Duration Matching Strategy

The pipeline uses a 4-stage approach:

1. **Translation shortening**: Remove filler words from translated text
2. **Neural TTS speed control**: Request specific speaking rate from TTS
3. **FFmpeg atempo chains**: Time-stretch audio (0.7x–1.4x range)
4. **Verification**: Confirm final duration within ≤100ms tolerance

Priority: semantic compression > speed adjustment > silence trimming.

## Error Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Validation failure (bad input) |
| 2 | Transcription failure |
| 3 | Translation failure |
| 4 | TTS failure |
| 5 | Rendering failure |

## Project Structure

```
sourceclip/
├── dub/
│   ├── __init__.py          # Package metadata
│   ├── __main__.py          # python -m dub
│   ├── cli.py               # Typer CLI commands
│   ├── pipeline.py          # Orchestrator
│   ├── config.py            # Configuration dataclass
│   ├── models.py            # Data models
│   ├── hardware.py          # Hardware detection
│   ├── video.py             # FFmpeg operations
│   ├── transcribe.py        # Whisper transcription
│   ├── diarize.py           # Speaker diarization
│   ├── translate.py         # Translation backends
│   ├── tts.py               # TTS / voice cloning
│   ├── align.py             # Duration matching
│   ├── subtitles.py         # SRT/VTT/ASS generation
│   ├── lipsync.py           # Lip-sync enhancement
│   └── utils.py             # Shared utilities
├── tests/
│   └── test_dub.py          # Unit tests
├── Dockerfile               # Multi-stage container
├── docker-compose.yml       # CPU + GPU services
├── pyproject.toml           # Package config
└── requirements.txt         # Flat dependency list
```

## Docker

```bash
# Build
docker build -t video-dubber .

# Run (CPU)
docker run --rm -v ./data:/data video-dubber video.mp4 --target hi

# Run (GPU)
docker compose --profile gpu run --rm dub-gpu video.mp4 --target es --lipsync
```

## License

MIT
# sourceclip
