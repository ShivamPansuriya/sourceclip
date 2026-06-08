"""TTS module using Microsoft Edge TTS (edge-tts).

edge-tts provides free, high-quality neural speech synthesis
supporting 300+ voices across 70+ languages including Hindi.
No GPU required, no API key needed.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import wave
from pathlib import Path

from dub.models import Segment, SpeakerProfile

logger = logging.getLogger("dub")

# Voice mapping: language code → preferred voice
VOICE_MAP: dict[str, str] = {
    "en": "en-US-GuyNeural",
    "hi": "hi-IN-MadhurNeural",
    "es": "es-ES-AlvaroNeural",
    "fr": "fr-FR-HenriNeural",
    "de": "de-DE-ConradNeural",
    "ja": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "zh": "zh-CN-YunxiNeural",
    "ar": "ar-SA-HamedNeural",
    "pt": "pt-BR-AntonioNeural",
    "ru": "ru-RU-DmitryNeural",
    "it": "it-IT-DiegoNeural",
    "bn": "bn-BD-BashirNeural",
    "gu": "gu-IN-MehulNeural",
    "ta": "ta-IN-ValluvarNeural",
    "te": "te-IN-ShravanNeural",
    "mr": "mr-IN-SuyogNeural",
    "pa": "pa-IN-GurpreetNeural",
    "ur": "ur-PK-AsadNeural",
    "tr": "tr-TR-AhmetNeural",
    "nl": "nl-NL-MaartenNeural",
    "pl": "pl-PL-MarekNeural",
    "th": "th-TH-NiwatNeural",
    "vi": "vi-VN-NamMinhNeural",
    "id": "id-ID-AryaNeural",
}


def synthesize_segments(
    segments: list[Segment],
    speaker_profiles: dict[str, SpeakerProfile],
    tts_model: str,
    device: str,
    output_dir: Path,
    voice_clone: bool = True,
    target_lang: str = "en",
) -> list[Segment]:
    """Generate dubbed audio for each segment using Edge TTS.

    Args:
        segments: Segments with translated_text to synthesize.
        speaker_profiles: Speaker profiles (unused for edge-tts).
        tts_model: TTS model name (ignored — always uses edge-tts).
        device: Device (ignored — edge-tts is cloud-based).
        output_dir: Directory for output WAV files.
        voice_clone: Ignored (edge-tts doesn't clone voices).
        target_lang: Target language code for voice selection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    voice = VOICE_MAP.get(target_lang, VOICE_MAP.get("en"))
    logger.info("Using Edge TTS voice: %s", voice)

    # Synthesize all segments
    _synthesize_edge_tts(segments, voice, output_dir)

    # Update durations from generated WAV files
    for seg in segments:
        if seg.dubbed_audio_path and seg.dubbed_audio_path.exists():
            with wave.open(str(seg.dubbed_audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                seg.dubbed_duration = frames / rate

    logger.info("Synthesized %d segments", len(segments))
    return segments


def _synthesize_edge_tts(
    segments: list[Segment],
    voice: str,
    output_dir: Path,
) -> None:
    """Synthesize using Microsoft Edge TTS."""
    import edge_tts

    async def _run():
        for seg in segments:
            if not seg.translated_text.strip():
                continue

            out_path = output_dir / f"dubbed_{seg.id:04d}.mp3"
            wav_path = output_dir / f"dubbed_{seg.id:04d}.wav"

            try:
                communicate = edge_tts.Communicate(seg.translated_text, voice)
                await communicate.save(str(out_path))

                # Convert MP3 → WAV for consistent processing
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(out_path),
                        "-ar", "24000", "-ac", "1",
                        "-acodec", "pcm_s16le",
                        str(wav_path),
                    ],
                    check=True,
                    capture_output=True,
                )

                # Remove temp MP3
                out_path.unlink(missing_ok=True)

                seg.dubbed_audio_path = wav_path
                logger.debug("Segment %d → %s", seg.id, wav_path.name)

            except Exception as e:
                logger.warning("TTS failed for segment %d: %s", seg.id, e)

    asyncio.run(_run())
