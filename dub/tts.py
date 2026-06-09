"""Multi-speaker TTS using Microsoft Edge TTS with gender-matched voices.

Supports:
- Gender-matched voice selection (male→male voice, female→female voice)
- Distinct voices per speaker (different male voices for 2 male speakers)
- Pitch variation fallback when we run out of unique voices
- Per-language voice pools with male/female options
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import wave
from pathlib import Path

from dub.models import Segment, SpeakerProfile

logger = logging.getLogger("dub")

# Voice pools per language: male and female voices available
# Source: `python -m edge_tts --list-voices` (2026)
# Includes regional variants and multilingual voices for maximum coverage
VOICE_POOLS: dict[str, dict[str, list[str]]] = {
    "hi": {
        "male": [
            "hi-IN-MadhurNeural",
            "en-US-AndrewMultilingualNeural",  # Supports Hindi
            "en-US-BrianMultilingualNeural",   # Supports Hindi
        ],
        "female": [
            "hi-IN-SwaraNeural",
            "en-US-AvaMultilingualNeural",     # Supports Hindi
            "en-US-EmmaMultilingualNeural",    # Supports Hindi
        ],
    },
    "en": {
        "male": [
            "en-US-GuyNeural", "en-US-BrianNeural",
            "en-US-ChristopherNeural", "en-US-EricNeural",
            "en-US-RogerNeural", "en-US-SteffanNeural",
            "en-US-AndrewNeural",
        ],
        "female": [
            "en-US-AriaNeural", "en-US-JennyNeural",
            "en-US-MichelleNeural", "en-US-EmmaNeural",
            "en-US-AvaNeural",
        ],
    },
    "es": {
        "male": [
            "es-ES-AlvaroNeural", "es-US-AlonsoNeural",
            "es-MX-JorgeNeural",
        ],
        "female": [
            "es-ES-ElviraNeural", "es-ES-XimenaNeural",
            "es-US-PalomaNeural",
        ],
    },
    "fr": {
        "male": [
            "fr-FR-HenriNeural", "fr-FR-RemyMultilingualNeural",
            "fr-CA-ThierryNeural",
        ],
        "female": [
            "fr-FR-DeniseNeural", "fr-FR-EloiseNeural",
            "fr-FR-VivienneMultilingualNeural",
        ],
    },
    "de": {
        "male": [
            "de-DE-ConradNeural", "de-DE-KillianNeural",
            "de-DE-FlorianMultilingualNeural",
        ],
        "female": [
            "de-DE-KatjaNeural", "de-DE-AmalaNeural",
            "de-DE-SeraphinaMultilingualNeural",
        ],
    },
    "ja": {
        "male": [
            "ja-JP-KeitaNeural",
            "en-US-AndrewMultilingualNeural",  # Supports Japanese
            "en-US-BrianMultilingualNeural",   # Supports Japanese
        ],
        "female": [
            "ja-JP-NanamiNeural",
            "en-US-AvaMultilingualNeural",     # Supports Japanese
            "en-US-EmmaMultilingualNeural",    # Supports Japanese
        ],
    },
    "ko": {
        "male": [
            "ko-KR-InJoonNeural",
            "ko-KR-HyunsuMultilingualNeural",
            "en-US-AndrewMultilingualNeural",  # Supports Korean
        ],
        "female": [
            "ko-KR-SunHiNeural",
            "en-US-AvaMultilingualNeural",     # Supports Korean
            "en-US-EmmaMultilingualNeural",    # Supports Korean
        ],
    },
    "zh": {
        "male": [
            "zh-CN-YunxiNeural", "zh-CN-YunjianNeural",
            "zh-CN-YunyangNeural",
        ],
        "female": [
            "zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural",
            "zh-TW-HsiaoChenNeural",
        ],
    },
    "ar": {
        "male": [
            "ar-SA-HamedNeural", "ar-EG-ShakirNeural",
            "ar-AE-HamdanNeural",
        ],
        "female": [
            "ar-SA-ZariyahNeural", "ar-EG-SalmaNeural",
            "ar-AE-FatimaNeural",
        ],
    },
    "pt": {
        "male": [
            "pt-BR-AntonioNeural", "pt-PT-DuarteNeural",
            "en-US-AndrewMultilingualNeural",  # Supports Portuguese
        ],
        "female": [
            "pt-BR-FranciscaNeural", "pt-BR-ThalitaMultilingualNeural",
            "pt-PT-RaquelNeural",
        ],
    },
    "ru": {
        "male": [
            "ru-RU-DmitryNeural",
            "en-US-AndrewMultilingualNeural",  # Supports Russian
            "en-US-BrianMultilingualNeural",   # Supports Russian
        ],
        "female": [
            "ru-RU-SvetlanaNeural",
            "en-US-AvaMultilingualNeural",     # Supports Russian
            "en-US-EmmaMultilingualNeural",    # Supports Russian
        ],
    },
    "it": {
        "male": [
            "it-IT-DiegoNeural", "it-IT-GiuseppeMultilingualNeural",
            "en-US-AndrewMultilingualNeural",
        ],
        "female": [
            "it-IT-ElsaNeural", "it-IT-IsabellaNeural",
            "en-US-AvaMultilingualNeural",
        ],
    },
    "bn": {
        "male": [
            "bn-BD-PradeepNeural", "bn-IN-BashkarNeural",
            "en-US-AndrewMultilingualNeural",
        ],
        "female": [
            "bn-BD-NabanitaNeural", "bn-IN-TanishaaNeural",
            "en-US-AvaMultilingualNeural",
        ],
    },
    "gu": {
        "male": [
            "gu-IN-NiranjanNeural",
            "en-US-AndrewMultilingualNeural",
            "en-US-BrianMultilingualNeural",
        ],
        "female": [
            "gu-IN-DhwaniNeural",
            "en-US-AvaMultilingualNeural",
            "en-US-EmmaMultilingualNeural",
        ],
    },
    "ta": {
        "male": [
            "ta-IN-ValluvarNeural", "ta-MY-SuryaNeural",
            "ta-LK-KumarNeural",
        ],
        "female": [
            "ta-IN-PallaviNeural", "ta-MY-KaniNeural",
            "ta-LK-SaranyaNeural",
        ],
    },
    "te": {
        "male": [
            "te-IN-MohanNeural",
            "en-US-AndrewMultilingualNeural",
            "en-US-BrianMultilingualNeural",
        ],
        "female": [
            "te-IN-ShrutiNeural",
            "en-US-AvaMultilingualNeural",
            "en-US-EmmaMultilingualNeural",
        ],
    },
    "mr": {
        "male": [
            "mr-IN-ManoharNeural",
            "en-US-AndrewMultilingualNeural",
            "en-US-BrianMultilingualNeural",
        ],
        "female": [
            "mr-IN-AarohiNeural",
            "en-US-AvaMultilingualNeural",
            "en-US-EmmaMultilingualNeural",
        ],
    },
    "pa": {
        "male": [
            "en-US-AndrewMultilingualNeural",
            "en-US-BrianMultilingualNeural",
            "en-US-ChristopherNeural",
        ],
        "female": [
            "en-US-AvaMultilingualNeural",
            "en-US-EmmaMultilingualNeural",
            "en-US-JennyNeural",
        ],
    },
    "ur": {
        "male": [
            "ur-PK-AsadNeural", "ur-IN-SalmanNeural",
            "en-US-AndrewMultilingualNeural",
        ],
        "female": [
            "ur-PK-UzmaNeural", "ur-IN-GulNeural",
            "en-US-AvaMultilingualNeural",
        ],
    },
    "tr": {
        "male": [
            "tr-TR-AhmetNeural",
            "en-US-AndrewMultilingualNeural",
            "en-US-BrianMultilingualNeural",
        ],
        "female": [
            "tr-TR-EmelNeural",
            "en-US-AvaMultilingualNeural",
            "en-US-EmmaMultilingualNeural",
        ],
    },
    "nl": {
        "male": [
            "nl-NL-MaartenNeural", "nl-BE-ArnaudNeural",
            "en-US-AndrewMultilingualNeural",
        ],
        "female": [
            "nl-NL-ColetteNeural", "nl-NL-FennaNeural",
            "nl-BE-DenaNeural",
        ],
    },
    "pl": {
        "male": [
            "pl-PL-MarekNeural",
            "en-US-AndrewMultilingualNeural",
            "en-US-BrianMultilingualNeural",
        ],
        "female": [
            "pl-PL-ZofiaNeural",
            "en-US-AvaMultilingualNeural",
            "en-US-EmmaMultilingualNeural",
        ],
    },
    "th": {
        "male": [
            "th-TH-NiwatNeural",
            "en-US-AndrewMultilingualNeural",
            "en-US-BrianMultilingualNeural",
        ],
        "female": [
            "th-TH-PremwadeeNeural",
            "en-US-AvaMultilingualNeural",
            "en-US-EmmaMultilingualNeural",
        ],
    },
    "vi": {
        "male": [
            "vi-VN-NamMinhNeural",
            "en-US-AndrewMultilingualNeural",
            "en-US-BrianMultilingualNeural",
        ],
        "female": [
            "vi-VN-HoaiMyNeural",
            "en-US-AvaMultilingualNeural",
            "en-US-EmmaMultilingualNeural",
        ],
    },
    "id": {
        "male": [
            "id-ID-ArdiNeural",
            "en-US-AndrewMultilingualNeural",
            "en-US-BrianMultilingualNeural",
        ],
        "female": [
            "id-ID-GadisNeural",
            "en-US-AvaMultilingualNeural",
            "en-US-EmmaMultilingualNeural",
        ],
    },
}

# Fallback voice when language not in pool
FALLBACK_VOICE = "en-US-GuyNeural"


def assign_voices(
    speaker_profiles: dict[str, SpeakerProfile],
    target_lang: str,
) -> dict[str, SpeakerProfile]:
    """Assign distinct Edge TTS voices to speakers based on gender.

    Strategy:
    1. Detect gender from speaker profile (set during diarization)
    2. Assign unique voices from the gender-appropriate pool
    3. If more speakers than available voices, apply pitch shifts
    4. If gender unknown, try to infer from segment duration patterns

    Returns:
        Updated speaker_profiles with assigned_voice set.
    """
    lang_code = target_lang.split("-")[0]  # "hi-IN" → "hi"
    pool = VOICE_POOLS.get(lang_code, VOICE_POOLS.get("en", VOICE_POOLS["en"]))

    male_speakers = []
    female_speakers = []
    unknown_speakers = []

    for sp_id, profile in speaker_profiles.items():
        if profile.gender == "male":
            male_speakers.append(profile)
        elif profile.gender == "female":
            female_speakers.append(profile)
        else:
            unknown_speakers.append(profile)

    # Assign male voices
    male_voices = pool["male"]
    for i, sp in enumerate(male_speakers):
        if i < len(male_voices):
            sp.assigned_voice = male_voices[i]
            sp.voice_pitch_shift = "+0Hz"
        else:
            # More male speakers than voices — reuse with pitch shift
            sp.assigned_voice = male_voices[i % len(male_voices)]
            cycle = i // len(male_voices)
            shifts = ["+0Hz", "+15Hz", "-15Hz", "+30Hz", "-30Hz"]
            sp.voice_pitch_shift = shifts[cycle % len(shifts)]

    # Assign female voices
    female_voices = pool["female"]
    for i, sp in enumerate(female_speakers):
        if i < len(female_voices):
            sp.assigned_voice = female_voices[i]
            sp.voice_pitch_shift = "+0Hz"
        else:
            sp.assigned_voice = female_voices[i % len(female_voices)]
            cycle = i // len(female_voices)
            shifts = ["+0Hz", "+15Hz", "-15Hz", "+30Hz", "-30Hz"]
            sp.voice_pitch_shift = shifts[cycle % len(shifts)]

    # Handle unknown gender — assign from male pool as default, with pitch variation
    all_voices_used = set(sp.assigned_voice for sp in speaker_profiles.values() if sp.assigned_voice)
    for i, sp in enumerate(unknown_speakers):
        # Try to find an unused voice
        for voice in male_voices + female_voices:
            if voice not in all_voices_used:
                sp.assigned_voice = voice
                sp.voice_pitch_shift = "+0Hz"
                all_voices_used.add(voice)
                break
        else:
            # All voices used, reuse with pitch shift
            sp.assigned_voice = male_voices[0] if male_voices else FALLBACK_VOICE
            sp.voice_pitch_shift = f"+{(i + 1) * 15}Hz"

        logger.info("  %s: gender unknown, assigned %s", sp.speaker_id, sp.assigned_voice)

    # Log assignments
    for sp_id, profile in speaker_profiles.items():
        logger.info(
            "  %s → %s (gender=%s, pitch=%s)",
            sp_id, profile.assigned_voice, profile.gender, profile.voice_pitch_shift,
        )

    return speaker_profiles


def synthesize_segments(
    segments: list[Segment],
    speaker_profiles: dict[str, SpeakerProfile],
    tts_model: str,
    device: str,
    output_dir: Path,
    voice_clone: bool = True,
    target_lang: str = "en",
) -> list[Segment]:
    """Generate dubbed audio for each segment using multi-speaker Edge TTS.

    Each speaker gets their assigned voice with appropriate gender and pitch.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assign voices if not already done
    if not any(sp.assigned_voice for sp in speaker_profiles.values()):
        assign_voices(speaker_profiles, target_lang)

    # Synthesize all segments
    _synthesize_edge_tts_multi(segments, speaker_profiles, output_dir)

    # Update durations from generated WAV files
    for seg in segments:
        if seg.dubbed_audio_path and seg.dubbed_audio_path.exists():
            with wave.open(str(seg.dubbed_audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                seg.dubbed_duration = frames / rate

    logger.info("Synthesized %d segments with %d voices",
                len(segments), len(set(sp.assigned_voice for sp in speaker_profiles.values())))
    return segments


def _synthesize_edge_tts_multi(
    segments: list[Segment],
    speaker_profiles: dict[str, SpeakerProfile],
    output_dir: Path,
) -> None:
    """Synthesize using Edge TTS with per-speaker voice assignment."""
    import edge_tts

    # Build segment → voice mapping
    seg_voice_map: dict[int, tuple[str, str]] = {}  # seg_id → (voice, pitch)
    for seg in segments:
        sp = speaker_profiles.get(seg.speaker)
        if sp and sp.assigned_voice:
            seg_voice_map[seg.id] = (sp.assigned_voice, sp.voice_pitch_shift)
        else:
            seg_voice_map[seg.id] = (FALLBACK_VOICE, "+0Hz")

    async def _run():
        for seg in segments:
            if not seg.translated_text.strip():
                continue

            voice, pitch = seg_voice_map.get(seg.id, (FALLBACK_VOICE, "+0Hz"))

            out_path = output_dir / f"dubbed_{seg.id:04d}.mp3"
            wav_path = output_dir / f"dubbed_{seg.id:04d}.wav"

            try:
                # Use voice + pitch for differentiation
                communicate = edge_tts.Communicate(
                    seg.translated_text,
                    voice,
                    pitch=pitch,
                )
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
                logger.debug("Segment %d → %s (voice=%s, pitch=%s)",
                           seg.id, wav_path.name, voice, pitch)

            except Exception as e:
                logger.warning("TTS failed for segment %d: %s", seg.id, e)

    asyncio.run(_run())


def list_voices_for_language(lang_code: str) -> dict[str, list[str]]:
    """Get available voices for a language.

    Returns:
        {"male": [...], "female": [...]}
    """
    lang = lang_code.split("-")[0]
    return VOICE_POOLS.get(lang, {"male": [FALLBACK_VOICE], "female": []})
