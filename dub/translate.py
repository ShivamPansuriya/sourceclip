"""Translation module — multi-backend, context-aware translation."""

from __future__ import annotations

import logging
from dub.models import Segment

logger = logging.getLogger("dub")

# Mapping from common ISO codes to Google Translate codes
_LANG_MAP = {
    "hi": "hi", "en": "en", "es": "es", "fr": "fr", "de": "de",
    "ja": "ja", "ko": "ko", "zh": "zh-CN", "ar": "ar", "pt": "pt",
    "ru": "ru", "it": "it", "bn": "bn", "gu": "gu", "ta": "ta",
    "te": "te", "mr": "mr", "pa": "pa", "ur": "ur", "tr": "tr",
    "nl": "nl", "pl": "pl", "th": "th", "vi": "vi", "id": "id",
}


def translate_segments(
    segments: list[Segment],
    source_lang: str,
    target_lang: str,
    backend: str = "google",
) -> list[Segment]:
    """Translate all segment texts in-place.

    Args:
        segments: Transcription segments to translate.
        source_lang: Source language code (or 'auto').
        target_lang: Target language code.
        backend: Translation backend ('google', 'nllb', 'm2m100').

    Returns:
        Segments with translated_text filled.
    """
    if source_lang == target_lang:
        for seg in segments:
            seg.translated_text = seg.text
        return segments

    if backend == "google":
        _translate_google(segments, source_lang, target_lang)
    elif backend == "nllb":
        _translate_nllb(segments, source_lang, target_lang)
    else:
        _translate_google(segments, source_lang, target_lang)

    # Post-process: preserve sentence-ending punctuation style
    for seg in segments:
        seg.translated_text = seg.translated_text.strip()

    logger.info("Translated %d segments (%s → %s)", len(segments), source_lang, target_lang)
    return segments


def _translate_google(segments: list[Segment], src: str, tgt: str) -> None:
    """Translate using googletrans (free, no API key)."""
    try:
        from deep_translator import GoogleTranslator

        src_code = "" if src == "auto" else _LANG_MAP.get(src, src)
        tgt_code = _LANG_MAP.get(tgt, tgt)

        translator = GoogleTranslator(source=src_code or "auto", target=tgt_code)

        # Batch translate for efficiency — join with separator
        BATCH_SIZE = 50
        texts = [seg.text for seg in segments]

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            # GoogleTranslator supports list input
            try:
                results = translator.translate_batch(batch)
                for j, result in enumerate(results):
                    segments[i + j].translated_text = result if isinstance(result, str) else result.text
            except Exception:
                # Fallback: translate one by one
                for j, text in enumerate(batch):
                    try:
                        segments[i + j].translated_text = translator.translate(text)
                    except Exception as e:
                        logger.warning("Translation failed for segment %d: %s", i + j, e)
                        segments[i + j].translated_text = text  # keep original on failure

    except ImportError:
        logger.error("deep-translator not installed. Install with: pip install deep-translator")
        raise


def _translate_nllb(segments: list[Segment], src: str, tgt: str) -> None:
    """Translate using Meta NLLB-200 via transformers."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = "facebook/nllb-200-distilled-600M"
        logger.info("Loading NLLB model: %s", model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # NLLB language codes
        nllb_codes = {
            "en": "eng_Latn", "hi": "hin_Deva", "es": "spa_Latn",
            "fr": "fra_Latn", "de": "deu_Latn", "ja": "jpn_Jpan",
            "ko": "kor_Hang", "zh": "zho_Hans", "ar": "arb_Arab",
            "pt": "por_Latn", "ru": "rus_Cyrl", "it": "ita_Latn",
            "bn": "ben_Beng", "gu": "guj_Gujr", "ta": "tam_Taml",
            "te": "tel_Telu", "mr": "mod_Deva", "pa": "pan_Guru",
            "ur": "urd_Arab", "tr": "tur_Latn", "nl": "nld_Latn",
            "pl": "pol_Latn", "th": "tha_Thai", "vi": "vie_Latn",
            "id": "ind_Latn",
        }

        src_code = nllb_codes.get(src, "eng_Latn")
        tgt_code = nllb_codes.get(tgt, "eng_Latn")

        for seg in segments:
            inputs = tokenizer(seg.text, return_tensors="pt", max_length=512, truncation=True)
            inputs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(tgt_code)
            outputs = model.generate(**inputs, max_length=512)
            seg.translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    except ImportError:
        logger.error("transformers not installed. Falling back to Google translate.")
        _translate_google(segments, src, tgt)


def shorten_translation(text: str, target_ratio: float = 0.85) -> str:
    """Attempt to shorten translated text while preserving meaning.

    Used for duration matching when translated text is longer than original.
    """
    # Simple heuristic: remove filler words and redundancy
    fillers = {
        "actually": "", "basically": "", "literally": "", "just": "",
        "really": "", "very": "", "quite": "", "rather": "",
        "somewhat": "", "definitely": "", "certainly": "",
    }
    words = text.split()
    result = []
    for w in words:
        lower = w.lower().strip(".,!?;")
        if lower in fillers:
            continue
        result.append(w)

    shortened = " ".join(result)
    # If still too long, remove adjectives/adverbs (heuristic: words ending in -ly, -ful, -ous)
    if len(shortened.split()) > len(words) * target_ratio:
        result2 = [w for w in result if not w.endswith(("ly", "ful", "ous", "ive"))]
        if result2:
            shortened = " ".join(result2)

    return shortened


def expand_translation(text: str) -> str:
    """Slightly expand a short translation to fill more time.

    Used for duration matching when translated text is shorter than original.
    """
    # Add polite filler that doesn't change meaning
    expansions = {
        ", ": ", actually, ",
        ".": " you know.",
        "!": " Indeed!",
        "?": " I wonder?",
    }
    return text  # Conservative: don't hallucinate content
