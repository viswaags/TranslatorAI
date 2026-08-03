"""
LanguageDetector
================
Unified language detection with Unicode-range heuristics (fast, no model needed)
+ langdetect fallback for ambiguous cases.

Returns IndicTrans2 language codes throughout.
"""

import logging
import re
from typing import Optional

from ai_translator_api.utils.languages import (
    ENGLISH_CODE,
    HINDI_CODE,
    ISO_TO_INDICTRANS,
    LANGUAGE_NAMES,
    MARATHI_CODE,
)

logger = logging.getLogger(__name__)


LANG_CODE_MAP = ISO_TO_INDICTRANS

# Unicode block → IndicTrans2 code
UNICODE_RANGES = [
    (r"[\u0B80-\u0BFF]", ISO_TO_INDICTRANS["ta"]),  # Tamil
    (r"[\u0C00-\u0C7F]", ISO_TO_INDICTRANS["te"]),  # Telugu
    (r"[\u0C80-\u0CFF]", ISO_TO_INDICTRANS["kn"]),  # Kannada
    (r"[\u0D00-\u0D7F]", ISO_TO_INDICTRANS["ml"]),  # Malayalam
    (r"[\u0980-\u09FF]", ISO_TO_INDICTRANS["bn"]),  # Bengali/Assamese
    (r"[\u0A80-\u0AFF]", ISO_TO_INDICTRANS["gu"]),  # Gujarati
    (r"[\u0A00-\u0A7F]", ISO_TO_INDICTRANS["pa"]),  # Punjabi
    (r"[\u0B00-\u0B7F]", ISO_TO_INDICTRANS["or"]),  # Odia
    (r"[\u0600-\u06FF]", ISO_TO_INDICTRANS["ur"]),  # Urdu
    (r"[\u0900-\u097F]", HINDI_CODE),  # Hindi or Marathi
]


class LanguageDetector:
    """
    Fast language detector — unicode-range first, langdetect fallback.
    No model loading required.
    """

    def detect(self, text: str) -> str:
        """
        Detect language from text.

        Returns:
            IndicTrans2 language code (e.g. "tam_Taml")
            Falls back to "eng_Latn" if detection fails.
        """
        if not text or not text.strip():
            return ENGLISH_CODE

        # Step 1: Unicode range analysis (fast, no library)
        unicode_result = self._detect_by_unicode(text)
        if unicode_result and unicode_result != ENGLISH_CODE:
            logger.debug("Language detected via unicode: %s", unicode_result)
            return unicode_result

        # Step 2: langdetect (handles Latin-script ambiguity well)
        langdetect_result = self._detect_by_langdetect(text)
        if langdetect_result:
            logger.debug("Language detected via langdetect: %s", langdetect_result)
            return langdetect_result

        # Step 3: Default
        logger.debug("Language detection defaulted to eng_Latn")
        return ENGLISH_CODE

    def detect_with_confidence(self, text: str) -> tuple[str, float]:
        """
        Returns (lang_code, confidence_score).
        Confidence is estimated — unicode detection is 0.95, langdetect varies.
        """
        if not text or not text.strip():
            return ENGLISH_CODE, 0.5

        unicode_result = self._detect_by_unicode(text)
        if unicode_result and unicode_result != ENGLISH_CODE:
            return unicode_result, 0.95

        try:
            from langdetect import detect_langs
            results = detect_langs(text)
            if results:
                top = results[0]
                code = LANG_CODE_MAP.get(top.lang, ENGLISH_CODE)
                return code, round(top.prob, 3)
        except Exception:
            pass

        return ENGLISH_CODE, 0.5

    def language_name(self, code: str) -> str:
        """Return human-readable name for a language code."""
        return LANGUAGE_NAMES.get(code, code)

    def _detect_by_unicode(self, text: str) -> Optional[str]:
        total = len(re.sub(r"\s", "", text))
        if total == 0:
            return None

        scores: dict[str, int] = {}
        for pattern, lang in UNICODE_RANGES:
            count = len(re.findall(pattern, text))
            if count > 0:
                scores[lang] = count

        if not scores:
            latin = len(re.findall(r"[A-Za-z]", text))
            if latin / total > 0.4:
                return ENGLISH_CODE
            return None

        best_lang = max(scores, key=scores.get)
        best_count = scores[best_lang]

        # Devanagari: Hindi vs Marathi disambiguation
        if best_lang == HINDI_CODE:
            mr_check = self._detect_by_langdetect(text)
            if mr_check == MARATHI_CODE:
                return MARATHI_CODE

        if best_count / total >= 0.25:
            return best_lang

        return None

    def _detect_by_langdetect(self, text: str) -> Optional[str]:
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
            code = detect(text)
            return LANG_CODE_MAP.get(code, ENGLISH_CODE)
        except Exception as e:
            logger.debug("langdetect failed: %s", e)
            return None
