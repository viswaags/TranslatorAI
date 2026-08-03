"""Consistency tests for the canonical backend language catalog."""

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

BACKEND_ROOT = Path(__file__).parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from ai_translator_api.services.ocr.ocr_service import (  # noqa: E402
    INDICTRANS_TO_PADDLE,
    SUPPORTED_LANGUAGES as OCR_VALIDATION_LANGUAGES,
)
from ai_translator_api.services.stt.stt_service import (  # noqa: E402
    LANGUAGE_NAMES as STT_LANGUAGE_NAMES,
    WHISPER_TO_INDICTRANS,
)
from ai_translator_api.services.translation.translation_service import (  # noqa: E402
    SUPPORTED_LANGUAGES as TRANSLATION_LANGUAGES,
)
from ai_translator_api.services.tts.tts_service import (  # noqa: E402
    PIPER_LANGUAGE_PREFIXES,
    SUPPORTED_LANGUAGES as TTS_LANGUAGES,
)
from ai_translator_api.utils.language_detector import (  # noqa: E402
    LANG_CODE_MAP,
    LANGUAGE_NAMES as DETECTOR_LANGUAGE_NAMES,
)
from ai_translator_api.utils.languages import (  # noqa: E402
    LANGUAGES,
    LANGUAGE_NAMES,
    OCR_LANGUAGE_MAP,
    STT_LANGUAGE_CODES,
    STT_TO_INDICTRANS,
    TRANSLATION_LANGUAGE_CODES,
    TTS_LANGUAGE_CODES,
    TTS_VOICE_ALIASES,
)


class LanguageCatalogTests(unittest.TestCase):
    def test_all_public_language_sets_remain_consistent(self):
        expected = {language.code for language in LANGUAGES}
        self.assertEqual(expected, set(TRANSLATION_LANGUAGE_CODES))
        self.assertEqual(expected, set(TRANSLATION_LANGUAGES))
        self.assertEqual(expected, set(OCR_VALIDATION_LANGUAGES))
        self.assertEqual(expected, set(STT_LANGUAGE_CODES))
        self.assertEqual(expected, set(TTS_LANGUAGE_CODES))
        self.assertEqual(expected, set(TTS_LANGUAGES))

    def test_engine_aliases_are_derived_from_catalog(self):
        self.assertEqual(WHISPER_TO_INDICTRANS, STT_TO_INDICTRANS)
        self.assertEqual(LANG_CODE_MAP, STT_TO_INDICTRANS)
        self.assertEqual(INDICTRANS_TO_PADDLE, OCR_LANGUAGE_MAP)
        self.assertEqual(PIPER_LANGUAGE_PREFIXES, TTS_VOICE_ALIASES)

    def test_display_names_are_identical_and_odia_is_correct(self):
        self.assertEqual(DETECTOR_LANGUAGE_NAMES, LANGUAGE_NAMES)
        self.assertEqual(STT_LANGUAGE_NAMES, LANGUAGE_NAMES)
        self.assertEqual(LANGUAGE_NAMES["ory_Orya"], "Odia")
        self.assertNotIn("ury_Orya", LANGUAGE_NAMES)

    def test_catalog_and_derived_mappings_are_immutable(self):
        with self.assertRaises(TypeError):
            LANGUAGE_NAMES["eng_Latn"] = "Changed"
        with self.assertRaises(FrozenInstanceError):
            LANGUAGES[0].display_name = "Changed"

    def test_ocr_capability_subset_is_preserved(self):
        self.assertEqual(
            set(OCR_LANGUAGE_MAP),
            {
                "eng_Latn",
                "tam_Taml",
                "hin_Deva",
                "tel_Telu",
                "kan_Knda",
                "mar_Deva",
            },
        )
        self.assertEqual(OCR_LANGUAGE_MAP["kan_Knda"], ("ka", "ka"))


if __name__ == "__main__":
    unittest.main()
