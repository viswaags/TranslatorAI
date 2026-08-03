"""TranslationService unit tests that do not load model weights."""

import sys
import threading
import time
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from ai_translator_api.services.translation.translation_service import (  # noqa: E402
    TranslationBackend,
    TranslationService,
    TranslationValidationError,
    UnsupportedLanguageError,
)
from ai_translator_api.core.lifecycle import LazyBackendSlot  # noqa: E402


class FakeBackend(TranslationBackend):
    def __init__(self, prefix):
        self.prefix = prefix
        self.calls = []
        self.unloaded = False
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()

    def translate(self, sentences, src_lang, tgt_lang):
        self.calls.append((list(sentences), src_lang, tgt_lang))
        if self.block:
            self.started.set()
            self.release.wait(timeout=2)
        return [f"{self.prefix}:{text}" for text in sentences]

    def unload(self):
        self.unloaded = True


class TranslationServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = TranslationService()
        self.service._en_indic_slot = LazyBackendSlot(
            lambda: FakeBackend("en-indic"),
            lambda backend: backend.unload(),
            FakeBackend("en-indic"),
        )
        self.service._indic_en_slot = LazyBackendSlot(
            lambda: FakeBackend("indic-en"),
            lambda backend: backend.unload(),
            FakeBackend("indic-en"),
        )

    def test_empty_text_is_rejected(self):
        with self.assertRaises(TranslationValidationError):
            self.service.translate("   ", "tam_Taml", "eng_Latn")

    def test_invalid_language_is_rejected(self):
        with self.assertRaises(UnsupportedLanguageError):
            self.service.translate("hello", "invalid", "eng_Latn")

    def test_same_language_is_structured_passthrough(self):
        result = self.service.translate("hello", "eng_Latn", "eng_Latn")
        self.assertEqual(result["translated_text"], "hello")
        self.assertEqual(result["engine"], "passthrough")
        self.assertEqual(result["route"], "passthrough")

    def test_direct_translation_preserves_contract(self):
        result = self.service.translate("hello", "tam_Taml", "eng_Latn")
        self.assertEqual(result["translated_text"], "en-indic:hello")
        self.assertEqual(result["engine"], "indictrans2")
        self.assertEqual(result["route"], "direct")

    def test_batch_is_structured_and_batched(self):
        results = self.service.translate_batch(
            ["one", "two"], "tam_Taml", "eng_Latn"
        )
        self.assertEqual(
            [item["translated_text"] for item in results],
            ["en-indic:one", "en-indic:two"],
        )
        self.assertTrue(all(item["engine"] == "indictrans2" for item in results))
        self.assertEqual(
            self.service._get_en_indic().calls,
            [(["one", "two"], "eng_Latn", "tam_Taml")],
        )

    def test_indic_pair_uses_english_pivot(self):
        result = self.service.translate("source", "hin_Deva", "tam_Taml")
        self.assertEqual(result["translated_text"], "en-indic:indic-en:source")
        self.assertEqual(result["route"], "pivot")

    def test_unload_waits_for_active_inference(self):
        backend = self.service._get_en_indic()
        backend.block = True
        inference = threading.Thread(
            target=self.service.translate,
            args=("hello", "tam_Taml", "eng_Latn"),
        )
        inference.start()
        self.assertTrue(backend.started.wait(timeout=1))

        unloading = threading.Thread(target=self.service.unload)
        unloading.start()
        time.sleep(0.05)
        self.assertTrue(unloading.is_alive())

        backend.release.set()
        inference.join(timeout=1)
        unloading.join(timeout=1)
        self.assertTrue(backend.unloaded)


if __name__ == "__main__":
    unittest.main()
