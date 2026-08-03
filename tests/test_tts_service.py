"""Model-independent tests for the canonical Piper TTS service."""

import base64
import inspect
import sys
import threading
import time
import unittest
from pathlib import Path

BACKEND_ROOT = Path(__file__).parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from ai_translator_api.core.config import settings  # noqa: E402
from ai_translator_api.services.tts.tts_service import (  # noqa: E402
    BackendSynthesis,
    TTSBackend,
    TTSService,
)


class FakeBackend(TTSBackend):
    def __init__(self):
        self.calls = []
        self.unloaded = False
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()

    def synthesize(self, text, language, speed):
        self.calls.append((text, language, speed))
        if self.block:
            self.started.set()
            self.release.wait(timeout=2)
        return BackendSynthesis(
            audio_bytes=b"RIFFfake-wav",
            sample_rate=22050,
            duration_seconds=0.25,
            voice="test-voice",
        )

    def unload(self):
        self.unloaded = True


class TTSServiceTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.service = TTSService(backend=self.backend)

    def test_structured_result_preserves_route_contract(self):
        result = self.service._synthesise_sync(
            "hello", "eng_Latn"
        ).to_public_dict()
        self.assertTrue(result["success"])
        self.assertEqual(result["engine_used"], "piper")
        self.assertEqual(result["engine"], "piper")
        self.assertEqual(result["language"], "eng_Latn")
        self.assertEqual(result["voice"], "test-voice")
        self.assertEqual(base64.b64decode(result["audio_base64"]), b"RIFFfake-wav")
        self.assertIsNone(result["error"])
        self.assertIsNone(result["error_type"])

    def test_empty_text_is_explicit(self):
        result = self.service._synthesise_sync(
            "   ", "eng_Latn"
        ).to_public_dict()
        self.assertEqual(result["error_type"], "TTSEmptyTextError")

    def test_unsupported_language_is_explicit(self):
        result = self.service._synthesise_sync(
            "hello", "invalid"
        ).to_public_dict()
        self.assertEqual(result["error_type"], "TTSUnsupportedLanguageError")

    def test_invalid_voice_selection_is_explicit(self):
        result = self.service._synthesise_sync(
            "hello", "eng_Latn", engine="espeak"
        ).to_public_dict()
        self.assertEqual(result["error_type"], "TTSInvalidVoiceError")

    def test_excessively_long_text_is_explicit(self):
        original = settings.TTS_MAX_TEXT_CHARS
        settings.TTS_MAX_TEXT_CHARS = 3
        try:
            result = self.service._synthesise_sync(
                "long", "eng_Latn"
            ).to_public_dict()
            self.assertEqual(result["error_type"], "TTSTextTooLongError")
        finally:
            settings.TTS_MAX_TEXT_CHARS = original

    def test_batch_contract(self):
        results = self.service._synthesise_batch_sync(
            ["one", "two"], "eng_Latn"
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(item["success"] for item in results))
        self.assertTrue(all(item["engine_used"] == "piper" for item in results))

    def test_missing_local_voice_fails_without_download(self):
        service = TTSService()
        original_dir = settings.PIPER_VOICES_DIR
        original_map = settings.PIPER_VOICE_MAP
        settings.PIPER_VOICES_DIR = ""
        settings.PIPER_VOICE_MAP = ""
        try:
            result = service._synthesise_sync(
                "hello", "eng_Latn"
            ).to_public_dict()
            self.assertEqual(result["error_type"], "TTSModelUnavailableError")
            self.assertIn("runtime downloads are disabled", result["error"])
        finally:
            settings.PIPER_VOICES_DIR = original_dir
            settings.PIPER_VOICE_MAP = original_map

    def test_unload_waits_for_synthesis(self):
        self.backend.block = True

        synthesis = threading.Thread(
            target=self.service._synthesise_sync,
            args=("hello", "eng_Latn"),
        )
        synthesis.start()
        self.assertTrue(self.backend.started.wait(timeout=1))

        unloading = threading.Thread(target=self.service.unload)
        unloading.start()
        time.sleep(0.05)
        self.assertTrue(unloading.is_alive())

        self.backend.release.set()
        synthesis.join(timeout=1)
        unloading.join(timeout=1)
        self.assertTrue(self.backend.unloaded)

    def test_public_api_remains_async(self):
        self.assertTrue(inspect.iscoroutinefunction(self.service.synthesise))
        self.assertTrue(inspect.iscoroutinefunction(self.service.synthesise_batch))


if __name__ == "__main__":
    unittest.main()
