"""Model-independent tests for the canonical OCR service."""

import inspect
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image

BACKEND_ROOT = Path(__file__).parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from ai_translator_api.services.ocr.ocr_service import (  # noqa: E402
    OCRBackend,
    OCRResult,
    OCRService,
)


class FakeBackend(OCRBackend):
    def __init__(self):
        self.calls = []
        self.unloaded = False
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()

    def extract(self, image_path, language_hint):
        self.calls.append((image_path, language_hint))
        if self.block:
            self.started.set()
            self.release.wait(timeout=2)
        return OCRResult(
            text="Hello",
            raw_text="Hello",
            language=language_hint or "eng_Latn",
            confidence=0.9,
            line_count=1,
        )

    def unload(self):
        self.unloaded = True


class OCRServiceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.image = Path(self.tempdir.name) / "test.png"
        Image.new("RGB", (32, 16), "white").save(self.image)
        self.backend = FakeBackend()
        self.service = OCRService(backend=self.backend)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_structured_result_contract(self):
        result = self.service._extract_sync(
            self.image, "eng_Latn"
        ).to_public_dict()
        self.assertEqual(result["text"], "Hello")
        self.assertEqual(result["engine_used"], "paddle")
        self.assertEqual(result["language"], "eng_Latn")
        self.assertIsNone(result["error"])
        self.assertEqual(result["line_count"], 1)

    def test_missing_file_is_explicit(self):
        result = self.service._extract_sync(
            Path(self.tempdir.name) / "missing.png", None
        ).to_public_dict()
        self.assertEqual(result["error_type"], "OCRFileNotFoundError")
        self.assertEqual(result["text"], "")

    def test_invalid_extension_is_explicit(self):
        bad = Path(self.tempdir.name) / "image.txt"
        bad.write_text("not an image")
        result = self.service._extract_sync(bad, None).to_public_dict()
        self.assertEqual(result["error_type"], "OCRUnsupportedImageError")

    def test_corrupted_image_is_explicit(self):
        bad = Path(self.tempdir.name) / "broken.png"
        bad.write_bytes(b"not a PNG")
        result = self.service._extract_sync(bad, None).to_public_dict()
        self.assertEqual(result["error_type"], "OCRCorruptedImageError")

    def test_invalid_language_hint_is_explicit(self):
        result = self.service._extract_sync(
            self.image, "invalid"
        ).to_public_dict()
        self.assertEqual(result["error_type"], "OCRValidationError")

    def test_batch_contract(self):
        results = self.service._extract_batch_sync(
            [self.image, self.image], "eng_Latn"
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["text"] == "Hello" for result in results))

    def test_unload_waits_for_inference(self):
        self.backend.block = True

        def run_extract():
            self.service._extract_sync(self.image, "eng_Latn")

        inference = threading.Thread(target=run_extract)
        inference.start()
        self.assertTrue(self.backend.started.wait(timeout=1))

        unloading = threading.Thread(target=self.service.unload)
        unloading.start()
        time.sleep(0.05)
        self.assertTrue(unloading.is_alive())

        self.backend.release.set()
        inference.join(timeout=1)
        unloading.join(timeout=1)
        self.assertTrue(self.backend.unloaded)

    def test_public_api_remains_async(self):
        self.assertTrue(inspect.iscoroutinefunction(self.service.extract))
        self.assertTrue(inspect.iscoroutinefunction(self.service.extract_batch))


if __name__ == "__main__":
    unittest.main()
