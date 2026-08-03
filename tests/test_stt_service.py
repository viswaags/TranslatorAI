"""Model-independent tests for the canonical STT service."""

import inspect
import math
import struct
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path

BACKEND_ROOT = Path(__file__).parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from ai_translator_api.core.config import settings  # noqa: E402
from ai_translator_api.services.stt.stt_service import (  # noqa: E402
    AudioMetadata,
    STTBackend,
    STTCorruptedAudioError,
    STTFileNotFoundError,
    STTService,
    STTUnsupportedAudioError,
    TranscriptionResult,
)


class FakeBackend(STTBackend):
    def __init__(self):
        self.calls = []
        self.unloaded = False
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe(self, audio_path, metadata):
        self.calls.append((audio_path, metadata))
        if self.block:
            self.started.set()
            self.release.wait(timeout=2)
        return TranscriptionResult(
            text="hello",
            whisper_lang="en",
            indictrans_lang="eng_Latn",
            lang_name="English",
            confidence=0.95,
            duration_seconds=metadata.duration_seconds,
            segment_count=1,
        )

    def unload(self):
        self.unloaded = True


def write_wav(path: Path, sample_rate: int = 16000, duration: float = 0.2):
    frame_count = int(sample_rate * duration)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        for index in range(frame_count):
            sample = int(1000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            output.writeframesraw(struct.pack("<h", sample))


class STTServiceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.audio = Path(self.tempdir.name) / "test.wav"
        write_wav(self.audio)
        self.backend = FakeBackend()
        self.service = STTService(backend=self.backend)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_structured_result_contract(self):
        result = self.service._transcribe_sync(self.audio).to_public_dict()
        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["transcript"], "hello")
        self.assertEqual(result["engine"], "faster-whisper")
        self.assertEqual(result["indictrans_lang"], "eng_Latn")
        self.assertIsNone(result["error"])

    def test_missing_file_is_explicit(self):
        with self.assertRaises(STTFileNotFoundError):
            self.service._transcribe_sync(Path(self.tempdir.name) / "missing.wav")

    def test_unsupported_extension_is_explicit(self):
        path = Path(self.tempdir.name) / "audio.txt"
        path.write_bytes(b"audio")
        with self.assertRaises(STTUnsupportedAudioError):
            self.service._transcribe_sync(path)

    def test_corrupted_audio_is_explicit(self):
        path = Path(self.tempdir.name) / "broken.wav"
        path.write_bytes(b"not wave data")
        with self.assertRaises(STTCorruptedAudioError):
            self.service._transcribe_sync(path)

    def test_batch_contract(self):
        results = self.service._transcribe_batch_sync([self.audio, self.audio])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["text"] == "hello" for result in results))

    def test_excessive_duration_is_rejected(self):
        original = settings.STT_MAX_DURATION_SECONDS
        settings.STT_MAX_DURATION_SECONDS = 0.01
        try:
            with self.assertRaises(Exception) as raised:
                self.service._transcribe_sync(self.audio)
            self.assertEqual(type(raised.exception).__name__, "STTAudioTooLongError")
        finally:
            settings.STT_MAX_DURATION_SECONDS = original

    def test_unload_waits_for_inference(self):
        self.backend.block = True
        inference = threading.Thread(
            target=self.service._transcribe_sync, args=(self.audio,)
        )
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
        self.assertTrue(inspect.iscoroutinefunction(self.service.transcribe))
        self.assertTrue(inspect.iscoroutinefunction(self.service.transcribe_batch))


if __name__ == "__main__":
    unittest.main()
