"""Tests for streamed, managed upload persistence."""

import asyncio
import io
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

BACKEND_ROOT = Path(__file__).parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from ai_translator_api.utils.uploads import (  # noqa: E402
    UploadPolicy,
    managed_upload,
    save_upload,
)


def make_upload(
    content: bytes,
    filename: str = "sample.wav",
    content_type: str = "audio/wav",
) -> UploadFile:
    return UploadFile(
        io.BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


class TrackingUpload:
    def __init__(self, content: bytes, filename: str = "sample.wav"):
        self.filename = filename
        self.content_type = "audio/wav"
        self._content = content
        self._offset = 0
        self.read_sizes = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            raise AssertionError("Upload utility attempted an unbounded read")
        start = self._offset
        self._offset += size
        return self._content[start : self._offset]


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.directory = Path(self.tempdir.name)
        self.policy = UploadPolicy(
            name="audio",
            upload_dir=self.directory,
            max_size_bytes=16,
            chunk_size_bytes=4,
            allowed_extensions=frozenset({".wav"}),
            allowed_mime_types=frozenset({"audio/wav"}),
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_valid_upload_is_streamed_to_disk(self):
        path = asyncio.run(
            save_upload(make_upload(b"abcdefgh"), self.policy)
        )
        self.assertEqual(path.read_bytes(), b"abcdefgh")
        path.unlink()

    def test_oversized_upload_is_rejected_and_removed(self):
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(save_upload(make_upload(b"x" * 17), self.policy))
        self.assertEqual(raised.exception.status_code, 413)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_empty_upload_is_rejected_and_removed(self):
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(save_upload(make_upload(b""), self.policy))
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_unsupported_extension_is_rejected_before_allocation(self):
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(
                save_upload(
                    make_upload(b"data", filename="sample.exe"),
                    self.policy,
                )
            )
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_cleanup_after_processing_failure(self):
        upload = make_upload(b"data")
        captured = None

        async def process():
            nonlocal captured
            with self.assertRaises(RuntimeError):
                async with managed_upload(upload, self.policy) as path:
                    captured = Path(path)
                    self.assertTrue(captured.is_file())
                    raise RuntimeError("processing failed")

        asyncio.run(process())
        self.assertIsNotNone(captured)
        self.assertFalse(captured.exists())

    def test_cleanup_after_successful_processing(self):
        upload = make_upload(b"data")
        captured = None

        async def process():
            nonlocal captured
            async with managed_upload(upload, self.policy) as path:
                captured = Path(path)
                self.assertEqual(captured.read_bytes(), b"data")

        asyncio.run(process())
        self.assertIsNotNone(captured)
        self.assertFalse(captured.exists())

    def test_filenames_are_unique_and_preserve_valid_suffix(self):
        first = asyncio.run(
            save_upload(make_upload(b"first"), self.policy)
        )
        second = asyncio.run(
            save_upload(make_upload(b"second"), self.policy)
        )
        try:
            self.assertNotEqual(first.name, second.name)
            self.assertEqual(first.suffix, ".wav")
            self.assertEqual(second.suffix, ".wav")
        finally:
            first.unlink()
            second.unlink()

    def test_streaming_limit_stops_before_reading_full_upload(self):
        upload = TrackingUpload(b"x" * 100)
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(save_upload(upload, self.policy))
        self.assertEqual(raised.exception.status_code, 413)
        self.assertEqual(upload.read_sizes, [4, 4, 4, 4, 4])
        self.assertNotIn(-1, upload.read_sizes)
        self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
