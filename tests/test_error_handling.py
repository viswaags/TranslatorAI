"""Tests for shared service errors and centralized HTTP mapping."""

import asyncio
import json
import sys
import unittest
from pathlib import Path

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

BACKEND_ROOT = Path(__file__).parents[1] / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from ai_translator_api.api.error_handlers import (  # noqa: E402
    http_error_handler,
    request_validation_error_handler,
    service_error_handler,
    service_error_status,
    unexpected_error_handler,
)
from ai_translator_api.core.errors import (  # noqa: E402
    InferenceError,
    LifecycleError,
    ModelLoadError,
    ModelUnavailableError,
    OverloadError,
    ServiceError,
    UnsupportedInputError,
    ValidationError,
)
from ai_translator_api.main import app  # noqa: E402
from ai_translator_api.services.ocr.ocr_service import (  # noqa: E402
    OCRInferenceError,
    OCRLifecycleError,
    OCRModelLoadError,
    OCRModelUnavailableError,
    OCRUnsupportedImageError,
    OCRValidationError,
)
from ai_translator_api.services.stt.stt_service import (  # noqa: E402
    STTInferenceError,
    STTLifecycleError,
    STTModelLoadError,
    STTModelUnavailableError,
    STTUnsupportedAudioError,
    STTValidationError,
)
from ai_translator_api.services.translation.translation_service import (  # noqa: E402
    TranslationInferenceError,
    TranslationLifecycleError,
    TranslationModelLoadError,
    TranslationModelUnavailableError,
    TranslationValidationError,
    UnsupportedLanguageError,
)
from ai_translator_api.services.tts.tts_service import (  # noqa: E402
    TTSBackendUnavailableError,
    TTSInferenceError,
    TTSLifecycleError,
    TTSModelUnavailableError,
    TTSUnsupportedLanguageError,
    TTSValidationError,
)


class ErrorInheritanceTests(unittest.TestCase):
    def test_subsystem_validation_errors_keep_names_and_shared_base(self):
        for error_type in (
            TranslationValidationError,
            OCRValidationError,
            STTValidationError,
            TTSValidationError,
        ):
            self.assertTrue(issubclass(error_type, ValidationError))
            self.assertTrue(issubclass(error_type, ValueError))

    def test_subsystem_categories_have_shared_bases(self):
        groups = (
            (
                ModelUnavailableError,
                TranslationModelUnavailableError,
                OCRModelUnavailableError,
                STTModelUnavailableError,
                TTSModelUnavailableError,
                TTSBackendUnavailableError,
            ),
            (
                ModelLoadError,
                TranslationModelLoadError,
                OCRModelLoadError,
                STTModelLoadError,
            ),
            (
                InferenceError,
                TranslationInferenceError,
                OCRInferenceError,
                STTInferenceError,
                TTSInferenceError,
            ),
            (
                LifecycleError,
                TranslationLifecycleError,
                OCRLifecycleError,
                STTLifecycleError,
                TTSLifecycleError,
            ),
            (
                UnsupportedInputError,
                UnsupportedLanguageError,
                OCRUnsupportedImageError,
                STTUnsupportedAudioError,
                TTSUnsupportedLanguageError,
            ),
        )
        for shared_base, *subsystem_errors in groups:
            for error_type in subsystem_errors:
                self.assertTrue(issubclass(error_type, shared_base))
                self.assertTrue(issubclass(error_type, ServiceError))


class ErrorHTTPMappingTests(unittest.TestCase):
    def test_status_mapping(self):
        cases = (
            (ValidationError("bad"), 422, "validation"),
            (UnsupportedInputError("bad"), 400, "unsupported_input"),
            (ModelUnavailableError("missing"), 503, "model_unavailable"),
            (ModelLoadError("broken"), 500, "model_load"),
            (InferenceError("failed"), 500, "inference"),
            (LifecycleError("stopping"), 503, "lifecycle"),
            (OverloadError("busy"), 429, "overload"),
            (ServiceError("unknown"), 500, "service"),
        )
        for error, status, category in cases:
            self.assertEqual(service_error_status(error), (status, category))

    def test_service_error_response_is_structured(self):
        response = asyncio.run(
            service_error_handler(None, STTModelUnavailableError("missing"))
        )
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            payload,
            {
                "success": False,
                "error": {
                    "type": "STTModelUnavailableError",
                    "category": "model_unavailable",
                    "message": "missing",
                },
            },
        )

    def test_existing_http_exceptions_use_same_envelope(self):
        response = asyncio.run(
            http_error_handler(
                None, HTTPException(status_code=413, detail="too large")
            )
        )
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 413)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"]["category"], "http")
        self.assertEqual(payload["error"]["message"], "too large")

    def test_request_validation_uses_same_envelope(self):
        error = RequestValidationError(
            [
                {
                    "type": "missing",
                    "loc": ("body", "text"),
                    "msg": "Field required",
                    "input": {},
                }
            ]
        )
        response = asyncio.run(request_validation_error_handler(None, error))
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 422)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["error"]["category"], "validation")
        self.assertIn("details", payload["error"])

    def test_handlers_are_registered_on_production_app(self):
        self.assertIs(app.exception_handlers[ServiceError], service_error_handler)
        self.assertIs(
            app.exception_handlers[RequestValidationError],
            request_validation_error_handler,
        )
        self.assertIs(
            app.exception_handlers[StarletteHTTPException],
            http_error_handler,
        )
        self.assertIs(
            app.exception_handlers[Exception], unexpected_error_handler
        )


if __name__ == "__main__":
    unittest.main()
