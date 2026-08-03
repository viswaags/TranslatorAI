"""
OCR Pipeline
============
POST /api/v1/ocr/extract
  - Upload image → OCR only (no translation)
  - Returns extracted text + detected language

POST /api/v1/ocr/extract-and-translate
  - Upload image + target_lang form field
  - OCR → language detection → translation → optional TTS
  - Source language is auto-detected from the image

Both endpoints support optional source_lang_hint to improve OCR accuracy.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse

from ai_translator_api.core.config import settings
from ai_translator_api.core.registry import ServiceRegistry
from ai_translator_api.models.schemas import (
    OCRExtractResponse,
    OCRTranslateResponse,
    TTSResult,
)
from ai_translator_api.services.ocr.ocr_service import (
    OCRCorruptedImageError,
    OCREmptyPageError,
    OCRError,
    OCRFileNotFoundError,
    OCRInferenceError,
    OCRLifecycleError,
    OCRModelLoadError,
    OCRModelUnavailableError,
    OCRUnsupportedImageError,
    OCRValidationError,
)
from ai_translator_api.utils.language_detector import LANGUAGE_NAMES
from ai_translator_api.utils.uploads import UploadPolicy, managed_upload

logger = logging.getLogger(__name__)
router = APIRouter()

OCR_UPLOAD_POLICY = UploadPolicy(
    name="image",
    upload_dir=Path(settings.UPLOAD_DIR),
    max_size_bytes=settings.OCR_MAX_UPLOAD_SIZE_BYTES,
    chunk_size_bytes=settings.UPLOAD_CHUNK_SIZE_BYTES,
    allowed_extensions=frozenset(settings.OCR_SUPPORTED_EXTENSIONS),
    allowed_mime_types=frozenset(
        {
            "image/jpeg",
            "image/png",
            "image/bmp",
            "image/tiff",
            "image/webp",
        }
    ),
)

OCR_ERROR_TYPES = {
    error_type.__name__: error_type
    for error_type in (
        OCRValidationError,
        OCRFileNotFoundError,
        OCRUnsupportedImageError,
        OCRCorruptedImageError,
        OCREmptyPageError,
        OCRModelUnavailableError,
        OCRModelLoadError,
        OCRInferenceError,
        OCRLifecycleError,
    )
}


def get_registry(request: Request) -> ServiceRegistry:
    return request.app.state.registry


def _raise_for_ocr_failure(result: dict) -> None:
    if not result.get("error") or result.get("text"):
        return
    error_type = OCR_ERROR_TYPES.get(result.get("error_type"), OCRError)
    raise error_type(result["error"])


@router.post(
    "/extract",
    response_model=OCRExtractResponse,
    summary="Extract text from image (OCR only)",
)
async def ocr_extract(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP, TIFF, WebP)"),
    source_lang_hint: Optional[str] = Form(default=None, description="Optional language hint (IndicTrans2 code)"),
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    **OCR only — no translation.**

    Extracts text from an uploaded image using local PaddleOCR detection,
    classification, and language-specific recognition models.

    Returns extracted text + detected language in IndicTrans2 format.
    """
    t0 = time.monotonic()

    async with managed_upload(file, OCR_UPLOAD_POLICY) as file_path:
        # Load OCR service (lazy)
        ocr_svc = await registry.get_ocr_service()

        ocr_result = await ocr_svc.extract(file_path, language_hint=source_lang_hint)

        _raise_for_ocr_failure(ocr_result)

        lang_code = ocr_result["language"]

        return OCRExtractResponse(
            success=True,
            extracted_text=ocr_result["text"],
            detected_lang=lang_code,
            detected_lang_name=LANGUAGE_NAMES.get(lang_code, lang_code),
            ocr_confidence=ocr_result["confidence"],
            ocr_engine_used=ocr_result["engine_used"],
            llm_corrected=ocr_result["llm_corrected"],
            processing_ms=int((time.monotonic() - t0) * 1000),
        )

@router.post(
    "/extract-and-translate",
    response_model=OCRTranslateResponse,
    summary="Extract text from image and translate",
)
async def ocr_extract_and_translate(
    file: UploadFile = File(..., description="Image file"),
    target_lang: str = Form(..., description="Target language code e.g. 'eng_Latn'"),
    source_lang_hint: Optional[str] = Form(default=None, description="Optional OCR language hint"),
    tts_enabled: bool = Form(default=True),
    tts_engine: str = Form(default="auto"),
    tts_return_audio: bool = Form(default=True),
    tts_speed: int = Form(default=150),
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    **Full OCR → Translation → TTS pipeline.**

    1. Upload image → PaddleOCR text extraction
    2. Auto-detect source language from extracted text
    3. IndicTrans2 translation to `target_lang`
    4. Optional Piper speech synthesis

    All three pipeline stages start loading their models in parallel.
    """
    t0 = time.monotonic()

    async with managed_upload(file, OCR_UPLOAD_POLICY) as file_path:
        # ── Start loading all needed services in parallel ──────────────────────
        ocr_task    = asyncio.create_task(registry.get_ocr_service())
        trans_task  = asyncio.create_task(registry.get_translation_service())

        ocr_svc, trans_svc = await asyncio.gather(ocr_task, trans_task)

        # ── Step 1: OCR ───────────────────────────────────────────────────────
        ocr_result = await ocr_svc.extract(file_path, language_hint=source_lang_hint)

        _raise_for_ocr_failure(ocr_result)

        extracted_text = ocr_result["text"]
        src_lang = ocr_result["language"]

        logger.info(
            "OCR done: lang=%s conf=%.2f engine=%s | translating to %s",
            src_lang, ocr_result["confidence"], ocr_result["engine_used"], target_lang,
        )

        # ── Step 2: Translation ───────────────────────────────────────────────
        trans_result = await asyncio.to_thread(
            trans_svc.translate,
            extracted_text,
            target_lang,
            src_lang,
        )

        translated_text = trans_result["translated_text"]

        # ── Step 3: TTS (optional) ────────────────────────────────────────────
        tts_result: Optional[TTSResult] = None

        if tts_enabled:
            tts_svc = await registry.get_tts_service()
            raw_tts = await tts_svc.synthesise(
                text=translated_text,
                lang_code=target_lang,
                engine=tts_engine,
                speed=tts_speed,
                use_llm_enhancement=True,
            )
            tts_result = TTSResult(
                success=raw_tts["success"],
                engine_used=raw_tts.get("engine_used"),
                audio_base64=raw_tts.get("audio_base64") if tts_return_audio else None,
                audio_format="wav",
                sample_rate=raw_tts.get("sample_rate", 22050),
                processing_ms=raw_tts.get("processing_ms", 0),
            )

        return OCRTranslateResponse(
            success=True,
            extracted_text=extracted_text,
            detected_source_lang=src_lang,
            detected_source_lang_name=LANGUAGE_NAMES.get(src_lang, src_lang),
            target_lang=target_lang,
            target_lang_name=LANGUAGE_NAMES.get(target_lang, target_lang),
            translated_text=translated_text,
            ocr_confidence=ocr_result["confidence"],
            ocr_engine_used=ocr_result["engine_used"],
            llm_corrected=ocr_result["llm_corrected"],
            translation_engine=trans_result["engine"],
            tts=tts_result,
            processing_ms=int((time.monotonic() - t0) * 1000),
        )
