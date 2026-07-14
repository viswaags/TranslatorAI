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
import os
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from core.config import settings
from core.registry import ServiceRegistry
from models.schemas import (
    OCRExtractResponse,
    OCRTranslateResponse,
    TTSResult,
)
from utils.language_detector import LANGUAGE_NAMES

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
MAX_FILE_SIZE_MB = 10


def get_registry(request: Request) -> ServiceRegistry:
    return request.app.state.registry


async def _save_upload(file: UploadFile) -> str:
    """Save uploaded file to temp directory, return path."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG, BMP, TIFF, WebP",
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE_MB}MB)")

    ext = file.filename.rsplit(".", 1)[-1] if "." in (file.filename or "") else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(content)

    return file_path


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

    Extracts text from an uploaded image using:
    - Layer 1: PaddleOCR (primary, confidence-aware)
    - Layer 2: Tesseract (fallback if confidence < threshold)
    - Layer 3: Qwen2.5 LLM correction (fixes garbled characters)

    Returns extracted text + detected language in IndicTrans2 format.
    """
    t0 = time.monotonic()

    file_path = await _save_upload(file)

    try:
        # Load OCR service (lazy)
        ocr_svc = await registry.get_ocr_service()

        ocr_result = await ocr_svc.extract(file_path, language_hint=source_lang_hint)

        if ocr_result.get("error") and not ocr_result.get("text"):
            raise HTTPException(status_code=422, detail=f"OCR failed: {ocr_result['error']}")

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

    finally:
        # Clean up temp file
        try:
            os.remove(file_path)
        except OSError:
            pass


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

    1. Upload image → OCR (PaddleOCR → Tesseract → LLM correction)
    2. Auto-detect source language from extracted text
    3. Translate to `target_lang`
    4. Optionally synthesise translated text to speech

    All three pipeline stages start loading their models in parallel.
    """
    t0 = time.monotonic()

    file_path = await _save_upload(file)

    try:
        # ── Start loading all needed services in parallel ──────────────────────
        ocr_task    = asyncio.create_task(registry.get_ocr_service())
        trans_task  = asyncio.create_task(registry.get_translation_service())

        ocr_svc, trans_svc = await asyncio.gather(ocr_task, trans_task)

        # ── Step 1: OCR ───────────────────────────────────────────────────────
        ocr_result = await ocr_svc.extract(file_path, language_hint=source_lang_hint)

        if ocr_result.get("error") and not ocr_result.get("text"):
            raise HTTPException(status_code=422, detail=f"OCR failed: {ocr_result['error']}")

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

    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass
