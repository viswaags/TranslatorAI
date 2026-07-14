"""
Text Pipeline
=============
POST /api/v1/text/translate
  - Accept plain text + target language
  - Auto-detect source language (parallel with translation start)
  - Translate using IndicTrans2 (M2M-100 fallback commented)
  - Optionally synthesise translated text to speech (TTS)

POST /api/v1/text/detect-language
  - Detect the language of input text
"""

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from core.registry import ServiceRegistry
from models.schemas import (
    TTSResult,
    TextTranslateRequest,
    TextTranslateResponse,
)
from utils.language_detector import LANGUAGE_NAMES

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Dependency: get registry from app.state ───────────────────────────────────

def get_registry(request: Request) -> ServiceRegistry:
    return request.app.state.registry


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/translate", response_model=TextTranslateResponse, summary="Translate text")
async def translate_text(
    body: TextTranslateRequest,
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    **Full text translation pipeline.**

    1. Auto-detect source language (skip if `source_lang` provided)
    2. Translate to `target_lang` via IndicTrans2
    3. Optionally synthesise translated text to speech

    Language detection and service loading happen in parallel where possible.
    """
    t0 = time.monotonic()

    # ── Step 1: Detect source language (parallel with service loading) ─────────
    translation_svc_task = asyncio.create_task(registry.get_translation_service())
    lang_detector_task   = asyncio.create_task(registry.get_lang_detector())

    translation_svc, lang_detector = await asyncio.gather(
        translation_svc_task, lang_detector_task
    )

    if body.source_lang:
        src_lang = body.source_lang
        logger.info("Source lang provided: %s", src_lang)
    else:
        src_lang, confidence = lang_detector.detect_with_confidence(body.text)
        logger.info("Detected source lang: %s (conf=%.2f)", src_lang, confidence)

    target_lang = body.target_lang

    # ── Step 2: Translate ──────────────────────────────────────────────────────
    try:
        trans_result = await asyncio.to_thread(
            translation_svc.translate,
            body.text,
            target_lang,
            src_lang,
        )
    except Exception as e:
        logger.error("Translation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Translation error: {e}")

    translated_text = trans_result["translated_text"]
    translation_engine = trans_result["engine"]

    # ── Step 3: TTS (optional, parallel with building response) ───────────────
    tts_result: Optional[TTSResult] = None

    if body.tts.enabled:
        tts_svc = await registry.get_tts_service()
        raw_tts = await tts_svc.synthesise(
            text=translated_text,
            lang_code=target_lang,
            engine=body.tts.engine,
            speed=body.tts.speed,
            use_llm_enhancement=True,
        )

        # Strip audio bytes if caller didn't want them
        audio_b64 = raw_tts.get("audio_base64") if body.tts.return_audio else None

        tts_result = TTSResult(
            success=raw_tts["success"],
            engine_used=raw_tts.get("engine_used"),
            audio_base64=audio_b64,
            audio_format=raw_tts.get("audio_format", "wav"),
            sample_rate=raw_tts.get("sample_rate", 22050),
            processing_ms=raw_tts.get("processing_ms", 0),
        )

    processing_ms = int((time.monotonic() - t0) * 1000)

    return TextTranslateResponse(
        success=True,
        input_text=body.text,
        detected_source_lang=src_lang,
        detected_source_lang_name=LANGUAGE_NAMES.get(src_lang, src_lang),
        target_lang=target_lang,
        target_lang_name=LANGUAGE_NAMES.get(target_lang, target_lang),
        translated_text=translated_text,
        translation_engine=translation_engine,
        tts=tts_result,
        processing_ms=processing_ms,
    )


@router.post(
    "/detect-language",
    summary="Detect the language of text",
    response_class=JSONResponse,
)
async def detect_language(
    body: dict,
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    Detect the language of input text.

    Request body: `{"text": "..."}`

    Returns detected language code and name.
    """
    text = body.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=422, detail="text field is required")

    lang_detector = await registry.get_lang_detector()
    lang_code, confidence = lang_detector.detect_with_confidence(text)

    return {
        "detected_lang": lang_code,
        "detected_lang_name": LANGUAGE_NAMES.get(lang_code, lang_code),
        "confidence": confidence,
    }
