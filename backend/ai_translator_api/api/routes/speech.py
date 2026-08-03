"""
Speech Pipeline
===============
POST /api/v1/speech/transcribe-and-translate
  - Upload audio file (WAV/MP3/OGG)
  - Whisper STT → auto-detect language → IndicTrans2 → TTS
  - Returns transcribed text + translated text + optional audio

POST /api/v1/speech/transcribe
  - STT only — returns transcribed text + detected language (no translation)
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ai_translator_api.core.config import settings
from ai_translator_api.core.registry import ServiceRegistry
from ai_translator_api.models.schemas import SpeechTranslateResponse, TTSResult
from ai_translator_api.utils.language_detector import LANGUAGE_NAMES
from ai_translator_api.utils.uploads import UploadPolicy, managed_upload

logger = logging.getLogger(__name__)
router = APIRouter()

SPEECH_UPLOAD_POLICY = UploadPolicy(
    name="audio",
    upload_dir=Path(settings.UPLOAD_DIR),
    max_size_bytes=settings.SPEECH_MAX_UPLOAD_SIZE_BYTES,
    chunk_size_bytes=settings.UPLOAD_CHUNK_SIZE_BYTES,
    allowed_extensions=frozenset(settings.STT_SUPPORTED_EXTENSIONS),
)


def get_registry(request: Request) -> ServiceRegistry:
    return request.app.state.registry


@router.post(
    "/transcribe",
    summary="Transcribe speech (STT only, no translation)",
)
async def transcribe_only(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC)"),
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    **Speech-to-Text only** — returns transcribed text and detected language.
    Uses faster-whisper with language auto-detection.
    """
    t0 = time.monotonic()
    async with managed_upload(file, SPEECH_UPLOAD_POLICY) as file_path:
        stt_svc = await registry.get_stt_service()
        stt_result = await stt_svc.transcribe(file_path)

        lang = stt_result["indictrans_lang"]
        return {
            "success": True,
            "text": stt_result["text"],
            "detected_lang": lang,
            "detected_lang_name": LANGUAGE_NAMES.get(lang, lang),
            "whisper_lang": stt_result["whisper_lang"],
            "confidence": stt_result["confidence"],
            "processing_ms": int((time.monotonic() - t0) * 1000),
        }
@router.post(
    "/transcribe-and-translate",
    response_model=SpeechTranslateResponse,
    summary="Transcribe, translate and optionally synthesise speech",
)
async def transcribe_and_translate(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC)"),
    target_lang: str = Form(..., description="Target language code e.g. 'tam_Taml'"),
    tts_enabled: bool = Form(default=True),
    tts_engine: str = Form(default="auto"),
    tts_return_audio: bool = Form(default=True),
    tts_speed: int = Form(default=150),
    registry: ServiceRegistry = Depends(get_registry),
):
    """
    **Full Speech-to-Speech Translation pipeline.**

    1. Faster Whisper — transcribes audio and detects its language
    2. IndicTrans2 — translates to `target_lang`
    3. Piper — optionally synthesises the translated text

    Faster Whisper confidence is returned with the result. Services load local
    artifacts lazily; speech recognition and translation service acquisition
    run in parallel, and Piper loads only when speech synthesis is enabled.
    """
    t0 = time.monotonic()

    async with managed_upload(file, SPEECH_UPLOAD_POLICY) as file_path:
        # ── Load STT + Translation services in parallel ────────────────────────
        stt_task   = asyncio.create_task(registry.get_stt_service())
        trans_task = asyncio.create_task(registry.get_translation_service())

        stt_svc, trans_svc = await asyncio.gather(stt_task, trans_task)

        # ── Step 1: Speech → Text ─────────────────────────────────────────────
        stt_result = await stt_svc.transcribe(file_path)

        transcribed_text = stt_result["text"]
        src_lang = stt_result["indictrans_lang"]
        whisper_lang = stt_result["whisper_lang"]
        whisper_confidence = stt_result["confidence"]

        if not transcribed_text:
            raise HTTPException(
                status_code=422, detail="No speech detected in the audio file"
            )

        logger.info(
            "STT: lang=%s conf=%.2f text_len=%d | translating to %s",
            src_lang, whisper_confidence, len(transcribed_text), target_lang,
        )

        # ── Step 2: Translate ─────────────────────────────────────────────────
        trans_result = await asyncio.to_thread(
            trans_svc.translate,
            transcribed_text,
            target_lang,
            src_lang,
        )

        translated_text = trans_result["translated_text"]

        # ── Step 3: TTS ───────────────────────────────────────────────────────
        tts_result: Optional[TTSResult] = None

        if tts_enabled:
            tts_svc = await registry.get_tts_service()

            # Low Whisper confidence → enhance text with LLM before TTS
            llm_enhance = whisper_confidence < settings.TTS_LOW_QUALITY_THRESHOLD

            raw_tts = await tts_svc.synthesise(
                text=translated_text,
                lang_code=target_lang,
                engine=tts_engine,
                speed=tts_speed,
                use_llm_enhancement=llm_enhance,
            )

            tts_result = TTSResult(
                success=raw_tts["success"],
                engine_used=raw_tts.get("engine_used"),
                audio_base64=raw_tts.get("audio_base64") if tts_return_audio else None,
                audio_format="wav",
                sample_rate=raw_tts.get("sample_rate", 22050),
                processing_ms=raw_tts.get("processing_ms", 0),
            )

        return SpeechTranslateResponse(
            success=True,
            transcribed_text=transcribed_text,
            detected_source_lang=whisper_lang,
            detected_source_lang_indictrans=src_lang,
            detected_source_lang_name=LANGUAGE_NAMES.get(src_lang, src_lang),
            whisper_confidence=whisper_confidence,
            target_lang=target_lang,
            target_lang_name=LANGUAGE_NAMES.get(target_lang, target_lang),
            translated_text=translated_text,
            translation_engine=trans_result["engine"],
            tts=tts_result,
            processing_ms=int((time.monotonic() - t0) * 1000),
        )
