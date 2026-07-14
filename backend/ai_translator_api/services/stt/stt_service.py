"""
STTService
==========
Async wrapper around WhisperSTT (faster-whisper).

Accepts audio file path → returns transcribed text + detected language.
Whisper auto-detects the spoken language — no hint required.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from core.config import settings

logger = logging.getLogger(__name__)

# Whisper lang code → IndicTrans2 code
WHISPER_TO_INDICTRANS = {
    "en": "eng_Latn",
    "ta": "tam_Taml",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "mr": "mar_Deva",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "or": "ory_Orya",
}

LANGUAGE_NAMES = {
    "eng_Latn": "English",
    "tam_Taml": "Tamil",
    "hin_Deva": "Hindi",
    "tel_Telu": "Telugu",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "ben_Beng": "Bengali",
    "guj_Gujr": "Gujarati",
    "mar_Deva": "Marathi",
    "pan_Guru": "Punjabi",
    "urd_Arab": "Urdu",
    "ory_Orya": "Odia",
}


class STTService:
    """
    Lazy-loading Whisper STT service.
    Model is loaded on the first transcription call.
    """

    def __init__(self):
        self._model = None
        self._model_size = settings.WHISPER_MODEL_SIZE
        logger.info(
            "STTService created (Whisper '%s' loads on first use)", self._model_size
        )

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info("Loading Whisper model: %s", self._model_size)
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type="int8",
            )
            logger.info("✅ Whisper ready: %s", self._model_size)
        return self._model

    async def transcribe(self, audio_path: str | Path) -> dict:
        """
        Transcribe an audio file.

        Returns:
            {
                "text": str,
                "whisper_lang": str,           # e.g. "ta"
                "indictrans_lang": str,        # e.g. "tam_Taml"
                "lang_name": str,              # e.g. "Tamil"
                "confidence": float,
                "processing_ms": int,
            }
        """
        t0 = time.monotonic()
        audio_path = str(audio_path)

        def _run():
            model = self._get_model()
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                best_of=5,
                language=None,           # auto-detect
                task="transcribe",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text, info

        text, info = await asyncio.to_thread(_run)

        whisper_lang = info.language
        indictrans_lang = WHISPER_TO_INDICTRANS.get(whisper_lang, "eng_Latn")
        lang_name = LANGUAGE_NAMES.get(indictrans_lang, whisper_lang)
        confidence = round(info.language_probability, 3)

        processing_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "STT done | lang=%s→%s | conf=%.2f | len=%d | %dms",
            whisper_lang, indictrans_lang, confidence, len(text), processing_ms,
        )

        return {
            "text":           text,
            "whisper_lang":   whisper_lang,
            "indictrans_lang": indictrans_lang,
            "lang_name":      lang_name,
            "confidence":     confidence,
            "processing_ms":  processing_ms,
        }

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
