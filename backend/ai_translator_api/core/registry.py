"""
ServiceRegistry
===============
Central lazy-loading hub for all AI models.

Design principles:
  - Models are NEVER loaded at import time
  - Each model loads on its FIRST actual use
  - Models stay in memory after loading (warm cache)
  - unload_all() releases everything (used on shutdown)
  - Thread-safe via asyncio locks (FastAPI is async)

Access pattern:
    registry = ServiceRegistry.get_instance()
    translator = await registry.get_translation_service()
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """
    Singleton registry. Call get_instance() — never instantiate directly.
    """

    _instance: Optional["ServiceRegistry"] = None

    def __init__(self):
        # ── Service slots (None = not loaded yet) ──────────────────────────
        self._translation_service = None
        self._ocr_service = None
        self._stt_service = None
        self._tts_service = None
        self._lang_detector = None

        # ── Per-service asyncio locks (prevent duplicate loads) ────────────
        self._translation_lock = asyncio.Lock()
        self._ocr_lock = asyncio.Lock()
        self._stt_lock = asyncio.Lock()
        self._tts_lock = asyncio.Lock()
        self._lang_lock = asyncio.Lock()

        logger.info("ServiceRegistry created (all services unloaded)")

    @classmethod
    def get_instance(cls) -> "ServiceRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Translation ────────────────────────────────────────────────────────────

    async def get_translation_service(self):
        """Lazy-load and return the TranslationService."""
        if self._translation_service is None:
            async with self._translation_lock:
                if self._translation_service is None:   # double-check after lock
                    logger.info("Loading TranslationService…")
                    from services.translation.translation_service import TranslationService
                    self._translation_service = TranslationService()
                    logger.info("✅ TranslationService ready")
        return self._translation_service

    # ── OCR ───────────────────────────────────────────────────────────────────

    async def get_ocr_service(self):
        """Lazy-load and return the OCRService."""
        if self._ocr_service is None:
            async with self._ocr_lock:
                if self._ocr_service is None:
                    logger.info("Loading OCRService…")
                    from services.ocr.ocr_service import OCRService
                    self._ocr_service = OCRService()
                    logger.info("✅ OCRService ready")
        return self._ocr_service

    # ── STT ───────────────────────────────────────────────────────────────────

    async def get_stt_service(self):
        """Lazy-load and return the STTService."""
        if self._stt_service is None:
            async with self._stt_lock:
                if self._stt_service is None:
                    logger.info("Loading STTService…")
                    from services.stt.stt_service import STTService
                    self._stt_service = STTService()
                    logger.info("✅ STTService ready")
        return self._stt_service

    # ── TTS ───────────────────────────────────────────────────────────────────

    async def get_tts_service(self):
        """Lazy-load and return the TTSService."""
        if self._tts_service is None:
            async with self._tts_lock:
                if self._tts_service is None:
                    logger.info("Loading TTSService…")
                    from services.tts.tts_service_final import TTSService
                    self._tts_service = TTSService()
                    logger.info("✅ TTSService ready")
        return self._tts_service

    # ── Language Detector ─────────────────────────────────────────────────────

    async def get_lang_detector(self):
        """Lazy-load and return the LanguageDetector."""
        if self._lang_detector is None:
            async with self._lang_lock:
                if self._lang_detector is None:
                    logger.info("Loading LanguageDetector…")
                    from utils.language_detector import LanguageDetector
                    self._lang_detector = LanguageDetector()
                    logger.info("✅ LanguageDetector ready")
        return self._lang_detector

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return which services are currently loaded."""
        return {
            "translation": self._translation_service is not None,
            "ocr":         self._ocr_service is not None,
            "stt":         self._stt_service is not None,
            "tts":         self._tts_service is not None,
            "lang_detect": self._lang_detector is not None,
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def unload_all(self):
        """Release all models. Called on shutdown."""
        if self._ocr_service:
            try:
                self._ocr_service.unload()
            except Exception:
                pass
        if self._translation_service:
            try:
                self._translation_service.unload()
            except Exception:
                pass
        self._translation_service = None
        self._ocr_service = None
        self._stt_service = None
        self._tts_service = None
        self._lang_detector = None
        logger.info("All services unloaded")
