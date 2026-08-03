"""Application service registry for canonical shared service instances."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _create_translation_service():
    from ai_translator_api.services.translation.translation_service import TranslationService

    return TranslationService()


def _create_ocr_service():
    from ai_translator_api.services.ocr.ocr_service import OCRService

    return OCRService()


def _create_stt_service():
    from ai_translator_api.services.stt.stt_service import STTService

    return STTService()


def _create_tts_service():
    from ai_translator_api.services.tts.tts_service import TTSService

    return TTSService()


def _create_language_detector():
    from ai_translator_api.utils.language_detector import LanguageDetector

    return LanguageDetector()


class ServiceRegistry:
    """Construct and expose one lazy shared instance of each application service."""

    _instance: Optional["ServiceRegistry"] = None
    _instance_lock = threading.Lock()

    _SERVICE_ORDER = (
        "translation",
        "ocr",
        "stt",
        "tts",
        "lang_detector",
    )
    _SHUTDOWN_ORDER = ("ocr", "translation", "stt", "tts", "lang_detector")
    _STATUS_NAMES = {
        "translation": "translation",
        "ocr": "ocr",
        "stt": "stt",
        "tts": "tts",
        "lang_detector": "lang_detect",
    }
    _DEFAULT_FACTORIES: dict[str, Callable[[], object]] = {
        "translation": _create_translation_service,
        "ocr": _create_ocr_service,
        "stt": _create_stt_service,
        "tts": _create_tts_service,
        "lang_detector": _create_language_detector,
    }

    def __init__(self):
        self._factories = self._DEFAULT_FACTORIES.copy()
        self._services: dict[str, Optional[object]] = {
            name: None for name in self._SERVICE_ORDER
        }
        self._creation_locks = {
            name: asyncio.Lock() for name in self._SERVICE_ORDER
        }
        logger.info("ServiceRegistry created (all services unloaded)")

    @classmethod
    def get_instance(cls) -> "ServiceRegistry":
        """Return the process-wide registry singleton."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def _get_service(self, name: str):
        service = self._services[name]
        if service is not None:
            return service

        async with self._creation_locks[name]:
            service = self._services[name]
            if service is None:
                logger.info("Creating shared service | service=%s", name)
                service = self._factories[name]()
                self._services[name] = service
                logger.info("Shared service ready | service=%s", name)
            return service

    async def get_translation_service(self):
        """Return the shared, lazily constructed TranslationService."""
        return await self._get_service("translation")

    async def get_ocr_service(self):
        """Return the shared, lazily constructed OCRService."""
        return await self._get_service("ocr")

    async def get_stt_service(self):
        """Return the shared, lazily constructed STTService."""
        return await self._get_service("stt")

    async def get_tts_service(self):
        """Return the shared, lazily constructed TTSService."""
        return await self._get_service("tts")

    async def get_lang_detector(self):
        """Return the shared, lazily constructed LanguageDetector."""
        return await self._get_service("lang_detector")

    def status(self) -> dict:
        """Return the existing public loaded-state mapping."""
        return {
            public_name: self._services[service_name] is not None
            for service_name, public_name in self._STATUS_NAMES.items()
        }

    def unload_all(self) -> None:
        """Delegate resource release to constructed services during shutdown."""
        for name in self._SHUTDOWN_ORDER:
            service = self._services[name]
            unload = getattr(service, "unload", None)
            if callable(unload):
                try:
                    unload()
                except Exception:
                    logger.exception(
                        "Service unload failed during shutdown | service=%s",
                        name,
                    )
            self._services[name] = None
        logger.info("All shared services released")
