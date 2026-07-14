"""
OCRService
==========
Thin async-friendly wrapper around the existing OCREngine.

Pipeline:
  Image → PaddleOCR (Layer 1)
       → Tesseract fallback (Layer 2)
       → Qwen2.5 LLM correction (Layer 3)
       → Returns OCRResult with text + detected language
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from core.config import settings

logger = logging.getLogger(__name__)


class OCRService:
    """
    Async wrapper around OCREngine (which is itself sync / CPU-bound).
    Heavy work is off-loaded to a thread pool via asyncio.to_thread().
    """

    def __init__(self):
        self._engine = None      # lazy — loaded on first extract() call
        self._engine_ready = False
        logger.info("OCRService created (OCREngine loads on first use)")

    def _get_engine(self):
        if self._engine is None:
            # Try multiple import paths to work regardless of where uvicorn is launched from
            OCREngine = None
            import_errors = []
            for module_path in [
                "services.ocr.ocr_engine",          # running from backend/
                "backend.services.ocr.ocr_engine",   # running from project root
                "ocr_engine",                         # running from services/ocr/
            ]:
                try:
                    import importlib
                    mod = importlib.import_module(module_path)
                    OCREngine = mod.OCREngine
                    break
                except ImportError as e:
                    import_errors.append(f"{module_path}: {e}")

            if OCREngine is None:
                raise ImportError(
                    "Cannot find OCREngine. Tried:\n" + "\n".join(import_errors)
                )

            self._engine = OCREngine(
                confidence_threshold=settings.OCR_CONFIDENCE_THRESHOLD,
                use_llm_correction=settings.OCR_LLM_CORRECTION,
                ollama_url=settings.OLLAMA_URL,
                ollama_model=settings.OLLAMA_MODEL,
            )
            self._engine_ready = True
            logger.info("OCREngine loaded")
        return self._engine

    async def extract(
        self,
        image_path: str | Path,
        language_hint: Optional[str] = None,
    ) -> dict:
        """
        Run OCR on an image file.

        Returns:
            {
                "text": str,
                "language": str,         # IndicTrans2 format
                "confidence": float,
                "engine_used": str,
                "llm_corrected": bool,
                "processing_ms": int,
                "error": str | None,
            }
        """
        t0 = time.monotonic()

        def _run():
            engine = self._get_engine()
            return engine.extract(str(image_path), language_hint=language_hint)

        # Off-load the CPU-bound OCR to a thread
        result = await asyncio.to_thread(_run)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        return {
            "text":          result.text,
            "language":      result.language or "eng_Latn",
            "confidence":    round(result.confidence, 3),
            "engine_used":   result.engine_used,
            "llm_corrected": result.corrected,
            "processing_ms": elapsed_ms,
            "error":         result.error,
        }

    def unload(self):
        if self._engine:
            self._engine.unload()
            self._engine = None
        logger.info("OCRService unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._engine_ready
