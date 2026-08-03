"""Canonical offline OCR service using PaddleOCR."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ai_translator_api.core.config import settings
from ai_translator_api.core.errors import (
    InferenceError,
    LifecycleError,
    ModelLoadError,
    ModelUnavailableError,
    ServiceError,
    UnsupportedInputError,
    ValidationError,
)
from ai_translator_api.core.lifecycle import LazyBackendSlot, LifecycleCoordinator
from ai_translator_api.utils.languages import (
    ENGLISH_CODE,
    OCR_LANGUAGE_MAP,
    TRANSLATION_LANGUAGE_CODES,
)

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = TRANSLATION_LANGUAGE_CODES
INDICTRANS_TO_PADDLE = OCR_LANGUAGE_MAP


class OCRError(ServiceError):
    """Base OCR subsystem error."""


class OCRValidationError(ValidationError, OCRError):
    """OCR input or configuration is invalid."""


class OCRFileNotFoundError(OCRValidationError):
    """The requested image does not exist."""


class OCRUnsupportedImageError(UnsupportedInputError, OCRValidationError):
    """The input is not a supported image type."""


class OCRCorruptedImageError(OCRValidationError):
    """The input cannot be decoded as an image."""


class OCREmptyPageError(UnsupportedInputError, OCRError):
    """PaddleOCR found no text on the image."""


class OCRModelUnavailableError(ModelUnavailableError, OCRError):
    """Required local PaddleOCR artifacts are unavailable."""


class OCRModelLoadError(ModelLoadError, OCRError):
    """Local PaddleOCR artifacts could not be loaded."""


class OCRInferenceError(InferenceError, OCRError):
    """PaddleOCR failed during inference."""


class OCRLifecycleError(LifecycleError, OCRError):
    """The OCR service is undergoing a lifecycle transition."""


@dataclass
class OCRResult:
    """Canonical internal OCR result."""

    text: str
    language: str
    confidence: float
    engine_used: str = "paddle"
    corrected: bool = False
    raw_text: str = ""
    processing_time_ms: int = 0
    error: Optional[str] = None
    error_type: Optional[str] = None
    page_count: int = 1
    line_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict:
        data = asdict(self)
        return {
            "text": data["text"],
            "language": data["language"] or ENGLISH_CODE,
            "confidence": round(data["confidence"], 3),
            "engine_used": data["engine_used"],
            "llm_corrected": data["corrected"],
            "processing_ms": data["processing_time_ms"],
            "error": data["error"],
            "error_type": data["error_type"],
            "raw_text": data["raw_text"],
            "page_count": data["page_count"],
            "line_count": data["line_count"],
            "warnings": data["warnings"],
        }


class OCRBackend(ABC):
    """Backend boundary used by OCRService."""

    @abstractmethod
    def extract(self, image_path: Path, language_hint: Optional[str]) -> OCRResult:
        """Extract text from one validated image."""

    @abstractmethod
    def unload(self) -> None:
        """Release backend resources."""


class PaddleOCRBackend(OCRBackend):
    """Offline PaddleOCR backend with per-language lazy model loading."""

    def __init__(self, model_root: str | Path):
        self.model_root = Path(model_root).expanduser()
        self._readers: dict[str, object] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _inference_files_present(model_dir: Path) -> bool:
        return all(
            (model_dir / filename).is_file()
            for filename in (
                "inference.pdmodel",
                "inference.pdiparams",
                "inference.pdiparams.info",
            )
        )

    @classmethod
    def _find_model_dir(cls, parent: Path) -> Path:
        if cls._inference_files_present(parent):
            return parent
        if parent.is_dir():
            for child in sorted(parent.iterdir()):
                if child.is_dir() and cls._inference_files_present(child):
                    return child
        raise OCRModelUnavailableError(
            f"No complete local PaddleOCR model found under {parent}"
        )

    def _resolve_model_paths(self, family: str) -> tuple[Path, Path, Path]:
        rec_dir = self._find_model_dir(self.model_root / "rec" / family)
        det_family = "en" if family == "en" else "ml"
        det_dir = self._find_model_dir(self.model_root / "det" / det_family)
        cls_dir = self._find_model_dir(self.model_root / "cls")
        return det_dir, rec_dir, cls_dir

    def _get_reader(self, language_hint: Optional[str]):
        resolved_language = language_hint or ENGLISH_CODE
        if resolved_language not in INDICTRANS_TO_PADDLE:
            raise OCRModelUnavailableError(
                f"No configured local PaddleOCR model for {resolved_language!r}"
            )
        paddle_lang, family = INDICTRANS_TO_PADDLE[resolved_language]

        with self._lock:
            if resolved_language in self._readers:
                return self._readers[resolved_language]

            det_dir, rec_dir, cls_dir = self._resolve_model_paths(family)
            logger.info(
                "Loading offline PaddleOCR | language=%s | recognition=%s",
                resolved_language,
                rec_dir,
            )
            try:
                from paddleocr import PaddleOCR

                reader = PaddleOCR(
                    lang=paddle_lang,
                    det_model_dir=str(det_dir),
                    rec_model_dir=str(rec_dir),
                    cls_model_dir=str(cls_dir),
                    use_angle_cls=settings.OCR_USE_ANGLE_CLASSIFIER,
                    use_gpu=False,
                    enable_mkldnn=settings.OCR_ENABLE_MKLDNN,
                    cpu_threads=settings.OCR_CPU_THREADS,
                    show_log=False,
                )
            except Exception as exc:
                raise OCRModelLoadError(
                    f"Failed to load local PaddleOCR model for {resolved_language}"
                ) from exc

            self._readers[resolved_language] = reader
            logger.info("Offline PaddleOCR ready | language=%s", resolved_language)
            return reader

    def extract(self, image_path: Path, language_hint: Optional[str]) -> OCRResult:
        # Paddle predictors are not assumed to be safe for concurrent calls.
        with self._lock:
            reader = self._get_reader(language_hint)
            try:
                raw_results = reader.ocr(
                    str(image_path), cls=settings.OCR_USE_ANGLE_CLASSIFIER
                )
            except Exception as exc:
                raise OCRInferenceError(
                    f"PaddleOCR inference failed for {image_path.name}"
                ) from exc

        lines, confidences, page_count = self._parse_results(raw_results)
        if not lines:
            raise OCREmptyPageError(
                f"PaddleOCR found no text in {image_path.name}"
            )

        text = "\n".join(lines)
        language = self._resolve_result_language(text, language_hint)
        confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        return OCRResult(
            text=text,
            raw_text=text,
            language=language,
            confidence=confidence,
            page_count=page_count,
            line_count=len(lines),
        )

    @staticmethod
    def _parse_results(raw_results) -> tuple[list[str], list[float], int]:
        if raw_results is None:
            return [], [], 0
        if not isinstance(raw_results, list):
            raw_results = list(raw_results)

        lines: list[str] = []
        confidences: list[float] = []
        page_count = len(raw_results)

        for page in raw_results:
            if page is None:
                continue
            if hasattr(page, "rec_texts"):
                texts = page.rec_texts
                scores = getattr(page, "rec_scores", [])
                PaddleOCRBackend._append_lines(lines, confidences, texts, scores)
                continue
            if isinstance(page, dict) and "rec_texts" in page:
                PaddleOCRBackend._append_lines(
                    lines,
                    confidences,
                    page["rec_texts"],
                    page.get("rec_scores", []),
                )
                continue
            if isinstance(page, list):
                for entry in page:
                    try:
                        text_score = entry[1]
                        text = str(text_score[0]).strip()
                        score = float(text_score[1])
                    except (IndexError, TypeError, ValueError):
                        logger.debug("Ignoring unrecognized PaddleOCR entry: %r", entry)
                        continue
                    if text:
                        lines.append(text)
                        confidences.append(score)

        return lines, confidences, max(page_count, 1)

    @staticmethod
    def _append_lines(lines, confidences, texts, scores) -> None:
        scores = list(scores)
        for index, text in enumerate(texts):
            normalized = str(text).strip()
            if normalized:
                lines.append(normalized)
                score = scores[index] if index < len(scores) else 0.0
                confidences.append(float(score))

    @staticmethod
    def _resolve_result_language(
        text: str, language_hint: Optional[str]
    ) -> str:
        from ai_translator_api.utils.language_detector import LanguageDetector

        detected = LanguageDetector().detect(text)
        if detected == ENGLISH_CODE and language_hint and not any(
            "A" <= char <= "z" for char in text
        ):
            return language_hint
        return detected

    def unload(self) -> None:
        with self._lock:
            self._readers.clear()
        logger.info("PaddleOCR models unloaded")


class OCRService:
    """Async-compatible canonical OCR service."""

    def __init__(self, backend: Optional[OCRBackend] = None):
        self._backend_slot = LazyBackendSlot(
            lambda: PaddleOCRBackend(settings.OCR_MODEL_ROOT),
            lambda resource: resource.unload(),
            backend,
        )
        self._lifecycle = LifecycleCoordinator(
            lambda: OCRLifecycleError("OCR service is currently unloading")
        )
        self._validate_configuration()
        logger.info("OCRService ready (PaddleOCR, offline, lazy loading)")

    @staticmethod
    def _validate_configuration() -> None:
        if not 0.0 <= settings.OCR_CONFIDENCE_THRESHOLD <= 1.0:
            raise OCRValidationError(
                "OCR_CONFIDENCE_THRESHOLD must be between 0 and 1"
            )
        if settings.OCR_CPU_THREADS < 1 or settings.OCR_BATCH_SIZE < 1:
            raise OCRValidationError(
                "OCR_CPU_THREADS and OCR_BATCH_SIZE must be positive"
            )

    def _get_backend(self) -> OCRBackend:
        return self._backend_slot.get()

    @staticmethod
    def _validate_language_hint(language_hint: Optional[str]) -> None:
        if language_hint is not None and language_hint not in SUPPORTED_LANGUAGES:
            raise OCRValidationError(
                f"Unsupported PaddleOCR language hint: {language_hint!r}"
            )

    @staticmethod
    def _validate_image(image_path: str | Path) -> tuple[Path, int]:
        if not isinstance(image_path, (str, Path)):
            raise OCRValidationError("image_path must be a string or Path")
        path = Path(image_path).expanduser()
        if not path.exists() or not path.is_file():
            raise OCRFileNotFoundError(f"Image not found: {path}")
        if path.suffix.lower() not in set(settings.OCR_SUPPORTED_EXTENSIONS):
            raise OCRUnsupportedImageError(
                f"Unsupported image type: {path.suffix or '<none>'}"
            )

        try:
            from PIL import Image, UnidentifiedImageError

            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                page_count = getattr(image, "n_frames", 1)
                if width < 1 or height < 1 or page_count < 1:
                    raise OCRCorruptedImageError(
                        f"Image has no readable pages: {path.name}"
                    )
        except OCRCorruptedImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise OCRCorruptedImageError(
                f"Corrupted or invalid image: {path.name}"
            ) from exc

        return path.resolve(), page_count

    def _extract_sync(
        self, image_path: str | Path, language_hint: Optional[str]
    ) -> OCRResult:
        started = time.monotonic()
        try:
            self._validate_language_hint(language_hint)
            path, page_count = self._validate_image(image_path)
            with self._lifecycle.operation():
                result = self._get_backend().extract(path, language_hint)
            result.page_count = max(result.page_count, page_count)
            if result.confidence < settings.OCR_CONFIDENCE_THRESHOLD:
                result.warnings.append(
                    "PaddleOCR confidence "
                    f"{result.confidence:.2f} is below configured threshold "
                    f"{settings.OCR_CONFIDENCE_THRESHOLD:.2f}"
                )
            return result
        except OCRError as exc:
            logger.warning("OCR failed | type=%s | error=%s", type(exc).__name__, exc)
            result = OCRResult(
                text="",
                language=language_hint or ENGLISH_CODE,
                confidence=0.0,
                engine_used="paddle",
                error=str(exc),
                error_type=type(exc).__name__,
                page_count=0,
            )
            return result
        finally:
            elapsed = int((time.monotonic() - started) * 1000)
            if "result" in locals():
                result.processing_time_ms = elapsed

    async def extract(
        self,
        image_path: str | Path,
        language_hint: Optional[str] = None,
    ) -> dict:
        result = await asyncio.to_thread(
            self._extract_sync, image_path, language_hint
        )
        logger.info(
            "OCR complete | engine=%s | lang=%s | conf=%.3f | lines=%d | %dms",
            result.engine_used,
            result.language,
            result.confidence,
            result.line_count,
            result.processing_time_ms,
        )
        return result.to_public_dict()

    async def extract_batch(
        self,
        image_paths: Sequence[str | Path],
        language_hint: Optional[str] = None,
    ) -> list[dict]:
        return await asyncio.to_thread(
            self._extract_batch_sync, image_paths, language_hint
        )

    def _extract_batch_sync(
        self,
        image_paths: Sequence[str | Path],
        language_hint: Optional[str] = None,
    ) -> list[dict]:
        if not isinstance(image_paths, Sequence) or isinstance(
            image_paths, (str, bytes)
        ):
            raise OCRValidationError("image_paths must be a sequence of paths")
        results: list[dict] = []
        for offset in range(0, len(image_paths), settings.OCR_BATCH_SIZE):
            chunk = image_paths[offset : offset + settings.OCR_BATCH_SIZE]
            results.extend(
                self._extract_sync(path, language_hint).to_public_dict()
                for path in chunk
            )
        return results

    def unload(self) -> None:
        self._lifecycle.unload(self._backend_slot.unload)
        logger.info("OCRService unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._backend_slot.is_initialized
