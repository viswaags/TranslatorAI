"""Canonical translation service using the CTranslate2 IndicTrans2 backend."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
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
from ai_translator_api.utils.languages import ENGLISH_CODE, TRANSLATION_LANGUAGE_CODES

logger = logging.getLogger(__name__)

EN = ENGLISH_CODE
SUPPORTED_LANGUAGES = TRANSLATION_LANGUAGE_CODES


class TranslationError(ServiceError):
    """Base translation subsystem error."""


class TranslationValidationError(ValidationError, TranslationError):
    """Translation request or configuration is invalid."""


class UnsupportedLanguageError(
    UnsupportedInputError, TranslationValidationError
):
    """A language code or pair is unsupported."""


class TranslationModelUnavailableError(
    ModelUnavailableError, TranslationError
):
    """Required model artifacts are unavailable locally."""


class TranslationModelLoadError(ModelLoadError, TranslationError):
    """Locally available model artifacts could not be loaded."""


class TranslationInferenceError(InferenceError, TranslationError):
    """The backend failed during translation."""


class TranslationLifecycleError(LifecycleError, TranslationError):
    """The service cannot accept work during a lifecycle transition."""


class TranslationBackend(ABC):
    """Inference backend boundary used by TranslationService."""

    @abstractmethod
    def translate(
        self, sentences: Sequence[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        """Translate a homogeneous batch while preserving input order."""

    @abstractmethod
    def unload(self) -> None:
        """Release backend resources."""


class CTranslate2IndicTransBackend(TranslationBackend):
    """CTranslate2/SentencePiece implementation for one IndicTrans2 direction."""

    MODEL_MAP = {
        "en-indic": settings.INDICTRANS2_EN_INDIC,
        "indic-en": settings.INDICTRANS2_INDIC_EN,
    }

    def __init__(self, direction: str):
        if direction not in self.MODEL_MAP:
            raise TranslationValidationError(
                f"Unknown translation direction: {direction!r}"
            )
        self.direction = direction
        self.model_path = self.MODEL_MAP[direction]
        self._lock = threading.RLock()
        self._translator = None
        self._source_sp = None
        self._target_sp = None
        self._processor = None

    @staticmethod
    def _sentencepiece_path(model_dir, suffix: str):
        from pathlib import Path

        nested = Path(model_dir) / "vocab" / f"model.{suffix}"
        return nested if nested.is_file() else Path(model_dir) / f"model.{suffix}"

    @classmethod
    def _validate_model_directory(cls, model_reference: str):
        from pathlib import Path

        model_dir = Path(model_reference).expanduser()
        required = {
            "CTranslate2 model": model_dir / "model.bin",
            "source SentencePiece model": cls._sentencepiece_path(model_dir, "SRC"),
            "target SentencePiece model": cls._sentencepiece_path(model_dir, "TGT"),
        }
        missing = [label for label, path in required.items() if not path.is_file()]
        if not model_dir.is_dir() or missing:
            detail = ", ".join(missing) if missing else "model directory"
            raise TranslationModelUnavailableError(
                f"Local IndicTrans2 [{model_reference}] is missing: {detail}"
            )
        return model_dir.resolve(), required

    def _ensure_loaded(self) -> None:
        if self._translator is not None:
            return
        with self._lock:
            if self._translator is not None:
                return

            model_dir, required = self._validate_model_directory(self.model_path)
            logger.info(
                "Loading local CTranslate2 IndicTrans2 [%s]: %s",
                self.direction,
                model_dir,
            )
            try:
                import ctranslate2
                import sentencepiece as spm
                from IndicTransToolkit import IndicProcessor

                processor = IndicProcessor(inference=True)
                translator = ctranslate2.Translator(
                    str(model_dir),
                    device="cpu",
                    compute_type=settings.TRANSLATION_COMPUTE_TYPE,
                    inter_threads=settings.TRANSLATION_INTER_THREADS,
                    intra_threads=settings.TRANSLATION_INTRA_THREADS,
                )
                source_sp = spm.SentencePieceProcessor(
                    model_file=str(required["source SentencePiece model"])
                )
                target_sp = spm.SentencePieceProcessor(
                    model_file=str(required["target SentencePiece model"])
                )
            except Exception as exc:
                raise TranslationModelLoadError(
                    f"Failed to load local CTranslate2 IndicTrans2 "
                    f"[{self.direction}] model"
                ) from exc

            self._processor = processor
            self._source_sp = source_sp
            self._target_sp = target_sp
            self._translator = translator
            logger.info("CTranslate2 IndicTrans2 [%s] ready on CPU", self.direction)

    def _encode_batch(self, sentences: Sequence[str]) -> list[list[str]]:
        encoded = []
        for sentence in sentences:
            parts = sentence.split(" ", 2)
            if (
                len(parts) == 3
                and len(parts[0]) == 8
                and len(parts[1]) == 8
                and "_" in parts[0]
                and "_" in parts[1]
            ):
                tokens = [
                    parts[0],
                    parts[1],
                    *self._source_sp.encode(parts[2], out_type=str),
                ]
            else:
                tokens = self._source_sp.encode(sentence, out_type=str)

            if not tokens:
                raise TranslationInferenceError(
                    "SentencePiece returned an empty token sequence"
                )
            encoded.append(tokens[: settings.TRANSLATION_MAX_INPUT_TOKENS])
        return encoded

    def translate(
        self, sentences: Sequence[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        if not sentences:
            return []

        # The same lock serializes first-load, inference and unload for this model.
        with self._lock:
            self._ensure_loaded()
            try:
                batch = self._processor.preprocess_batch(
                    list(sentences),
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    visualize=False,
                )
                token_batches = self._encode_batch(batch)
                translations = self._translator.translate_batch(
                    token_batches,
                    beam_size=settings.TRANSLATION_NUM_BEAMS,
                    max_decoding_length=settings.TRANSLATION_MAX_OUTPUT_TOKENS,
                    no_repeat_ngram_size=settings.TRANSLATION_NO_REPEAT_NGRAM_SIZE,
                )
                decoded = [
                    self._target_sp.decode(result.hypotheses[0])
                    for result in translations
                ]
                results = self._processor.postprocess_batch(decoded, lang=tgt_lang)
            except Exception as exc:
                if isinstance(exc, TranslationError):
                    raise
                raise TranslationInferenceError(
                    f"CTranslate2 IndicTrans2 [{self.direction}] inference failed"
                ) from exc

            if len(results) != len(sentences):
                raise TranslationInferenceError(
                    "Translation backend output count does not match input count"
                )
            if any(not isinstance(item, str) or not item.strip() for item in results):
                raise TranslationInferenceError(
                    "Translation backend returned an empty or invalid output"
                )
            return list(results)

    def translate_one(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Backward-compatible single-string wrapper."""
        return self.translate([text], src_lang, tgt_lang)[0]

    def unload(self) -> None:
        with self._lock:
            translator = self._translator
            self._translator = None
            self._source_sp = None
            self._target_sp = None
            self._processor = None
            if translator is not None:
                del translator
                logger.info("Unloaded CTranslate2 IndicTrans2 [%s]", self.direction)


# Preserve the previously public wrapper name.
IndicTranslator = CTranslate2IndicTransBackend


class TranslationService:
    """Validate requests and route them through interchangeable backends."""

    def __init__(self):
        self._en_indic_slot = LazyBackendSlot(
            lambda: CTranslate2IndicTransBackend("en-indic"),
            lambda resource: resource.unload(),
        )
        self._indic_en_slot = LazyBackendSlot(
            lambda: CTranslate2IndicTransBackend("indic-en"),
            lambda resource: resource.unload(),
        )
        self._lifecycle = LifecycleCoordinator(
            lambda: TranslationLifecycleError(
                "Translation service is currently unloading"
            )
        )
        self._validate_configuration()
        logger.info("TranslationService ready (CTranslate2, offline, lazy loading)")

    @staticmethod
    def _validate_configuration() -> None:
        values = {
            "TRANSLATION_MAX_INPUT_TOKENS": settings.TRANSLATION_MAX_INPUT_TOKENS,
            "TRANSLATION_MAX_OUTPUT_TOKENS": settings.TRANSLATION_MAX_OUTPUT_TOKENS,
            "TRANSLATION_NUM_BEAMS": settings.TRANSLATION_NUM_BEAMS,
            "TRANSLATION_BATCH_SIZE": settings.TRANSLATION_BATCH_SIZE,
            "TRANSLATION_INTER_THREADS": settings.TRANSLATION_INTER_THREADS,
            "TRANSLATION_INTRA_THREADS": settings.TRANSLATION_INTRA_THREADS,
        }
        invalid = [name for name, value in values.items() if value < 1]
        if invalid:
            raise TranslationValidationError(
                f"Translation settings must be positive: {', '.join(invalid)}"
            )

    def _get_en_indic(self) -> TranslationBackend:
        return self._en_indic_slot.get()

    def _get_indic_en(self) -> TranslationBackend:
        return self._indic_en_slot.get()

    @staticmethod
    def _validate_language(code: str, role: str) -> None:
        if not isinstance(code, str) or code not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Unsupported {role} language code: {code!r}"
            )

    @classmethod
    def _validate_request(cls, text: str, src_lang: str, tgt_lang: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise TranslationValidationError(
                "Translation text must be a non-empty string"
            )
        cls._validate_language(src_lang, "source")
        cls._validate_language(tgt_lang, "target")

    @staticmethod
    def _detect_language(text: str) -> str:
        from ai_translator_api.utils.language_detector import LanguageDetector

        return LanguageDetector().detect(text)

    def translate(
        self,
        text: str,
        tgt_lang: str,
        src_lang: Optional[str] = None,
    ) -> dict:
        """Translate one string and return the canonical result structure."""
        started = time.monotonic()
        resolved_src = src_lang or self._detect_language(text)
        self._validate_request(text, resolved_src, tgt_lang)

        if resolved_src == tgt_lang:
            return self._result(
                text, resolved_src, tgt_lang, "passthrough", "passthrough", started
            )

        with self._lifecycle.operation():
            output, route = self._translate_routed([text], resolved_src, tgt_lang)
        return self._result(
            output[0], resolved_src, tgt_lang, "indictrans2", route, started
        )

    def translate_batch(
        self,
        texts: list,
        tgt_lang: str,
        src_lang: Optional[str] = None,
    ) -> list[dict]:
        """Translate a homogeneous batch and return one result dict per item."""
        if not isinstance(texts, list):
            raise TranslationValidationError("texts must be a list")
        if not texts:
            return []

        resolved_src = src_lang or self._detect_language(texts[0])
        for text in texts:
            self._validate_request(text, resolved_src, tgt_lang)

        started = time.monotonic()
        if resolved_src == tgt_lang:
            return [
                self._result(
                    text,
                    resolved_src,
                    tgt_lang,
                    "passthrough",
                    "passthrough",
                    started,
                )
                for text in texts
            ]

        translated: list[str] = []
        route = ""
        with self._lifecycle.operation():
            for offset in range(0, len(texts), settings.TRANSLATION_BATCH_SIZE):
                chunk = texts[offset : offset + settings.TRANSLATION_BATCH_SIZE]
                chunk_output, route = self._translate_routed(
                    chunk, resolved_src, tgt_lang
                )
                translated.extend(chunk_output)

        return [
            self._result(
                output, resolved_src, tgt_lang, "indictrans2", route, started
            )
            for output in translated
        ]

    def _translate_routed(
        self, texts: Sequence[str], src_lang: str, tgt_lang: str
    ) -> tuple[list[str], str]:
        if src_lang == EN:
            return self._get_en_indic().translate(texts, src_lang, tgt_lang), "direct"
        if tgt_lang == EN:
            return self._get_indic_en().translate(texts, src_lang, tgt_lang), "direct"

        pivoted = self._get_indic_en().translate(texts, src_lang, EN)
        return self._get_en_indic().translate(pivoted, EN, tgt_lang), "pivot"

    @staticmethod
    def _result(
        text: str,
        src: str,
        tgt: str,
        engine: str,
        route: str,
        started: float,
    ) -> dict:
        return {
            "translated_text": text,
            "src_lang": src,
            "tgt_lang": tgt,
            "engine": engine,
            "route": route,
            "processing_ms": int((time.monotonic() - started) * 1000),
        }

    def unload(self) -> None:
        """Wait for active calls before releasing backend resources."""
        self._lifecycle.unload(self._unload_backends)
        logger.info("TranslationService unloaded")

    def _unload_backends(self) -> None:
        first_error: Optional[Exception] = None
        for slot in (self._en_indic_slot, self._indic_en_slot):
            try:
                slot.unload()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
