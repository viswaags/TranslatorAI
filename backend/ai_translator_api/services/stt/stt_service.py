"""Canonical offline Speech-to-Text service using Faster Whisper."""

from __future__ import annotations

import asyncio
import logging
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
    LANGUAGE_NAMES,
    STT_TO_INDICTRANS,
)

logger = logging.getLogger(__name__)

WHISPER_TO_INDICTRANS = STT_TO_INDICTRANS


class STTError(ServiceError):
    """Base STT subsystem error."""


class STTValidationError(ValidationError, STTError):
    """Audio input or STT configuration is invalid."""


class STTFileNotFoundError(STTValidationError):
    """The requested audio file does not exist."""


class STTUnsupportedAudioError(UnsupportedInputError, STTValidationError):
    """The audio container or format is unsupported."""


class STTCorruptedAudioError(STTValidationError):
    """The audio file cannot be decoded."""


class STTEmptyAudioError(STTValidationError):
    """The audio contains no usable samples."""


class STTAudioTooLongError(STTValidationError):
    """Audio duration exceeds the configured limit."""


class STTInvalidSampleRateError(STTValidationError):
    """The audio sample rate is outside the accepted range."""


class STTModelUnavailableError(ModelUnavailableError, STTError):
    """Required local Faster Whisper artifacts are unavailable."""


class STTModelLoadError(ModelLoadError, STTError):
    """The local Faster Whisper model could not be loaded."""


class STTInferenceError(InferenceError, STTError):
    """Faster Whisper failed during transcription."""


class STTLifecycleError(LifecycleError, STTError):
    """The service is undergoing a lifecycle transition."""


@dataclass
class AudioMetadata:
    duration_seconds: float
    sample_rate: int
    channels: int
    format_name: str


@dataclass
class TranscriptionResult:
    """Canonical internal transcription result."""

    text: str
    whisper_lang: str
    indictrans_lang: str
    lang_name: str
    confidence: float
    duration_seconds: float
    processing_ms: int = 0
    engine: str = "faster-whisper"
    segment_count: int = 0
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    error_type: Optional[str] = None

    def to_public_dict(self) -> dict:
        data = asdict(self)
        # Existing keys remain unchanged; richer metadata is additive.
        return {
            "text": data["text"],
            "transcript": data["text"],
            "whisper_lang": data["whisper_lang"],
            "indictrans_lang": data["indictrans_lang"],
            "lang_name": data["lang_name"],
            "confidence": round(data["confidence"], 3),
            "duration_seconds": round(data["duration_seconds"], 3),
            "processing_ms": data["processing_ms"],
            "engine": data["engine"],
            "segment_count": data["segment_count"],
            "warnings": data["warnings"],
            "error": data["error"],
            "error_type": data["error_type"],
        }


class STTBackend(ABC):
    """Inference backend boundary used by STTService."""

    @abstractmethod
    def transcribe(
        self, audio_path: Path, metadata: AudioMetadata
    ) -> TranscriptionResult:
        """Transcribe one validated audio file."""

    @abstractmethod
    def unload(self) -> None:
        """Release backend resources."""


class FasterWhisperBackend(STTBackend):
    """Local-only Faster Whisper backend."""

    REQUIRED_MODEL_FILES = frozenset(
        {"model.bin", "config.json", "tokenizer.json"}
    )

    def __init__(self):
        self._model = None
        self._model_path: Optional[Path] = None
        self._lock = threading.RLock()

    @classmethod
    def _validate_model_directory(cls, model_path: Path) -> Path:
        if not model_path.is_dir():
            raise STTModelUnavailableError(
                f"Local Faster Whisper model directory not found: {model_path}"
            )
        missing = [
            name for name in cls.REQUIRED_MODEL_FILES
            if not (model_path / name).is_file()
        ]
        if missing:
            raise STTModelUnavailableError(
                f"Local Faster Whisper model is incomplete at {model_path}; "
                f"missing: {', '.join(sorted(missing))}"
            )
        return model_path.resolve()

    @classmethod
    def _resolve_local_model(cls) -> Path:
        if settings.WHISPER_MODEL_PATH:
            return cls._validate_model_directory(
                Path(settings.WHISPER_MODEL_PATH).expanduser()
            )

        cache_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--Systran--faster-whisper-{settings.WHISPER_MODEL_SIZE}"
            / "snapshots"
        )
        if cache_root.is_dir():
            snapshots = sorted(
                (path for path in cache_root.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for snapshot in snapshots:
                try:
                    return cls._validate_model_directory(snapshot)
                except STTModelUnavailableError:
                    continue

        raise STTModelUnavailableError(
            "No complete local Faster Whisper model found. Set "
            "WHISPER_MODEL_PATH or cache "
            f"Systran/faster-whisper-{settings.WHISPER_MODEL_SIZE}."
        )

    def _ensure_loaded(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model

            model_path = self._resolve_local_model()
            logger.info("Loading local Faster Whisper model: %s", model_path)
            try:
                from faster_whisper import WhisperModel

                model = WhisperModel(
                    str(model_path),
                    device=settings.WHISPER_DEVICE,
                    compute_type=settings.WHISPER_COMPUTE_TYPE,
                    cpu_threads=settings.WHISPER_CPU_THREADS,
                    num_workers=settings.WHISPER_NUM_WORKERS,
                    local_files_only=True,
                )
            except Exception as exc:
                raise STTModelLoadError(
                    f"Failed to load local Faster Whisper model at {model_path}"
                ) from exc

            self._model = model
            self._model_path = model_path
            logger.info("Faster Whisper ready: %s", model_path.name)
            return model

    def transcribe(
        self, audio_path: Path, metadata: AudioMetadata
    ) -> TranscriptionResult:
        # Model generation is serialized with load/unload.
        with self._lock:
            model = self._ensure_loaded()
            try:
                segments, info = model.transcribe(
                    str(audio_path),
                    beam_size=settings.WHISPER_BEAM_SIZE,
                    best_of=settings.WHISPER_BEST_OF,
                    language=None,
                    task="transcribe",
                    vad_filter=settings.WHISPER_VAD_FILTER,
                    vad_parameters={
                        "min_silence_duration_ms": settings.WHISPER_MIN_SILENCE_MS
                    },
                )
                segment_list = list(segments)
            except Exception as exc:
                raise STTInferenceError(
                    f"Faster Whisper inference failed for {audio_path.name}"
                ) from exc

        text = " ".join(
            segment.text.strip() for segment in segment_list if segment.text.strip()
        ).strip()
        whisper_lang = getattr(info, "language", None) or "en"
        indictrans_lang = WHISPER_TO_INDICTRANS.get(
            whisper_lang, ENGLISH_CODE
        )
        confidence = float(getattr(info, "language_probability", 0.0) or 0.0)
        warnings = []
        if not text:
            warnings.append("No speech detected in the audio file")
        if whisper_lang not in WHISPER_TO_INDICTRANS:
            warnings.append(
                f"Unsupported detected language {whisper_lang!r}; mapped to English"
            )

        return TranscriptionResult(
            text=text,
            whisper_lang=whisper_lang,
            indictrans_lang=indictrans_lang,
            lang_name=LANGUAGE_NAMES.get(indictrans_lang, whisper_lang),
            confidence=confidence,
            duration_seconds=metadata.duration_seconds,
            segment_count=len(segment_list),
            warnings=warnings,
        )

    def unload(self) -> None:
        with self._lock:
            model = self._model
            self._model = None
            self._model_path = None
            if model is not None:
                del model
                logger.info("Faster Whisper model unloaded")


class STTService:
    """Async-compatible canonical STT service."""

    def __init__(self, backend: Optional[STTBackend] = None):
        self._backend_slot = LazyBackendSlot(
            FasterWhisperBackend,
            lambda resource: resource.unload(),
            backend,
        )
        self._lifecycle = LifecycleCoordinator(
            lambda: STTLifecycleError("STT service is currently unloading")
        )
        self._validate_configuration()
        logger.info("STTService ready (Faster Whisper, offline, lazy loading)")

    @staticmethod
    def _validate_configuration() -> None:
        positive = {
            "WHISPER_CPU_THREADS": settings.WHISPER_CPU_THREADS,
            "WHISPER_NUM_WORKERS": settings.WHISPER_NUM_WORKERS,
            "WHISPER_BEAM_SIZE": settings.WHISPER_BEAM_SIZE,
            "WHISPER_BEST_OF": settings.WHISPER_BEST_OF,
            "WHISPER_MIN_SILENCE_MS": settings.WHISPER_MIN_SILENCE_MS,
            "STT_MAX_DURATION_SECONDS": settings.STT_MAX_DURATION_SECONDS,
            "STT_MIN_SAMPLE_RATE": settings.STT_MIN_SAMPLE_RATE,
            "STT_MAX_SAMPLE_RATE": settings.STT_MAX_SAMPLE_RATE,
            "STT_BATCH_SIZE": settings.STT_BATCH_SIZE,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise STTValidationError(
                f"STT settings must be positive: {', '.join(invalid)}"
            )
        if settings.STT_MIN_SAMPLE_RATE > settings.STT_MAX_SAMPLE_RATE:
            raise STTValidationError(
                "STT_MIN_SAMPLE_RATE cannot exceed STT_MAX_SAMPLE_RATE"
            )

    def _get_backend(self) -> STTBackend:
        return self._backend_slot.get()

    @staticmethod
    def _validate_audio(audio_path: str | Path) -> tuple[Path, AudioMetadata]:
        if not isinstance(audio_path, (str, Path)):
            raise STTValidationError("audio_path must be a string or Path")
        path = Path(audio_path).expanduser()
        if not path.exists() or not path.is_file():
            raise STTFileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() not in set(settings.STT_SUPPORTED_EXTENSIONS):
            raise STTUnsupportedAudioError(
                f"Unsupported audio type: {path.suffix or '<none>'}"
            )
        if path.stat().st_size == 0:
            raise STTEmptyAudioError(f"Audio file is empty: {path.name}")

        try:
            import av

            with av.open(str(path), mode="r") as container:
                streams = [stream for stream in container.streams if stream.type == "audio"]
                if not streams:
                    raise STTCorruptedAudioError(
                        f"No audio stream found in {path.name}"
                    )
                stream = streams[0]
                sample_rate = int(
                    getattr(stream.codec_context, "sample_rate", 0)
                    or getattr(stream, "rate", 0)
                    or 0
                )
                channels = int(getattr(stream.codec_context, "channels", 0) or 0)
                if stream.duration is not None and stream.time_base is not None:
                    duration = float(stream.duration * stream.time_base)
                elif container.duration is not None:
                    duration = float(container.duration / av.time_base)
                else:
                    duration = 0.0
                    for frame in container.decode(stream):
                        if frame.sample_rate:
                            duration += frame.samples / frame.sample_rate
        except STTError:
            raise
        except Exception as exc:
            raise STTCorruptedAudioError(
                f"Corrupted or invalid audio: {path.name}"
            ) from exc

        if sample_rate <= 0 or channels <= 0 or duration <= 0:
            raise STTEmptyAudioError(
                f"Audio contains no usable samples: {path.name}"
            )
        if not settings.STT_MIN_SAMPLE_RATE <= sample_rate <= settings.STT_MAX_SAMPLE_RATE:
            raise STTInvalidSampleRateError(
                f"Unsupported sample rate {sample_rate}Hz for {path.name}"
            )
        if duration > settings.STT_MAX_DURATION_SECONDS:
            raise STTAudioTooLongError(
                f"Audio duration {duration:.1f}s exceeds the "
                f"{settings.STT_MAX_DURATION_SECONDS:.1f}s limit"
            )

        return path.resolve(), AudioMetadata(
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            format_name=path.suffix.lower().lstrip("."),
        )

    def _transcribe_sync(self, audio_path: str | Path) -> TranscriptionResult:
        started = time.monotonic()
        path, metadata = self._validate_audio(audio_path)
        with self._lifecycle.operation():
            result = self._get_backend().transcribe(path, metadata)
        result.processing_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "STT complete | engine=%s | lang=%s→%s | conf=%.3f | "
            "duration=%.2fs | segments=%d | %dms",
            result.engine,
            result.whisper_lang,
            result.indictrans_lang,
            result.confidence,
            result.duration_seconds,
            result.segment_count,
            result.processing_ms,
        )
        return result

    async def transcribe(self, audio_path: str | Path) -> dict:
        result = await asyncio.to_thread(self._transcribe_sync, audio_path)
        return result.to_public_dict()

    def _transcribe_batch_sync(
        self, audio_paths: Sequence[str | Path]
    ) -> list[dict]:
        if not isinstance(audio_paths, Sequence) or isinstance(
            audio_paths, (str, bytes)
        ):
            raise STTValidationError("audio_paths must be a sequence of paths")
        results: list[dict] = []
        for offset in range(0, len(audio_paths), settings.STT_BATCH_SIZE):
            chunk = audio_paths[offset : offset + settings.STT_BATCH_SIZE]
            results.extend(
                self._transcribe_sync(path).to_public_dict() for path in chunk
            )
        return results

    async def transcribe_batch(
        self, audio_paths: Sequence[str | Path]
    ) -> list[dict]:
        return await asyncio.to_thread(self._transcribe_batch_sync, audio_paths)

    def unload(self) -> None:
        self._lifecycle.unload(self._backend_slot.unload)
        logger.info("STTService unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._backend_slot.is_initialized
