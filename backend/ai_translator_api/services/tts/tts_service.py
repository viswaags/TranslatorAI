"""Production Text-to-Speech service backed exclusively by local Piper voices."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import threading
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ai_translator_api.core.config import settings
from ai_translator_api.core.errors import (
    InferenceError,
    LifecycleError,
    ModelUnavailableError,
    ServiceError,
    UnsupportedInputError,
    ValidationError,
)
from ai_translator_api.core.lifecycle import LazyBackendSlot, LifecycleCoordinator
from ai_translator_api.utils.languages import TTS_LANGUAGE_CODES, TTS_VOICE_ALIASES

logger = logging.getLogger(__name__)


PIPER_LANGUAGE_PREFIXES = TTS_VOICE_ALIASES
SUPPORTED_LANGUAGES = TTS_LANGUAGE_CODES
SUPPORTED_ENGINES = frozenset({"auto", "piper"})


class TTSError(ServiceError):
    """Base class for all TTS failures."""


class TTSValidationError(ValidationError, TTSError):
    """A synthesis request is invalid."""


class TTSEmptyTextError(TTSValidationError):
    """No synthesizable text was supplied."""


class TTSTextTooLongError(TTSValidationError):
    """Text exceeds the configured synthesis limit."""


class TTSUnsupportedLanguageError(
    UnsupportedInputError, TTSValidationError
):
    """The requested language is not recognized by this service."""


class TTSInvalidVoiceError(UnsupportedInputError, TTSValidationError):
    """The requested engine or configured voice selection is invalid."""


class TTSConfigurationError(TTSError):
    """TTS runtime configuration is invalid."""


class TTSModelUnavailableError(ModelUnavailableError, TTSError):
    """A required local Piper voice artifact is unavailable."""


class TTSBackendUnavailableError(ModelUnavailableError, TTSError):
    """The local Piper runtime is unavailable."""


class TTSInferenceError(InferenceError, TTSError):
    """Piper failed while synthesizing audio."""


class TTSLifecycleError(LifecycleError, TTSError):
    """The service is undergoing a lifecycle transition."""


@dataclass(frozen=True)
class VoiceArtifacts:
    """Resolved local Piper voice files."""

    language: str
    name: str
    model_path: Path
    config_path: Path


@dataclass
class BackendSynthesis:
    """Backend-neutral audio produced for one input."""

    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    voice: str


@dataclass
class SynthesisResult:
    """Canonical internal TTS result."""

    success: bool
    language: str
    voice: Optional[str] = None
    audio_bytes: Optional[bytes] = field(default=None, repr=False)
    audio_path: Optional[str] = None
    audio_format: str = "wav"
    sample_rate: int = 22050
    duration_seconds: Optional[float] = None
    processing_ms: int = 0
    engine: str = "piper"
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    error_type: Optional[str] = None

    def to_public_dict(self) -> dict:
        data = asdict(self)
        audio = data.pop("audio_bytes")
        audio_base64 = (
            base64.b64encode(audio).decode("ascii") if audio is not None else None
        )
        return {
            "success": data["success"],
            "engine_used": data["engine"] if data["success"] else None,
            "engine": data["engine"],
            "audio_base64": audio_base64,
            "audio_path": data["audio_path"],
            "audio_format": data["audio_format"],
            "sample_rate": data["sample_rate"],
            "duration_seconds": data["duration_seconds"],
            "language": data["language"],
            "voice": data["voice"],
            "processing_ms": data["processing_ms"],
            "warnings": data["warnings"],
            "error": data["error"],
            "error_type": data["error_type"],
        }


class TTSBackend(ABC):
    """Inference boundary used by :class:`TTSService`."""

    @abstractmethod
    def synthesize(
        self, text: str, language: str, speed: int
    ) -> BackendSynthesis:
        """Synthesize one validated input."""

    @abstractmethod
    def unload(self) -> None:
        """Release all loaded voice resources."""


class PiperBackend(TTSBackend):
    """CPU-only Piper backend using local voice artifacts."""

    def __init__(self) -> None:
        self._voices: dict[str, object] = {}
        self._artifacts: dict[str, VoiceArtifacts] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _configured_voice_map() -> dict[str, str]:
        raw = settings.PIPER_VOICE_MAP.strip()
        if not raw:
            return {}
        try:
            voice_map = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TTSConfigurationError(
                "PIPER_VOICE_MAP must be a JSON object of language codes "
                "to local .onnx model paths"
            ) from exc
        if not isinstance(voice_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in voice_map.items()
        ):
            raise TTSConfigurationError(
                "PIPER_VOICE_MAP must map language codes to model paths"
            )
        invalid = sorted(set(voice_map) - SUPPORTED_LANGUAGES)
        if invalid:
            raise TTSInvalidVoiceError(
                f"PIPER_VOICE_MAP contains unsupported languages: "
                f"{', '.join(invalid)}"
            )
        return voice_map

    @staticmethod
    def _validate_artifacts(language: str, model_path: Path) -> VoiceArtifacts:
        model_path = model_path.expanduser()
        config_path = Path(f"{model_path}.json")
        missing = [
            str(path) for path in (model_path, config_path) if not path.is_file()
        ]
        if missing:
            raise TTSModelUnavailableError(
                f"Local Piper voice for {language} is incomplete; missing: "
                f"{', '.join(missing)}"
            )
        return VoiceArtifacts(
            language=language,
            name=model_path.stem,
            model_path=model_path.resolve(),
            config_path=config_path.resolve(),
        )

    def _resolve_artifacts(self, language: str) -> VoiceArtifacts:
        configured = self._configured_voice_map()
        if language in configured:
            return self._validate_artifacts(
                language, Path(configured[language])
            )

        voices_dir_value = settings.PIPER_VOICES_DIR.strip()
        if not voices_dir_value:
            raise TTSModelUnavailableError(
                "No local Piper voices are configured. Set PIPER_VOICES_DIR "
                "or PIPER_VOICE_MAP; runtime downloads are disabled."
            )
        voices_dir = Path(voices_dir_value).expanduser()
        if not voices_dir.is_dir():
            raise TTSModelUnavailableError(
                f"Local Piper voices directory not found: {voices_dir}"
            )

        candidates: list[Path] = []
        for prefix in PIPER_LANGUAGE_PREFIXES[language]:
            candidates.extend(voices_dir.rglob(f"{prefix}*.onnx"))
        candidates = sorted(set(candidates))
        complete = [
            candidate
            for candidate in candidates
            if Path(f"{candidate}.json").is_file()
        ]
        if not complete:
            raise TTSModelUnavailableError(
                f"No complete local Piper voice found for {language} in "
                f"{voices_dir}; runtime downloads are disabled."
            )
        if len(complete) > 1:
            names = ", ".join(path.name for path in complete)
            raise TTSInvalidVoiceError(
                f"Multiple Piper voices found for {language}: {names}. "
                "Select one explicitly with PIPER_VOICE_MAP."
            )
        return self._validate_artifacts(language, complete[0])

    def _load_voice(self, language: str) -> tuple[object, VoiceArtifacts]:
        with self._lock:
            voice = self._voices.get(language)
            artifacts = self._artifacts.get(language)
            if voice is not None and artifacts is not None:
                return voice, artifacts

            artifacts = self._resolve_artifacts(language)
            try:
                from piper import PiperVoice
            except ImportError as exc:
                raise TTSBackendUnavailableError(
                    "Piper is not installed in the local runtime"
                ) from exc

            logger.info(
                "Loading local Piper voice | lang=%s | model=%s",
                language,
                artifacts.model_path,
            )
            try:
                voice = PiperVoice.load(
                    artifacts.model_path,
                    config_path=artifacts.config_path,
                    use_cuda=False,
                )
            except Exception as exc:
                raise TTSModelUnavailableError(
                    f"Unable to load local Piper voice: {artifacts.model_path}"
                ) from exc
            self._voices[language] = voice
            self._artifacts[language] = artifacts
            return voice, artifacts

    def synthesize(
        self, text: str, language: str, speed: int
    ) -> BackendSynthesis:
        # Piper voice objects and ONNX sessions are kept behind one lock. This
        # also prevents unload from releasing a voice during active inference.
        with self._lock:
            voice, artifacts = self._load_voice(language)
            try:
                from piper.config import SynthesisConfig

                relative_speed = settings.TTS_DEFAULT_SPEED / speed
                synthesis_config = SynthesisConfig(
                    length_scale=settings.PIPER_LENGTH_SCALE * relative_speed,
                    noise_scale=settings.PIPER_NOISE_SCALE,
                    noise_w_scale=settings.PIPER_NOISE_W_SCALE,
                    volume=settings.PIPER_VOLUME,
                )
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav_file:
                    voice.synthesize_wav(
                        text, wav_file, syn_config=synthesis_config
                    )
                audio_bytes = buffer.getvalue()
                with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    frame_count = wav_file.getnframes()
                if not audio_bytes or frame_count <= 0 or sample_rate <= 0:
                    raise TTSInferenceError("Piper produced empty audio")
            except TTSError:
                raise
            except Exception as exc:
                raise TTSInferenceError(
                    f"Piper synthesis failed for {language}"
                ) from exc

            return BackendSynthesis(
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                duration_seconds=frame_count / sample_rate,
                voice=artifacts.name,
            )

    def unload(self) -> None:
        with self._lock:
            self._voices.clear()
            self._artifacts.clear()
        logger.info("Piper backend unloaded")


class TTSService:
    """Canonical TTS service preserving the existing FastAPI contract."""

    def __init__(self, backend: Optional[TTSBackend] = None) -> None:
        self._backend_slot = LazyBackendSlot(
            PiperBackend,
            lambda resource: resource.unload(),
            backend,
        )
        self._lifecycle = LifecycleCoordinator(
            lambda: TTSLifecycleError("TTS service is currently unloading")
        )
        self._validate_settings()
        logger.info("TTSService ready (local Piper voices load lazily)")

    @staticmethod
    def _validate_settings() -> None:
        positive = {
            "TTS_MAX_TEXT_CHARS": settings.TTS_MAX_TEXT_CHARS,
            "TTS_BATCH_SIZE": settings.TTS_BATCH_SIZE,
            "TTS_MIN_SPEED": settings.TTS_MIN_SPEED,
            "TTS_DEFAULT_SPEED": settings.TTS_DEFAULT_SPEED,
            "TTS_MAX_SPEED": settings.TTS_MAX_SPEED,
            "PIPER_LENGTH_SCALE": settings.PIPER_LENGTH_SCALE,
            "PIPER_NOISE_SCALE": settings.PIPER_NOISE_SCALE,
            "PIPER_NOISE_W_SCALE": settings.PIPER_NOISE_W_SCALE,
            "PIPER_VOLUME": settings.PIPER_VOLUME,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise TTSConfigurationError(
                f"TTS settings must be positive: {', '.join(invalid)}"
            )
        if not (
            settings.TTS_MIN_SPEED
            <= settings.TTS_DEFAULT_SPEED
            <= settings.TTS_MAX_SPEED
        ):
            raise TTSConfigurationError(
                "TTS_DEFAULT_SPEED must be within the configured speed range"
            )

    def _get_backend(self) -> TTSBackend:
        return self._backend_slot.get()

    @staticmethod
    def _validate_request(
        text: str, language: str, engine: str, speed: int
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise TTSEmptyTextError("Text must not be empty or whitespace-only")
        normalized = text.strip()
        if len(normalized) > settings.TTS_MAX_TEXT_CHARS:
            raise TTSTextTooLongError(
                f"Text length {len(normalized)} exceeds the "
                f"{settings.TTS_MAX_TEXT_CHARS} character limit"
            )
        if not isinstance(language, str) or language not in SUPPORTED_LANGUAGES:
            raise TTSUnsupportedLanguageError(
                f"Unsupported TTS language code: {language!r}"
            )
        if not isinstance(engine, str) or engine.lower() not in SUPPORTED_ENGINES:
            raise TTSInvalidVoiceError(
                f"Unsupported TTS engine/voice selection: {engine!r}; "
                "only 'auto' and 'piper' are available"
            )
        if (
            isinstance(speed, bool)
            or not isinstance(speed, int)
            or not settings.TTS_MIN_SPEED <= speed <= settings.TTS_MAX_SPEED
        ):
            raise TTSValidationError(
                f"speed must be an integer from {settings.TTS_MIN_SPEED} "
                f"to {settings.TTS_MAX_SPEED}"
            )
        return normalized

    @staticmethod
    def _failure(
        error: TTSError, language: str, started: float
    ) -> SynthesisResult:
        logger.warning("TTS request failed | type=%s | %s", type(error).__name__, error)
        return SynthesisResult(
            success=False,
            language=language if isinstance(language, str) else "",
            processing_ms=int((time.monotonic() - started) * 1000),
            error=str(error),
            error_type=type(error).__name__,
        )

    def _synthesise_sync(
        self,
        text: str,
        lang_code: str,
        engine: str = "auto",
        speed: int = 150,
        use_llm_enhancement: bool = True,
    ) -> SynthesisResult:
        started = time.monotonic()
        try:
            normalized = self._validate_request(text, lang_code, engine, speed)
            with self._lifecycle.operation():
                output = self._get_backend().synthesize(
                    normalized, lang_code, speed
                )
            processing_ms = int((time.monotonic() - started) * 1000)
            result = SynthesisResult(
                success=True,
                language=lang_code,
                voice=output.voice,
                audio_bytes=output.audio_bytes,
                sample_rate=output.sample_rate,
                duration_seconds=output.duration_seconds,
                processing_ms=processing_ms,
            )
            logger.info(
                "TTS complete | engine=piper | lang=%s | voice=%s | "
                "duration=%.2fs | %dms",
                lang_code,
                output.voice,
                output.duration_seconds,
                processing_ms,
            )
            return result
        except TTSError as exc:
            return self._failure(exc, lang_code, started)
        except Exception as exc:
            logger.exception("Unexpected Piper synthesis failure")
            return self._failure(
                TTSInferenceError(f"Unexpected TTS failure: {exc}"),
                lang_code,
                started,
            )

    async def synthesise(
        self,
        text: str,
        lang_code: str,
        engine: str = "auto",
        speed: int = 150,
        use_llm_enhancement: bool = True,
    ) -> dict:
        """Synthesize speech while preserving the existing async API."""
        result = await asyncio.to_thread(
            self._synthesise_sync,
            text,
            lang_code,
            engine,
            speed,
            use_llm_enhancement,
        )
        return result.to_public_dict()

    def _synthesise_batch_sync(
        self,
        texts: Sequence[str],
        lang_code: str,
        engine: str = "auto",
        speed: int = 150,
        use_llm_enhancement: bool = True,
    ) -> list[dict]:
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise TTSValidationError("texts must be a sequence of strings")
        results: list[dict] = []
        for offset in range(0, len(texts), settings.TTS_BATCH_SIZE):
            chunk = texts[offset : offset + settings.TTS_BATCH_SIZE]
            results.extend(
                self._synthesise_sync(
                    text,
                    lang_code,
                    engine,
                    speed,
                    use_llm_enhancement,
                ).to_public_dict()
                for text in chunk
            )
        return results

    async def synthesise_batch(
        self,
        texts: Sequence[str],
        lang_code: str,
        engine: str = "auto",
        speed: int = 150,
        use_llm_enhancement: bool = True,
    ) -> list[dict]:
        """Synthesize a bounded batch with the same structured contract."""
        return await asyncio.to_thread(
            self._synthesise_batch_sync,
            texts,
            lang_code,
            engine,
            speed,
            use_llm_enhancement,
        )

    def unload(self) -> None:
        self._lifecycle.unload(self._backend_slot.unload)
        logger.info("TTSService unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._backend_slot.is_initialized
