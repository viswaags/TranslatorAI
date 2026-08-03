"""Read-only deployment diagnostics for local runtime artifacts."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ai_translator_api.core.config import settings
from ai_translator_api.utils.languages import OCR_LANGUAGE_MAP, TTS_VOICE_ALIASES


@dataclass(frozen=True)
class ArtifactStatus:
    available: bool
    location: str
    info: str

    def to_dict(self) -> dict:
        return asdict(self)


def _package_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _translation_status() -> ArtifactStatus:
    missing: list[str] = []
    locations: list[str] = []
    for label, configured in (
        ("en-indic", settings.INDICTRANS2_EN_INDIC),
        ("indic-en", settings.INDICTRANS2_INDIC_EN),
    ):
        root = Path(configured).expanduser()
        locations.append(str(root))
        source_sp = root / "vocab" / "model.SRC"
        target_sp = root / "vocab" / "model.TGT"
        if not source_sp.is_file():
            source_sp = root / "model.SRC"
        if not target_sp.is_file():
            target_sp = root / "model.TGT"
        for name, path in (
            ("model.bin", root / "model.bin"),
            ("model.SRC", source_sp),
            ("model.TGT", target_sp),
        ):
            if not path.is_file():
                missing.append(f"{label}/{name}")
    if not _package_available("ctranslate2"):
        missing.append("Python package ctranslate2")
    if not _package_available("sentencepiece"):
        missing.append("Python package sentencepiece")
    if not _package_available("IndicTransToolkit"):
        missing.append("Python package indictranstoolkit")
    return ArtifactStatus(
        available=not missing,
        location=", ".join(locations),
        info="ready" if not missing else f"missing: {', '.join(missing)}",
    )


def _whisper_status() -> ArtifactStatus:
    if settings.WHISPER_MODEL_PATH:
        candidates = [Path(settings.WHISPER_MODEL_PATH).expanduser()]
    else:
        snapshots = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / f"models--Systran--faster-whisper-{settings.WHISPER_MODEL_SIZE}"
            / "snapshots"
        )
        candidates = sorted(
            (path for path in snapshots.glob("*") if path.is_dir()),
            reverse=True,
        )
    required = ("model.bin", "config.json", "tokenizer.json")
    complete = next(
        (
            candidate
            for candidate in candidates
            if all((candidate / filename).is_file() for filename in required)
        ),
        None,
    )
    package_ready = _package_available("faster_whisper")
    available = complete is not None and package_ready
    location = str(complete or (candidates[0] if candidates else "not configured"))
    info = "ready" if available else "complete local model and faster-whisper are required"
    return ArtifactStatus(available, location, info)


def _paddle_model_complete(parent: Path) -> bool:
    required = ("inference.pdmodel", "inference.pdiparams", "inference.pdiparams.info")
    directories = [parent]
    if parent.is_dir():
        directories.extend(path for path in parent.iterdir() if path.is_dir())
    return any(
        all((directory / filename).is_file() for filename in required)
        for directory in directories
    )


def _ocr_status() -> ArtifactStatus:
    root = Path(settings.OCR_MODEL_ROOT).expanduser()
    families = sorted({family for _, family in OCR_LANGUAGE_MAP.values()})
    missing = [
        f"rec/{family}"
        for family in families
        if not _paddle_model_complete(root / "rec" / family)
    ]
    for family in ("en", "ml"):
        if not _paddle_model_complete(root / "det" / family):
            missing.append(f"det/{family}")
    if not _paddle_model_complete(root / "cls"):
        missing.append("cls")
    if not _package_available("paddle") or not _package_available("paddleocr"):
        missing.append("Python packages paddlepaddle/paddleocr")
    return ArtifactStatus(
        available=not missing,
        location=str(root),
        info="ready" if not missing else f"missing: {', '.join(missing)}",
    )


def _configured_piper_models() -> dict[str, Path]:
    configured: dict[str, Path] = {}
    if settings.PIPER_VOICE_MAP.strip():
        try:
            raw = json.loads(settings.PIPER_VOICE_MAP)
            if isinstance(raw, dict):
                configured.update(
                    {
                        language: Path(path).expanduser()
                        for language, path in raw.items()
                        if isinstance(language, str) and isinstance(path, str)
                    }
                )
        except json.JSONDecodeError:
            return {}
    root_value = settings.PIPER_VOICES_DIR.strip()
    if root_value:
        root = Path(root_value).expanduser()
        for language, prefixes in TTS_VOICE_ALIASES.items():
            matches = sorted(
                path
                for prefix in prefixes
                for path in root.rglob(f"{prefix}*.onnx")
                if Path(f"{path}.json").is_file()
            )
            if len(matches) == 1:
                configured.setdefault(language, matches[0])
    return configured


def _tts_status() -> ArtifactStatus:
    models = _configured_piper_models()
    complete = {
        language: path
        for language, path in models.items()
        if path.is_file() and Path(f"{path}.json").is_file()
    }
    required = set(TTS_VOICE_ALIASES)
    missing = sorted(required - set(complete))
    package_ready = _package_available("piper")
    available = not missing and package_ready
    location = settings.PIPER_VOICES_DIR or "PIPER_VOICE_MAP"
    if available:
        info = f"ready ({len(complete)} voices)"
    else:
        detail = ", ".join(missing) if missing else "Python package piper-tts"
        info = f"missing voices/runtime: {detail}"
    return ArtifactStatus(available, location, info)


def model_artifact_status() -> dict[str, ArtifactStatus]:
    """Return availability without loading a model or accessing the network."""
    return {
        "translation": _translation_status(),
        "ocr": _ocr_status(),
        "stt": _whisper_status(),
        "tts": _tts_status(),
    }
