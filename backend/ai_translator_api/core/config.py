"""
Configuration — loaded once at startup.
Override any value via environment variables.
"""

import os
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings:
    # ── CORS ───────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", "*"
    ).split(",")

    # ── Whisper STT ────────────────────────────────────────────────────────────
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    # tiny / base / small / medium — base is best balance for Raspberry Pi 5
    WHISPER_MODEL_PATH: str = os.getenv("WHISPER_MODEL_PATH", "")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    WHISPER_CPU_THREADS: int = int(os.getenv("WHISPER_CPU_THREADS", "4"))
    WHISPER_NUM_WORKERS: int = int(os.getenv("WHISPER_NUM_WORKERS", "1"))
    WHISPER_BEAM_SIZE: int = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    WHISPER_BEST_OF: int = int(os.getenv("WHISPER_BEST_OF", "5"))
    WHISPER_VAD_FILTER: bool = (
        os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
    )
    WHISPER_MIN_SILENCE_MS: int = int(
        os.getenv("WHISPER_MIN_SILENCE_MS", "500")
    )
    STT_MAX_DURATION_SECONDS: float = float(
        os.getenv("STT_MAX_DURATION_SECONDS", "300")
    )
    STT_MIN_SAMPLE_RATE: int = int(os.getenv("STT_MIN_SAMPLE_RATE", "8000"))
    STT_MAX_SAMPLE_RATE: int = int(os.getenv("STT_MAX_SAMPLE_RATE", "192000"))
    STT_BATCH_SIZE: int = int(os.getenv("STT_BATCH_SIZE", "4"))
    STT_SUPPORTED_EXTENSIONS: List[str] = os.getenv(
        "STT_SUPPORTED_EXTENSIONS", ".wav,.mp3,.ogg,.flac,.m4a,.aac"
    ).lower().split(",")

    # ── Translation ────────────────────────────────────────────────────────────
    INDICTRANS2_EN_INDIC: str = os.getenv(
        "INDICTRANS2_EN_INDIC",
        str(PROJECT_ROOT / "models" / "indictrans2-en-indic-dist-200M-ct2-int8"),
    )
    INDICTRANS2_INDIC_EN: str = os.getenv(
        "INDICTRANS2_INDIC_EN",
        str(PROJECT_ROOT / "models" / "indictrans2-indic-en-dist-200M-ct2-int8"),
    )

    TRANSLATION_MAX_INPUT_TOKENS: int = int(
        os.getenv("TRANSLATION_MAX_INPUT_TOKENS", "256")
    )
    TRANSLATION_MAX_OUTPUT_TOKENS: int = int(
        os.getenv("TRANSLATION_MAX_OUTPUT_TOKENS", "256")
    )
    TRANSLATION_NUM_BEAMS: int = int(os.getenv("TRANSLATION_NUM_BEAMS", "5"))
    TRANSLATION_BATCH_SIZE: int = int(os.getenv("TRANSLATION_BATCH_SIZE", "8"))
    TRANSLATION_COMPUTE_TYPE: str = os.getenv(
        "TRANSLATION_COMPUTE_TYPE", "int8"
    )
    TRANSLATION_INTER_THREADS: int = int(
        os.getenv("TRANSLATION_INTER_THREADS", "2")
    )
    TRANSLATION_INTRA_THREADS: int = int(
        os.getenv("TRANSLATION_INTRA_THREADS", "2")
    )
    TRANSLATION_NO_REPEAT_NGRAM_SIZE: int = int(
        os.getenv("TRANSLATION_NO_REPEAT_NGRAM_SIZE", "2")
    )

    # ── OCR ────────────────────────────────────────────────────────────────────
    OCR_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.60")
    )
    OCR_MODEL_ROOT: str = os.getenv(
        "OCR_MODEL_ROOT", str(Path.home() / ".paddleocr" / "whl")
    )
    OCR_USE_ANGLE_CLASSIFIER: bool = (
        os.getenv("OCR_USE_ANGLE_CLASSIFIER", "true").lower() == "true"
    )
    OCR_ENABLE_MKLDNN: bool = (
        os.getenv("OCR_ENABLE_MKLDNN", "true").lower() == "true"
    )
    OCR_CPU_THREADS: int = int(os.getenv("OCR_CPU_THREADS", "4"))
    OCR_BATCH_SIZE: int = int(os.getenv("OCR_BATCH_SIZE", "4"))
    OCR_SUPPORTED_EXTENSIONS: List[str] = os.getenv(
        "OCR_SUPPORTED_EXTENSIONS", ".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp"
    ).lower().split(",")

    # ── TTS ────────────────────────────────────────────────────────────────────
    PIPER_VOICES_DIR: str = os.getenv("PIPER_VOICES_DIR", "")
    PIPER_VOICE_MAP: str = os.getenv("PIPER_VOICE_MAP", "")
    PIPER_LENGTH_SCALE: float = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))
    PIPER_NOISE_SCALE: float = float(os.getenv("PIPER_NOISE_SCALE", "0.667"))
    PIPER_NOISE_W_SCALE: float = float(
        os.getenv("PIPER_NOISE_W_SCALE", "0.8")
    )
    PIPER_VOLUME: float = float(os.getenv("PIPER_VOLUME", "1.0"))
    TTS_MAX_TEXT_CHARS: int = int(os.getenv("TTS_MAX_TEXT_CHARS", "2000"))
    TTS_BATCH_SIZE: int = int(os.getenv("TTS_BATCH_SIZE", "4"))
    TTS_MIN_SPEED: int = int(os.getenv("TTS_MIN_SPEED", "80"))
    TTS_DEFAULT_SPEED: int = int(os.getenv("TTS_DEFAULT_SPEED", "150"))
    TTS_MAX_SPEED: int = int(os.getenv("TTS_MAX_SPEED", "350"))

    TTS_LOW_QUALITY_THRESHOLD: float = float(
        os.getenv("TTS_LOW_QUALITY_THRESHOLD", "0.60")
    )

    # ── Upload paths ───────────────────────────────────────────────────────────
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/ai_translator_uploads")
    UPLOAD_CHUNK_SIZE_BYTES: int = int(
        os.getenv("UPLOAD_CHUNK_SIZE_BYTES", str(1024 * 1024))
    )
    OCR_MAX_UPLOAD_SIZE_BYTES: int = int(
        os.getenv("OCR_MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024))
    )
    SPEECH_MAX_UPLOAD_SIZE_BYTES: int = int(
        os.getenv("SPEECH_MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024))
    )

settings = Settings()

# Ensure the active upload directory exists.
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
