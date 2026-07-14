"""
Configuration — loaded once at startup.
Override any value via environment variables.
"""

import os
from typing import List


class Settings:
    # ── Server ─────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # ── CORS ───────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", "*"
    ).split(",")

    # ── Whisper STT ────────────────────────────────────────────────────────────
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    # tiny / base / small / medium — base is best balance for Raspberry Pi 5

    # ── Translation ────────────────────────────────────────────────────────────
    TRANSLATION_PRIMARY: str = os.getenv("TRANSLATION_PRIMARY", "indictrans2")
    # "indictrans2" or "m2m100" (m2m100 is the commented fallback)

    INDICTRANS2_EN_INDIC: str = os.getenv(
        "INDICTRANS2_EN_INDIC",
        "ai4bharat/indictrans2-en-indic-dist-200M",
    )
    INDICTRANS2_INDIC_EN: str = os.getenv(
        "INDICTRANS2_INDIC_EN",
        "ai4bharat/indictrans2-indic-en-dist-200M",
    )

    # M2M-100 fallback (uncomment in TranslationService to activate)
    M2M100_MODEL: str = os.getenv("M2M100_MODEL", "facebook/m2m100_418M")

    # ── OCR ────────────────────────────────────────────────────────────────────
    OCR_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.60")
    )
    OCR_LLM_CORRECTION: bool = (
        os.getenv("OCR_LLM_CORRECTION", "true").lower() == "true"
    )

    # ── TTS ────────────────────────────────────────────────────────────────────
    TTS_PRIMARY: str = os.getenv("TTS_PRIMARY", "indic_parler")
    # "indic_parler" or "espeak" (espeak is the safe fallback)

    INDIC_PARLER_MODEL: str = os.getenv(
        "INDIC_PARLER_MODEL", "ai4bharat/indic-parler-tts"
    )
    PIPER_VOICES_DIR: str = os.getenv("PIPER_VOICES_DIR", "")

    TTS_LOW_QUALITY_THRESHOLD: float = float(
        os.getenv("TTS_LOW_QUALITY_THRESHOLD", "0.60")
    )
    # If TTS quality score drops below this, LLM enhancement is tried

    # ── LLM (Ollama / Qwen) ────────────────────────────────────────────────────
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "60"))

    # ── Upload paths ───────────────────────────────────────────────────────────
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/ai_translator_uploads")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/tmp/ai_translator_output")

    # ── Language defaults ──────────────────────────────────────────────────────
    DEFAULT_TARGET_LANG: str = os.getenv("DEFAULT_TARGET_LANG", "eng_Latn")


settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
