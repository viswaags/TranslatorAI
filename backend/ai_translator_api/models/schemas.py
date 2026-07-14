"""
Shared Pydantic Models
======================
Request/Response schemas for all API endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Supported language codes ──────────────────────────────────────────────────

class Language(str, Enum):
    """IndicTrans2 language codes supported across the pipeline."""
    ENGLISH    = "eng_Latn"
    TAMIL      = "tam_Taml"
    HINDI      = "hin_Deva"
    TELUGU     = "tel_Telu"
    KANNADA    = "kan_Knda"
    MALAYALAM  = "mal_Mlym"
    BENGALI    = "ben_Beng"
    GUJARATI   = "guj_Gujr"
    MARATHI    = "mar_Deva"
    PUNJABI    = "pan_Guru"
    ODIA       = "ory_Orya"
    URDU       = "urd_Arab"
    ASSAMESE   = "asm_Beng"


LANGUAGE_NAMES = {
    "eng_Latn": "English",
    "tam_Taml": "Tamil",
    "hin_Deva": "Hindi",
    "tel_Telu": "Telugu",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "ben_Beng": "Bengali",
    "guj_Gujr": "Gujarati",
    "mar_Deva": "Marathi",
    "pan_Guru": "Punjabi",
    "ory_Orya": "Odia",
    "urd_Arab": "Urdu",
    "asm_Beng": "Assamese",
}


# ── TTS config embedded in requests ──────────────────────────────────────────

class TTSOptions(BaseModel):
    enabled: bool = True
    engine: str = Field(
        default="auto",
        description="'auto' | 'indic_parler' | 'espeak' | 'piper'",
    )
    speed: int = Field(default=150, ge=80, le=350, description="eSpeak WPM")
    return_audio: bool = Field(
        default=True,
        description="If True, return base64 WAV audio in the response",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEXT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class TextTranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to translate")
    target_lang: str = Field(..., description="Target language code e.g. 'tam_Taml'")
    source_lang: Optional[str] = Field(
        default=None,
        description="Source language (auto-detected if omitted)",
    )
    tts: TTSOptions = Field(default_factory=TTSOptions)

    model_config = {"json_schema_extra": {
        "example": {
            "text": "Hello, how are you?",
            "target_lang": "tam_Taml",
            "tts": {"enabled": True, "return_audio": True},
        }
    }}


class TextTranslateResponse(BaseModel):
    success: bool
    input_text: str
    detected_source_lang: str
    detected_source_lang_name: str
    target_lang: str
    target_lang_name: str
    translated_text: str
    translation_engine: str          # "indictrans2" | "m2m100"
    tts: Optional["TTSResult"] = None
    processing_ms: int
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# OCR PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class OCRExtractResponse(BaseModel):
    """Response from POST /ocr/extract — OCR only, no translation."""
    success: bool
    extracted_text: str
    detected_lang: str
    detected_lang_name: str
    ocr_confidence: float
    ocr_engine_used: str             # "paddle" | "tesseract" | "paddle+tesseract"
    llm_corrected: bool
    processing_ms: int
    error: Optional[str] = None


class OCRTranslateRequest(BaseModel):
    """
    Sent as form-data alongside the image file upload.
    Target language is required; source language is auto-detected from the image.
    """
    target_lang: str = Field(..., description="Target language code")
    source_lang_hint: Optional[str] = Field(
        default=None,
        description="Optional hint for OCR language detection",
    )
    tts: TTSOptions = Field(default_factory=TTSOptions)


class OCRTranslateResponse(BaseModel):
    success: bool
    extracted_text: str
    detected_source_lang: str
    detected_source_lang_name: str
    target_lang: str
    target_lang_name: str
    translated_text: str
    ocr_confidence: float
    ocr_engine_used: str
    llm_corrected: bool
    translation_engine: str
    tts: Optional["TTSResult"] = None
    processing_ms: int
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# SPEECH PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class SpeechTranslateResponse(BaseModel):
    success: bool
    transcribed_text: str
    detected_source_lang: str           # Whisper code e.g. "ta"
    detected_source_lang_indictrans: str  # IndicTrans2 code e.g. "tam_Taml"
    detected_source_lang_name: str
    whisper_confidence: float
    target_lang: str
    target_lang_name: str
    translated_text: str
    translation_engine: str
    tts: Optional["TTSResult"] = None
    processing_ms: int
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# TTS RESULT (shared across all pipelines)
# ─────────────────────────────────────────────────────────────────────────────

class TTSResult(BaseModel):
    success: bool
    engine_used: Optional[str]          # "indic_parler" | "espeak" | "piper"
    audio_base64: Optional[str]         # WAV bytes, base64-encoded
    audio_format: str = "wav"
    sample_rate: int = 22050
    processing_ms: int


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH / STATUS
# ─────────────────────────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    loaded: bool
    model_name: Optional[str] = None
    info: Optional[str] = None


class HealthResponse(BaseModel):
    status: str                         # "ok" | "degraded" | "error"
    version: str
    services: dict[str, ServiceStatus]
    supported_languages: list[dict]


# Rebuild models with forward references
TextTranslateResponse.model_rebuild()
OCRTranslateResponse.model_rebuild()
SpeechTranslateResponse.model_rebuild()
