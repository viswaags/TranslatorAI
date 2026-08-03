"""
Shared Pydantic Models
======================
Request/Response schemas for all API endpoints.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── TTS config embedded in requests ──────────────────────────────────────────

class TTSOptions(BaseModel):
    enabled: bool = True
    engine: str = Field(
        default="auto",
        description="Speech synthesis engine: 'auto' or 'piper'",
    )
    speed: int = Field(
        default=150,
        ge=80,
        le=350,
        description="Piper speech speed; 150 is the default",
    )
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
    translation_engine: str          # "indictrans2" | "passthrough"
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
    ocr_engine_used: str             # "paddle"
    llm_corrected: bool
    processing_ms: int
    error: Optional[str] = None


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
    engine_used: Optional[str]          # "piper" when synthesis succeeds
    audio_base64: Optional[str]         # WAV bytes, base64-encoded
    audio_format: str = "wav"
    sample_rate: int = 22050
    processing_ms: int


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH / STATUS
# ─────────────────────────────────────────────────────────────────────────────

class ServiceStatus(BaseModel):
    loaded: bool
    available: Optional[bool] = None
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
