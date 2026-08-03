"""Immutable language metadata shared by every backend subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional


@dataclass(frozen=True)
class LanguageMetadata:
    """Canonical language identity plus current engine capabilities."""

    code: str
    display_name: str
    iso_code: str
    script: str
    ocr_alias: Optional[str] = None
    ocr_model_family: Optional[str] = None
    stt_alias: Optional[str] = None
    tts_voice_aliases: tuple[str, ...] = ()
    translation_supported: bool = True
    ocr_supported: bool = False
    stt_supported: bool = True
    tts_supported: bool = True


LANGUAGES: tuple[LanguageMetadata, ...] = (
    LanguageMetadata(
        "eng_Latn", "English", "en", "Latn",
        "en", "en", "en", ("en_",), ocr_supported=True,
    ),
    LanguageMetadata(
        "tam_Taml", "Tamil", "ta", "Taml",
        "ta", "ta", "ta", ("ta_",), ocr_supported=True,
    ),
    LanguageMetadata(
        "hin_Deva", "Hindi", "hi", "Deva",
        "hi", "devanagari", "hi", ("hi_",), ocr_supported=True,
    ),
    LanguageMetadata(
        "tel_Telu", "Telugu", "te", "Telu",
        "te", "te", "te", ("te_",), ocr_supported=True,
    ),
    LanguageMetadata(
        "kan_Knda", "Kannada", "kn", "Knda",
        "ka", "ka", "kn", ("kn_",), ocr_supported=True,
    ),
    LanguageMetadata(
        "mal_Mlym", "Malayalam", "ml", "Mlym",
        stt_alias="ml", tts_voice_aliases=("ml_",),
    ),
    LanguageMetadata(
        "ben_Beng", "Bengali", "bn", "Beng",
        stt_alias="bn", tts_voice_aliases=("bn_",),
    ),
    LanguageMetadata(
        "guj_Gujr", "Gujarati", "gu", "Gujr",
        stt_alias="gu", tts_voice_aliases=("gu_",),
    ),
    LanguageMetadata(
        "mar_Deva", "Marathi", "mr", "Deva",
        "mr", "devanagari", "mr", ("mr_",), ocr_supported=True,
    ),
    LanguageMetadata(
        "pan_Guru", "Punjabi", "pa", "Guru",
        stt_alias="pa", tts_voice_aliases=("pa_",),
    ),
    LanguageMetadata(
        "ory_Orya", "Odia", "or", "Orya",
        stt_alias="or", tts_voice_aliases=("or_",),
    ),
    LanguageMetadata(
        "urd_Arab", "Urdu", "ur", "Arab",
        stt_alias="ur", tts_voice_aliases=("ur_",),
    ),
    LanguageMetadata(
        "asm_Beng", "Assamese", "as", "Beng",
        stt_alias="as", tts_voice_aliases=("as_",),
    ),
)

LANGUAGE_BY_CODE: Mapping[str, LanguageMetadata] = MappingProxyType(
    {language.code: language for language in LANGUAGES}
)
LANGUAGE_NAMES: Mapping[str, str] = MappingProxyType(
    {language.code: language.display_name for language in LANGUAGES}
)
ISO_TO_INDICTRANS: Mapping[str, str] = MappingProxyType(
    {language.iso_code: language.code for language in LANGUAGES}
)
STT_TO_INDICTRANS: Mapping[str, str] = MappingProxyType(
    {
        language.stt_alias: language.code
        for language in LANGUAGES
        if language.stt_supported and language.stt_alias
    }
)
OCR_LANGUAGE_MAP: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        language.code: (language.ocr_alias, language.ocr_model_family)
        for language in LANGUAGES
        if (
            language.ocr_supported
            and language.ocr_alias
            and language.ocr_model_family
        )
    }
)
TTS_VOICE_ALIASES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        language.code: language.tts_voice_aliases
        for language in LANGUAGES
        if language.tts_supported and language.tts_voice_aliases
    }
)

TRANSLATION_LANGUAGE_CODES = frozenset(
    language.code for language in LANGUAGES if language.translation_supported
)
OCR_LANGUAGE_CODES = frozenset(
    language.code for language in LANGUAGES if language.ocr_supported
)
STT_LANGUAGE_CODES = frozenset(
    language.code for language in LANGUAGES if language.stt_supported
)
TTS_LANGUAGE_CODES = frozenset(
    language.code for language in LANGUAGES if language.tts_supported
)

ENGLISH_CODE = ISO_TO_INDICTRANS["en"]
HINDI_CODE = ISO_TO_INDICTRANS["hi"]
MARATHI_CODE = ISO_TO_INDICTRANS["mr"]

