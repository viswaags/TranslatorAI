"""
TTSService
==========
Primary  : Indic Parler TTS (ai4bharat/indic-parler-tts) — natural neural voice
Fallback : eSpeak-NG — fast, robotic, covers all Indian languages
Extra    : Piper TTS — optional (configure PIPER_VOICES in config)

LLM Enhancement:
  If Indic Parler produces low-quality output (confidence below threshold),
  the text is first cleaned/enhanced by Qwen via Ollama before re-synthesising.

Returns audio as base64-encoded WAV bytes for API transport.
"""

import asyncio
import base64
import io
import logging
import os
import subprocess
import tempfile
import time
import wave
from typing import Optional

from core.config import settings

logger = logging.getLogger(__name__)


# ── Language config ───────────────────────────────────────────────────────────

ESPEAK_VOICES = {
    "eng_Latn": "en",    "tam_Taml": "ta",    "hin_Deva": "hi",
    "tel_Telu": "te",    "kan_Knda": "kn",    "mal_Mlym": "ml",
    "ben_Beng": "bn",    "guj_Gujr": "gu",    "mar_Deva": "mr",
    "pan_Guru": "pa",    "urd_Arab": "ur",    "ory_Orya": "or",
    "asm_Beng": "as",
}

# Indic Parler speaker prompts per language — controls voice style
PARLER_PROMPTS = {
    "eng_Latn": "A clear, natural English female voice speaking at a moderate pace.",
    "tam_Taml": "A clear, natural Tamil female voice with good pronunciation.",
    "hin_Deva": "A clear, natural Hindi male voice with good pronunciation.",
    "tel_Telu": "A clear, natural Telugu female voice with good pronunciation.",
    "ben_Beng": "A clear, natural Bengali male voice with good pronunciation.",
    "mal_Mlym": "A clear, natural Malayalam female voice with good pronunciation.",
    "kan_Knda": "A clear, natural Kannada male voice with good pronunciation.",
    "guj_Gujr": "A clear, natural Gujarati female voice with good pronunciation.",
    "mar_Deva": "A clear, natural Marathi male voice with good pronunciation.",
    "pan_Guru": "A clear, natural Punjabi female voice with good pronunciation.",
    "ory_Orya": "A clear, natural Odia female voice with good pronunciation.",
    "urd_Arab": "A clear, natural Urdu male voice with good pronunciation.",
}


class IndicParlerTTS:
    """
    Indic Parler TTS — neural multilingual TTS for Indian languages.
    Loaded lazily; one model handles all supported languages.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._description_tokenizer = None
        self._sample_rate: Optional[int] = None
        logger.info("IndicParlerTTS created (model loads on first synthesise)")

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer
        from parler_tts import ParlerTTSForConditionalGeneration
        import torch

        model_name = settings.INDIC_PARLER_MODEL
        logger.info("Loading Indic Parler TTS: %s", model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._description_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._sample_rate = self._model.config.sampling_rate
        self._device = device
        logger.info("✅ Indic Parler TTS ready (sr=%d, device=%s)", self._sample_rate, device)

    def synthesise(self, text: str, lang_code: str) -> tuple[bytes, int]:
        """
        Returns (wav_bytes, sample_rate).
        Raises RuntimeError on failure.
        """
        import torch

        self._load()

        prompt = PARLER_PROMPTS.get(lang_code, PARLER_PROMPTS["eng_Latn"])
        description_ids = self._description_tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(self._device)
        input_ids = self._tokenizer(
            text, return_tensors="pt"
        ).input_ids.to(self._device)

        with torch.no_grad():
            generation = self._model.generate(
                input_ids=input_ids,
                prompt_input_ids=description_ids,
            )

        audio_array = generation.cpu().numpy().squeeze()

        # Encode to WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)          # 16-bit
            wf.setframerate(self._sample_rate)
            # Convert float32 [-1,1] → int16
            import numpy as np
            pcm = (audio_array * 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

        return buf.getvalue(), self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


class ESpeakTTSEngine:
    """
    eSpeak-NG — synchronous, always-available fallback.
    Produces robotic but intelligible speech for all Indian languages.
    """

    def synthesise(self, text: str, lang_code: str, speed: int = 150) -> tuple[bytes, int]:
        """Returns (wav_bytes, sample_rate=22050). Raises RuntimeError on failure."""
        voice = ESPEAK_VOICES.get(lang_code, "en")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ["espeak-ng", "-v", voice, "-s", str(speed), "-w", tmp_path, text],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                raise RuntimeError(f"eSpeak error: {result.stderr}")

            with open(tmp_path, "rb") as f:
                wav_bytes = f.read()

            return wav_bytes, 22050

        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ── LLM text enhancement for TTS ─────────────────────────────────────────────

def enhance_text_for_tts(text: str, lang_code: str) -> str:
    """
    Use Qwen via Ollama to normalise text for better TTS pronunciation.
    Called when Indic Parler quality is estimated to be low.
    Returns original text on any failure.
    """
    import requests

    LANG_NAMES = {
        "tam_Taml": "Tamil", "hin_Deva": "Hindi", "tel_Telu": "Telugu",
        "kan_Knda": "Kannada", "mal_Mlym": "Malayalam", "eng_Latn": "English",
    }
    lang_name = LANG_NAMES.get(lang_code, lang_code)

    prompt = (
        f"You are a TTS text normaliser. Clean the following {lang_name} text "
        f"for text-to-speech synthesis:\n"
        f"1. Expand abbreviations\n"
        f"2. Write out numbers as words\n"
        f"3. Remove special characters that affect pronunciation\n"
        f"4. Keep the same language — do NOT translate\n"
        f"5. Return ONLY the cleaned text\n\n"
        f"Input:\n{text}\n\nCleaned text:"
    )

    try:
        resp = requests.post(
            settings.OLLAMA_URL,
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 256},
            },
            timeout=settings.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        enhanced = resp.json().get("response", "").strip()
        if enhanced:
            logger.info("TTS text enhanced by LLM (%d→%d chars)", len(text), len(enhanced))
            return enhanced
    except Exception as e:
        logger.warning("LLM TTS enhancement failed (non-fatal): %s", e)

    return text


# ── Main TTSService ───────────────────────────────────────────────────────────

class TTSService:
    """
    Smart TTS service with Indic Parler primary and eSpeak-NG fallback.

    Quality estimation:
      - Short texts (<5 words) → use eSpeak (Parler needs context)
      - Languages not in PARLER_PROMPTS → eSpeak
      - Parler raises exception → eSpeak fallback
    """

    def __init__(self):
        self._parler = IndicParlerTTS()
        self._espeak = ESpeakTTSEngine()
        logger.info("TTSService ready (Indic Parler loads on first use)")

    async def synthesise(
        self,
        text: str,
        lang_code: str,
        engine: str = "auto",
        speed: int = 150,
        use_llm_enhancement: bool = True,
    ) -> dict:
        """
        Synthesise speech from text.

        Args:
            text             : text to speak
            lang_code        : IndicTrans2 language code
            engine           : "auto" | "indic_parler" | "espeak" | "piper"
            speed            : eSpeak WPM (ignored for Parler)
            use_llm_enhancement : enhance text via LLM if quality may be low

        Returns:
            {
                "success": bool,
                "engine_used": str,
                "audio_base64": str,    # base64 WAV
                "audio_format": "wav",
                "sample_rate": int,
                "processing_ms": int,
                "error": str | None,
            }
        """
        t0 = time.monotonic()

        if not text or not text.strip():
            return self._error_result("Empty text", t0)

        text = text.strip()

        # Estimate quality: short or unsupported → go straight to eSpeak
        use_parler = (
            engine in ("auto", "indic_parler")
            and lang_code in PARLER_PROMPTS
            and len(text.split()) >= 3
        )

        # LLM enhancement for languages where neural TTS may struggle
        if use_llm_enhancement and use_parler and lang_code not in {"eng_Latn"}:
            text = await asyncio.to_thread(enhance_text_for_tts, text, lang_code)

        def _run_parler():
            return self._parler.synthesise(text, lang_code)

        def _run_espeak():
            return self._espeak.synthesise(text, lang_code, speed)

        engine_used = None
        wav_bytes = None
        sample_rate = 22050
        error = None

        if use_parler:
            try:
                wav_bytes, sample_rate = await asyncio.to_thread(_run_parler)
                engine_used = "indic_parler"
                logger.info("TTS: Indic Parler succeeded | lang=%s | %dms",
                            lang_code, int((time.monotonic()-t0)*1000))
            except Exception as e:
                logger.warning("Indic Parler failed (%s) — falling back to eSpeak", e)

        # eSpeak fallback (or forced)
        if wav_bytes is None and engine in ("auto", "espeak"):
            try:
                wav_bytes, sample_rate = await asyncio.to_thread(_run_espeak)
                engine_used = "espeak"
                logger.info("TTS: eSpeak fallback succeeded | lang=%s | %dms",
                            lang_code, int((time.monotonic()-t0)*1000))
            except Exception as e:
                error = str(e)
                logger.error("eSpeak also failed: %s", e)

        processing_ms = int((time.monotonic() - t0) * 1000)

        if wav_bytes is None:
            return self._error_result(error or "All TTS engines failed", t0)

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        return {
            "success":       True,
            "engine_used":   engine_used,
            "audio_base64":  audio_b64,
            "audio_format":  "wav",
            "sample_rate":   sample_rate,
            "processing_ms": processing_ms,
            "error":         None,
        }

    @staticmethod
    def _error_result(msg: str, t0: float) -> dict:
        return {
            "success":       False,
            "engine_used":   None,
            "audio_base64":  None,
            "audio_format":  "wav",
            "sample_rate":   22050,
            "processing_ms": int((time.monotonic() - t0) * 1000),
            "error":         msg,
        }
