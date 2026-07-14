"""
Production TTS Engine (CPU Only + Optimized)
============================================

✔ Piper (primary - natural voice)
✔ eSpeak (fallback - fast + reliable)
✔ CPU only (no CUDA)
✔ ONNX (Piper internally)
✔ Voice caching (no reload)
✔ No playback blocking (safe for servers)
✔ Qwen refinement hook (optional)
✔ Fast + stable + production ready
"""

import os
import subprocess
import tempfile
import time
import wave
import logging
from typing import Optional

import soundfile as sf
import sounddevice as sd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 🔥 Audio defaults (important for speed)
sd.default.samplerate = 22050
sd.default.channels = 1

EN = "eng_Latn"


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ESPEAK_VOICES = {
    "eng_Latn": "en",
    "tam_Taml": "ta",
    "hin_Deva": "hi",
    "tel_Telu": "te",
    "kan_Knda": "kn",
    "mal_Mlym": "ml",
    "ben_Beng": "bn",
    "guj_Gujr": "gu",
    "mar_Deva": "mr",
    "pan_Guru": "pa",
    "urd_Arab": "ur",
}

PIPER_VOICES = {
    # Fill these paths
    # "tam_Taml": ("/models/ta.onnx", "/models/ta.json"),
}


# ─────────────────────────────────────────────
# OPTIONAL: QWEN TEXT REFINEMENT
# ─────────────────────────────────────────────
def refine_text(text: str) -> str:
    """
    Optional refinement using Qwen (Ollama).
    Improves fluency before TTS.
    """
    try:
        import requests

        prompt = f"Make this sentence natural for speech:\n{text}"

        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:1.5b",
                "prompt": prompt,
                "stream": False,
            },
            timeout=10,
        )

        return res.json()["response"].strip()

    except Exception:
        return text  # fallback silently


# ─────────────────────────────────────────────
# ESPEAK (FAST FALLBACK)
# ─────────────────────────────────────────────
class ESpeakTTS:

    def __init__(self):
        if subprocess.run(["which", "espeak-ng"], capture_output=True).returncode != 0:
            raise RuntimeError("Install espeak-ng: sudo apt install espeak-ng")

        logger.info("✅ eSpeak ready")

    def speak(self, text, lang_code, speed=150, output_file=None):
        voice = ESPEAK_VOICES.get(lang_code, "en")

        output = output_file or tempfile.mktemp(suffix=".wav")

        try:
            subprocess.run(
                ["espeak-ng", "-v", voice, "-s", str(speed), "-w", output, text],
                check=True,
                timeout=5,
            )
            return True, output

        except Exception as e:
            logger.error(f"eSpeak failed: {e}")
            return False, None


# ─────────────────────────────────────────────
# PIPER (PRIMARY)
# ─────────────────────────────────────────────
class PiperTTS:

    def __init__(self):
        self.voices = {}
        try:
            from piper import PiperVoice
            self.available = True
        except ImportError:
            self.available = False

        if self.available:
            logger.info("✅ Piper ready")
        else:
            logger.warning("⚠️ Piper not installed")

    def is_available(self, lang):
        return self.available and lang in PIPER_VOICES

    def _load_voice(self, lang):
        from piper import PiperVoice

        model, config = PIPER_VOICES[lang]

        if lang not in self.voices:
            logger.info(f"🔄 Loading Piper voice: {lang}")
            self.voices[lang] = PiperVoice.load(
                model,
                config_path=config,
                use_cuda=False
            )

        return self.voices[lang]

    def speak(self, text, lang, output_file=None):
        if not self.is_available(lang):
            return False, None

        try:
            voice = self._load_voice(lang)
            output = output_file or tempfile.mktemp(suffix=".wav")

            with wave.open(output, "wb") as f:
                voice.synthesize(text[:200], f)  # 🔥 limit text

            return True, output

        except Exception as e:
            logger.error(f"Piper failed: {e}")
            return False, None


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────
class TTSEngine:

    def __init__(self):
        logger.info("🚀 Initializing TTS...")
        self.espeak = ESpeakTTS()
        self.piper = PiperTTS()

    def speak(
        self,
        text: str,
        lang: str,
        output_file: Optional[str] = None,
        refine: bool = False,
        play: bool = False,
    ) -> dict:

        start = time.time()

        if not text.strip():
            return {"success": False}

        # 🔥 Optional refinement
        if refine:
            text = refine_text(text)

        # 🔥 Try Piper
        success, path = self.piper.speak(text, lang, output_file)

        # 🔁 Fallback
        if not success:
            success, path = self.espeak.speak(text, lang, output_file=output_file)

        # 🔊 Playback (optional)
        if play and path:
            try:
                data, sr = sf.read(path)
                sd.play(data, sr)
                sd.wait()
            except Exception:
                pass

        return {
            "success": success,
            "file": path,
            "time": round(time.time() - start, 3),
        }


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":

    tts = TTSEngine()

    tests = [
        ("Hello, how are you?", "eng_Latn"),
        ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "tam_Taml"),
        ("नमस्ते, आप कैसे हैं?", "hin_Deva"),
    ]

    for text, lang in tests:
        res = tts.speak(text, lang, output_file=f"/tmp/{lang}.wav", refine=True)
        print(res)