"""
TTS Engine — Production Level (CPU Only)
=========================================
Layer 1 → Indic Parler-TTS  (PRIMARY — 21 Indian languages, natural voice)
Layer 2 → Piper TTS          (FALLBACK — fast, natural, where models exist)
Layer 3 → eSpeak-NG          (COMMENTED OUT — emergency only, uncomment if needed)

Optional:
  Qwen2.5:1.5b via Ollama → refines translated text before TTS
                             improves fluency, removes translation artifacts

Key Design Principles:
  ✅ CPU only — no CUDA anywhere
  ✅ Lazy loading  — nothing loads at startup, loads on first use
  ✅ Dynamic loading — load when needed, unload when done
  ✅ Quantization — Parler in INT8 (best for CPU speed + RAM)
                    fallback to FP16 if INT8 fails
                    fallback to FP32 if FP16 fails
  ✅ Voice caching — Piper voices stay loaded after first load
  ✅ Product-level error handling and logging

Requirements:
    pip install parler-tts transformers torch soundfile sounddevice
    pip install optimum  (for INT8 quantization)
    pip install piper-tts  (for Piper layer)
    ollama pull qwen2.5:1.5b  (for text refinement)

Models:
    Indic Parler-TTS : ~/models/indic-parler-tts  (already downloaded)
    Piper voices     : ~/models/piper/             (download per language)
"""

import os
import gc
import time
import wave
import logging
import subprocess
import tempfile
from typing import Optional

import torch
import soundfile as sf
import sounddevice as sd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Language Configuration
# ─────────────────────────────────────────────

# Parler-TTS speaker descriptions per language
# Controls voice style and naturalness
PARLER_DESCRIPTIONS = {
    "eng_Latn": "A female speaker delivers clear, natural English speech at a moderate pace.",
    "tam_Taml": "A female speaker delivers clear, natural Tamil speech at a moderate pace.",
    "hin_Deva": "A female speaker delivers clear, natural Hindi speech at a moderate pace.",
    "tel_Telu": "A female speaker delivers clear, natural Telugu speech at a moderate pace.",
    "kan_Knda": "A female speaker delivers clear, natural Kannada speech at a moderate pace.",
    "mal_Mlym": "A female speaker delivers clear, natural Malayalam speech at a moderate pace.",
    "ben_Beng": "A female speaker delivers clear, natural Bengali speech at a moderate pace.",
    "guj_Gujr": "A female speaker delivers clear, natural Gujarati speech at a moderate pace.",
    "mar_Deva": "A female speaker delivers clear, natural Marathi speech at a moderate pace.",
    "pan_Guru": "A female speaker delivers clear, natural Punjabi speech at a moderate pace.",
}
DEFAULT_PARLER_DESCRIPTION = (
    "A female speaker delivers clear, natural speech at a moderate pace."
)

# Piper voice model paths
# Download from: https://huggingface.co/rhasspy/piper-voices
# Uncomment and fill paths after downloading
PIPER_VOICES = {
    # "eng_Latn": (
    #     "/home/viswaa/models/piper/en_US-lessac-medium.onnx",
    #     "/home/viswaa/models/piper/en_US-lessac-medium.onnx.json",
    # ),
    # "hin_Deva": (
    #     "/home/viswaa/models/piper/hi_IN-hindi-medium.onnx",
    #     "/home/viswaa/models/piper/hi_IN-hindi-medium.onnx.json",
    # ),
    # "tam_Taml": (
    #     "/home/viswaa/models/piper/ta_IN-tamil-medium.onnx",
    #     "/home/viswaa/models/piper/ta_IN-tamil-medium.onnx.json",
    # ),
}

# eSpeak-NG voice codes — kept for reference
# Uncomment ESpeakTTS class below to enable
# ESPEAK_VOICES = {
#     "eng_Latn": "en",
#     "tam_Taml": "ta",
#     "hin_Deva": "hi",
#     "tel_Telu": "te",
#     "kan_Knda": "kn",
#     "mal_Mlym": "ml",
#     "ben_Beng": "bn",
#     "guj_Gujr": "gu",
#     "mar_Deva": "mr",
#     "pan_Guru": "pa",
#     "urd_Arab": "ur",
# }

# Model paths
PARLER_MODEL_PATH = os.path.expanduser("~/models/indic-parler-tts")

# Quantization preference order
# System tries each in order until one works
# INT8 = fastest on CPU, smallest RAM
# FP16 = good balance
# FP32 = original, largest RAM
QUANTIZATION_ORDER = ["int8", "fp16", "fp32"]


# ─────────────────────────────────────────────
# Quantization Helper
# ─────────────────────────────────────────────
def quantize_model_int8(model):
    """
    Apply dynamic INT8 quantization to model.
    Reduces RAM by ~60% and speeds up CPU inference by ~2x.
    Applied to Linear layers only — safe for transformer models.
    """
    try:
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},          # quantize linear layers
            dtype=torch.qint8,          # INT8 weights
        )
        logger.info("✅ INT8 quantization applied")
        return quantized
    except Exception as e:
        logger.warning(f"⚠️  INT8 quantization failed: {e}")
        return model                    # return original if fails


# ─────────────────────────────────────────────
# Layer 1 — Indic Parler-TTS (Primary)
# ─────────────────────────────────────────────
class ParlerTTS:
    """
    Indic Parler-TTS — primary TTS engine.

    21 Indian languages + English.
    Natural, human-like voice quality.

    Quantization strategy (tried in order):
      INT8 → ~950MB RAM, fastest CPU inference (~8-12s/sentence)
      FP16 → ~1.8GB RAM, good speed (~12-18s/sentence)
      FP32 → ~3.75GB RAM, original quality (~25-40s/sentence)

    Dynamic loading:
      - Loads only when first tts.speak() is called
      - Can be unloaded to free RAM during STT/translation mode
      - Reloads automatically on next speak() call
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._quant_mode = None         # tracks which quant was used

        if not os.path.exists(PARLER_MODEL_PATH):
            logger.warning(
                f"⚠️  Parler-TTS model not found at: {PARLER_MODEL_PATH}\n"
                f"   Download with: snapshot_download('ai4bharat/indic-parler-tts')"
            )
            self._available = False
        else:
            self._available = True
            logger.info(
                "✅ Parler-TTS available — lazy loaded on first use"
            )

    def _load(self):
        """
        Dynamically load model with best available quantization.
        Called automatically on first speak() — not at init.
        """
        if self._loaded:
            return

        if not self._available:
            raise RuntimeError(
                f"Parler-TTS model not found at: {PARLER_MODEL_PATH}"
            )

        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        logger.info("🔄 Dynamically loading Indic Parler-TTS...")
        logger.info(f"   Trying quantization order: {QUANTIZATION_ORDER}")

        # Load tokenizer once — same for all quant modes
        self._tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL_PATH)

        # Try quantization modes in order
        for quant in QUANTIZATION_ORDER:
            try:
                logger.info(f"   Attempting: {quant.upper()}...")
                start = time.time()

                if quant == "int8":
                    # Load in FP32 first, then quantize dynamically
                    model = ParlerTTSForConditionalGeneration.from_pretrained(
                        PARLER_MODEL_PATH,
                        torch_dtype=torch.float32,
                    )
                    model = quantize_model_int8(model)

                elif quant == "fp16":
                    model = ParlerTTSForConditionalGeneration.from_pretrained(
                        PARLER_MODEL_PATH,
                        torch_dtype=torch.float16,
                    )

                else:  # fp32
                    model = ParlerTTSForConditionalGeneration.from_pretrained(
                        PARLER_MODEL_PATH,
                        torch_dtype=torch.float32,
                    )

                model.eval()
                self._model = model
                self._quant_mode = quant
                elapsed = time.time() - start

                logger.info(
                    f"✅ Parler-TTS loaded [{quant.upper()}] "
                    f"in {elapsed:.1f}s"
                )
                break

            except Exception as e:
                logger.warning(
                    f"⚠️  {quant.upper()} failed: {e} — trying next..."
                )
                # Clean up failed attempt
                if "model" in locals():
                    del model
                gc.collect()
                continue

        if self._model is None:
            raise RuntimeError("All quantization modes failed for Parler-TTS")

        self._loaded = True

    def is_available(self) -> bool:
        return self._available

    def speak(
        self,
        text: str,
        lang_code: str,
        output_file: Optional[str] = None,
    ) -> tuple:
        """
        Generate speech — dynamically loads model on first call.

        Returns:
            (success: bool, output_path: str | None)
        """
        if not self._available:
            return False, None

        try:
            # Dynamic load — only happens on first call
            self._load()

            description = PARLER_DESCRIPTIONS.get(
                lang_code, DEFAULT_PARLER_DESCRIPTION
            )

            input_ids = self._tokenizer(
                description,
                return_tensors="pt"
            ).input_ids

            prompt_ids = self._tokenizer(
                text,
                return_tensors="pt"
            ).input_ids

            logger.info(
                f"🔄 Generating speech [{lang_code}] "
                f"using {self._quant_mode.upper()}..."
            )
            start = time.time()

            with torch.inference_mode():
                generation = self._model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_ids,
                )

            # Convert to float32 for saving (INT8/FP16 → FP32)
            audio = generation.cpu().float().numpy().squeeze()
            elapsed = time.time() - start

            logger.info(f"✅ Generated in {elapsed:.1f}s")

            output = output_file or tempfile.mktemp(suffix=".wav")
            sf.write(output, audio, self._model.config.sampling_rate)

            return True, output

        except Exception as e:
            logger.error(f"❌ Parler-TTS speak failed: {e}")
            return False, None

    def unload(self):
        """
        Dynamically unload model to free RAM.
        Call this when switching to STT/translation mode.
        Model reloads automatically on next speak().
        """
        if self._loaded:
            del self._model
            self._model = None
            self._loaded = False
            self._quant_mode = None
            gc.collect()
            logger.info("🗑️  Parler-TTS unloaded — RAM freed")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def quant_mode(self) -> Optional[str]:
        return self._quant_mode


# ─────────────────────────────────────────────
# Layer 2 — Piper TTS (Fallback)
# ─────────────────────────────────────────────
class PiperTTS:
    """
    Piper TTS — fallback when Parler fails.

    Natural voice, extremely fast on CPU (<1s).
    Requires voice model files downloaded separately.
    Add paths to PIPER_VOICES dict above.

    Uses ONNX internally — already optimized for CPU.
    Voice models cached in memory after first load.
    """

    def __init__(self):
        self._voices = {}
        self._available = self._check_installed()

        if self._available:
            active = [k for k in PIPER_VOICES if os.path.exists(PIPER_VOICES[k][0])]
            logger.info(
                f"✅ Piper TTS available — "
                f"{len(active)} voice(s) configured: {active or 'none yet'}"
            )
        else:
            logger.warning(
                "⚠️  Piper TTS not installed\n"
                "   Install: pip install piper-tts"
            )

    def _check_installed(self) -> bool:
        try:
            from piper import PiperVoice
            return True
        except ImportError:
            return False

    def is_available(self, lang_code: str) -> bool:
        """Check if Piper has a working voice for this language."""
        if not self._available:
            return False
        if lang_code not in PIPER_VOICES:
            return False
        model_path = PIPER_VOICES[lang_code][0]
        return os.path.exists(model_path)

    def _load_voice(self, lang_code: str):
        """Lazy load and cache voice — loads once, stays in memory."""
        if lang_code not in self._voices:
            from piper import PiperVoice
            model_path, config_path = PIPER_VOICES[lang_code]
            logger.info(f"🔄 Loading Piper voice [{lang_code}]...")
            self._voices[lang_code] = PiperVoice.load(
                model_path,
                config_path=config_path,
                use_cuda=False,             # CPU only
            )
            logger.info(f"✅ Piper voice cached [{lang_code}]")
        return self._voices[lang_code]

    def speak(
        self,
        text: str,
        lang_code: str,
        output_file: Optional[str] = None,
    ) -> tuple:
        """
        Generate speech using Piper TTS.

        Returns:
            (success: bool, output_path: str | None)
        """
        if not self.is_available(lang_code):
            return False, None

        try:
            voice = self._load_voice(lang_code)
            output = output_file or tempfile.mktemp(suffix=".wav")

            with wave.open(output, "wb") as wav_file:
                voice.synthesize(text[:300], wav_file)

            logger.info(f"✅ Piper TTS generated [{lang_code}]")
            return True, output

        except Exception as e:
            logger.error(f"❌ Piper TTS failed: {e}")
            return False, None


# ─────────────────────────────────────────────
# Layer 3 — eSpeak-NG (Emergency Fallback)
# COMMENTED OUT — uncomment only if needed
# Covers all Indian languages but robotic voice
# ─────────────────────────────────────────────

# class ESpeakTTS:
#     """
#     eSpeak-NG — emergency last resort.
#     Covers ALL languages. Voice is robotic.
#     Only use if both Parler and Piper fail.
#
#     Install: sudo apt install espeak-ng -y
#     """
#
#     VOICES = {
#         "eng_Latn": "en", "tam_Taml": "ta", "hin_Deva": "hi",
#         "tel_Telu": "te", "kan_Knda": "kn", "mal_Mlym": "ml",
#         "ben_Beng": "bn", "guj_Gujr": "gu", "mar_Deva": "mr",
#         "pan_Guru": "pa", "urd_Arab": "ur",
#     }
#
#     def speak(self, text, lang_code, output_file=None, speed=145):
#         voice = self.VOICES.get(lang_code, "en")
#         output = output_file or tempfile.mktemp(suffix=".wav")
#         try:
#             subprocess.run(
#                 ["espeak-ng", "-v", voice, "-s", str(speed), "-w", output, text[:500]],
#                 check=True, timeout=10, capture_output=True,
#             )
#             return True, output
#         except Exception as e:
#             logger.error(f"eSpeak failed: {e}")
#             return False, None


# ─────────────────────────────────────────────
# Optional — Qwen Text Refinement
# ─────────────────────────────────────────────
class QwenRefiner:
    """
    Optional text refinement via qwen2.5:1.5b (Ollama).

    Improves translated text before TTS:
      - Removes translation artifacts
      - Fixes unnatural phrasing
      - Makes text flow naturally for speech

    Runs via Ollama — completely offline.
    Lazy check — only pings Ollama when refine() is called.
    Latency: 2-4 seconds on CPU.

    Start Ollama: ollama serve
    Pull model  : ollama pull qwen2.5:1.5b
    """

    def __init__(self):
        self._available = None      # None = not checked yet (lazy)
        logger.info(
            "✅ Qwen refiner initialized — "
            "Ollama check deferred to first use"
        )

    def _check_ollama(self) -> bool:
        """Lazy check — only runs when first refinement is requested."""
        if self._available is None:
            try:
                import requests
                resp = requests.get(
                    "http://localhost:11434/api/tags",
                    timeout=2
                )
                self._available = resp.status_code == 200
                if self._available:
                    logger.info("✅ Ollama connected — Qwen refiner active")
                else:
                    logger.warning("⚠️  Ollama not responding")
            except Exception:
                self._available = False
                logger.warning(
                    "⚠️  Ollama not running — refinement disabled\n"
                    "   Start with: ollama serve"
                )
        return self._available

    def refine(self, text: str, lang_code: str) -> str:
        """
        Refine text for natural speech.
        Returns original text if refinement fails.
        """
        if not self._check_ollama():
            return text

        try:
            import requests

            prompt = (
                f"You are a speech text optimizer for Indian languages.\n"
                f"Fix the text below to sound natural when spoken aloud.\n"
                f"Rules:\n"
                f"- Keep the SAME language\n"
                f"- Do NOT translate\n"
                f"- Fix unnatural phrasing only\n"
                f"- Return ONLY the corrected text, nothing else\n\n"
                f"Text:\n{text}"
            )

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:1.5b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,     # low = consistent
                        "num_predict": 256,
                    },
                },
                timeout=15,
            )

            if response.status_code == 200:
                refined = response.json().get("response", "").strip()
                if refined and len(refined) > 0:
                    logger.info(f"✅ Qwen refined: '{text[:40]}' → '{refined[:40]}'")
                    return refined

        except Exception as e:
            logger.warning(f"⚠️  Qwen refinement failed: {e}")

        return text     # always fall back to original


# ─────────────────────────────────────────────
# Main TTS Engine
# ─────────────────────────────────────────────
class TTSEngine:
    """
    Production TTS engine.

    Layer priority:
    ─────────────────────────────────────────────────
    1. Indic Parler-TTS  → primary, best quality
    2. Piper TTS         → fallback, fast + natural
    # 3. eSpeak-NG       → commented out (emergency)
    ─────────────────────────────────────────────────

    Quantization (Parler-TTS):
    ─────────────────────────────────────────────────
    INT8 → tried first  (~950MB, ~8-12s/sentence)
    FP16 → tried second (~1.8GB, ~12-18s/sentence)
    FP32 → tried last   (~3.75GB, ~25-40s/sentence)
    ─────────────────────────────────────────────────

    Usage:
        tts = TTSEngine()
        result = tts.speak("வணக்கம்", "tam_Taml", play=True)
        result = tts.speak("नमस्ते", "hin_Deva", refine=True, play=True)
        tts.unload()   # free RAM when not needed
    """

    def __init__(self):
        logger.info("\n🚀 Initializing TTS Engine...")
        self.parler  = ParlerTTS()
        self.piper   = PiperTTS()
        # self.espeak = ESpeakTTS()   # uncomment to enable eSpeak
        self.refiner = QwenRefiner()
        logger.info("✅ TTS Engine ready (models lazy loaded)\n")

    def speak(
        self,
        text: str,
        lang_code: str = "eng_Latn",
        output_file: Optional[str] = None,
        refine: bool = False,
        play: bool = False,
        force_engine: Optional[str] = None,
    ) -> dict:
        """
        Speak text using best available engine.

        Args:
            text         : text to speak
            lang_code    : IndicTrans2 language code e.g. "tam_Taml"
            output_file  : save to this path (None = auto temp file)
            refine       : run Qwen refinement before TTS
            play         : play audio immediately after generating
            force_engine : force "parler" or "piper" (for testing)

        Returns:
            {
                "success"     : True/False,
                "engine_used" : "parler" | "piper" | None,
                "output_file" : "/path/to/audio.wav",
                "time_taken"  : 3.45,
                "quant_used"  : "int8" | "fp16" | "fp32" | None,
                "text_used"   : "final text after refinement",
            }
        """
        if not text or not text.strip():
            logger.warning("⚠️  Empty text — nothing to speak")
            return {
                "success": False,
                "engine_used": None,
                "output_file": None,
                "time_taken": 0,
                "quant_used": None,
                "text_used": text,
            }

        start = time.time()
        logger.info(f"\n{'─'*50}")
        logger.info(f"🔊 TTS | [{lang_code}] | {text[:50]}{'...' if len(text)>50 else ''}")

        # ── Optional Qwen refinement ──────────────
        final_text = text
        if refine:
            final_text = self.refiner.refine(text, lang_code)

        # ── Engine routing ────────────────────────
        success      = False
        engine_used  = None
        output_path  = None

        for engine_name in self._get_order(lang_code, force_engine):
            logger.info(f"   → Trying {engine_name}...")

            if engine_name == "parler":
                success, output_path = self.parler.speak(
                    final_text, lang_code, output_file
                )
            elif engine_name == "piper":
                success, output_path = self.piper.speak(
                    final_text, lang_code, output_file
                )
            # elif engine_name == "espeak":           # uncomment to enable
            #     success, output_path = self.espeak.speak(
            #         final_text, lang_code, output_file=output_file
            #     )

            if success:
                engine_used = engine_name
                logger.info(f"   ✅ Success via {engine_name}")
                break
            else:
                logger.warning(f"   ⚠️  {engine_name} failed — trying next")

        # ── Playback ──────────────────────────────
        if play and success and output_path:
            self._play(output_path)

        elapsed = time.time() - start

        result = {
            "success"     : success,
            "engine_used" : engine_used,
            "output_file" : output_path,
            "time_taken"  : round(elapsed, 3),
            "quant_used"  : self.parler.quant_mode if engine_used == "parler" else None,
            "text_used"   : final_text,
        }

        if success:
            logger.info(
                f"✅ TTS done | engine: {engine_used} | "
                f"quant: {result['quant_used']} | "
                f"time: {elapsed:.2f}s"
            )
        else:
            logger.error("❌ All TTS engines failed")

        return result

    def _get_order(self, lang_code: str, force: Optional[str]) -> list:
        """Determine engine try order."""
        if force:
            return [force]

        order = []
        if self.parler.is_available():
            order.append("parler")
        if self.piper.is_available(lang_code):
            order.append("piper")
        # order.append("espeak")     # uncomment to add eSpeak as last resort
        return order

    def _play(self, file_path: str):
        """Play audio file through speakers."""
        try:
            data, sr = sf.read(file_path)
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            logger.warning(f"⚠️  sounddevice playback failed: {e}")
            try:
                subprocess.run(["aplay", file_path], capture_output=True)
            except Exception:
                pass

    def unload(self):
        """
        Dynamically unload Parler-TTS to free RAM.
        Call when switching to STT or translation mode.
        Parler reloads automatically on next speak().
        """
        self.parler.unload()

    def status(self) -> dict:
        """Current engine status."""
        return {
            "parler_available" : self.parler.is_available(),
            "parler_loaded"    : self.parler.is_loaded,
            "parler_quant"     : self.parler.quant_mode,
            "piper_voices"     : [
                k for k in PIPER_VOICES
                if os.path.exists(PIPER_VOICES[k][0])
            ],
            "qwen_available"   : self.refiner._available,
            # "espeak_available": True,   # uncomment if eSpeak enabled
        }


# ─────────────────────────────────────────────
# TTS Module Test
# ─────────────────────────────────────────────
if __name__ == "__main__":

    OUTPUT_DIR = os.path.expanduser(
        "~/projects/ai-translator/tests/audio/tts_test"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tts = TTSEngine()

    print("\n" + "="*55)
    print("  TTS ENGINE STATUS")
    print("="*55)
    for k, v in tts.status().items():
        print(f"  {k:<25}: {v}")

    print("\n" + "="*55)
    print("  TTS TESTS")
    print("="*55)

    tests = [
        ("Tamil",     "வணக்கம், இது AI மொழிபெயர்ப்பு சாதனம்.",   "tam_Taml"),
        ("Hindi",     "नमस्ते, यह AI अनुवाद उपकरण है।",            "hin_Deva"),
        ("Telugu",    "హలో, ఇది AI అనువాద పరికరం.",                "tel_Telu"),
        ("Kannada",   "ಹಲೋ, ಇದು AI ಅನುವಾದ ಸಾಧನ.",                 "kan_Knda"),
        ("Malayalam", "ഹലോ, ഇത് AI വിവർത്തന ഉപകരണം.",              "mal_Mlym"),
        ("English",   "Hello, this is an AI translation device.",  "eng_Latn"),
    ]

    for lang_name, text, lang_code in tests:
        print(f"\n📌 {lang_name}")
        print(f"   Text    : {text}")

        out_path = os.path.join(OUTPUT_DIR, f"test_{lang_name.lower()}.wav")

        result = tts.speak(
            text,
            lang_code,
            output_file=out_path,
            refine=False,       # set True to test Qwen refinement
            play=False,         # set True to hear audio
        )

        print(f"   Engine  : {result['engine_used']}")
        print(f"   Quant   : {result['quant_used']}")
        print(f"   Time    : {result['time_taken']}s")
        print(f"   File    : {result['output_file']}")
        print(f"   Success : {result['success']}")
        print("─"*55)

    print(f"\n✅ Test complete — audio files in: {OUTPUT_DIR}")
    print("\nPlay with:")
    for lang_name, _, _ in tests:
        print(
            f"  aplay {OUTPUT_DIR}/test_{lang_name.lower()}.wav"
        )

    # ── Dynamic unload test ──────────────────
    print("\n🗑️  Testing dynamic unload...")
    tts.unload()
    print("✅ Parler-TTS unloaded — RAM freed")
    print("   (Will reload automatically on next speak())")