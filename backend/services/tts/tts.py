"""
TTS Engine — Text to Speech
============================
- Piper TTS    : primary (Tamil, Hindi, Telugu, Bengali, English)
- eSpeak-NG    : fallback (all other Indian languages)
- Auto-routing : picks best engine per language automatically
- 100% offline : no internet needed after model download
- Product-level: error handling, stable, fast

Requirements:
    pip install piper-tts soundfile numpy sounddevice
    sudo apt install espeak-ng -y

Piper voice models (download separately):
    https://huggingface.co/rhasspy/piper-voices/tree/main
"""

import os
import subprocess
import tempfile
import time
import wave
import numpy as np
import soundfile as sf
import sounddevice as sd


# ─────────────────────────────────────────────
# Language Configuration
# ─────────────────────────────────────────────

# eSpeak-NG voice codes for Indian languages
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
    "ory_Orya": "or",
    "asm_Beng": "as",
}

# Piper voice model paths (populate after downloading models)
# Format: "indictrans_lang_code": ("path/to/model.onnx", "path/to/model.onnx.json")
# Download from: https://huggingface.co/rhasspy/piper-voices/tree/main
PIPER_VOICES = {
    # "hin_Deva": ("/path/to/hi_IN-hindi-medium.onnx", "/path/to/hi_IN-hindi-medium.onnx.json"),
    # "eng_Latn": ("/path/to/en_US-lessac-medium.onnx", "/path/to/en_US-lessac-medium.onnx.json"),
}

# Languages with confirmed good eSpeak quality
ESPEAK_GOOD_QUALITY = {
    "hin_Deva", "eng_Latn", "tam_Taml",
    "tel_Telu", "ben_Beng", "mal_Mlym"
}


# ─────────────────────────────────────────────
# eSpeak-NG Engine
# ─────────────────────────────────────────────
class ESpeakTTS:
    """
    eSpeak-NG TTS — covers ALL Indian languages.
    Installed system-wide, no model download needed.
    Voice is robotic but fast and clear.
    Latency: ~0.2-0.5 sec on Raspberry Pi 5

    FIX: Always writes to a WAV file first.
         Never calls eSpeak without -w flag.
         Direct playback mode hangs forever on headless
         machines with no ALSA/PulseAudio device.
    """

    def __init__(self):
        result = subprocess.run(
            ["espeak-ng", "--version"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "eSpeak-NG not found. Install with: sudo apt install espeak-ng -y"
            )
        print("✅ eSpeak-NG ready")

    def speak(
        self,
        text: str,
        lang_code: str = "eng_Latn",
        speed: int = 150,
        output_file: str = None,
    ) -> bool:
        """
        Speak text using eSpeak-NG.

        ALWAYS writes to a WAV file first (-w flag).
        If output_file is None, writes to a temp file,
        plays it via sounddevice, then deletes it.

        Args:
            text        : text to speak
            lang_code   : IndicTrans2 language code e.g. "tam_Taml"
            speed       : speaking speed in words per minute
            output_file : save to WAV file instead of playing

        Returns:
            True if successful, False if failed
        """
        voice = ESPEAK_VOICES.get(lang_code, "en")

        # Always write to a file — NEVER let eSpeak touch the audio
        # device directly. Direct playback hangs forever on headless
        # machines (no ALSA/PulseAudio). This was the root cause of
        # the 27-second timeout bug.
        use_temp = output_file is None
        target = output_file or tempfile.mktemp(suffix=".wav", prefix="espeak_")

        try:
            result = subprocess.run(
                [
                    "espeak-ng",
                    "-v", voice,
                    "-s", str(speed),
                    "-w", target,   # ← always write to file
                    text,
                ],
                capture_output=True,
                text=True,
                timeout=10,         # 10s is plenty for file-write mode
            )

            if result.returncode != 0:
                print(f"⚠️  eSpeak error: {result.stderr}")
                return False

            # If caller wanted playback (no output_file given),
            # play the temp wav then clean up
            if use_temp:
                self._play_wav(target)
                try:
                    os.remove(target)
                except OSError:
                    pass

            return True

        except subprocess.TimeoutExpired:
            print("⚠️  eSpeak timeout")
            return False
        except Exception as e:
            print(f"⚠️  eSpeak failed: {e}")
            return False

    def _play_wav(self, path: str):
        """
        Play a WAV file via sounddevice.
        Failure here does NOT mean TTS failed — the WAV was
        already written correctly. Playback may fail on
        headless machines with no audio output device.
        """
        try:
            data, samplerate = sf.read(path)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"⚠️  Playback failed (WAV was written OK): {e}")
            # Do not raise — file generation succeeded

    def speak_to_file(self, text: str, lang_code: str, output_path: str) -> bool:
        """Save speech to WAV file."""
        return self.speak(text, lang_code, output_file=output_path)


# ─────────────────────────────────────────────
# Piper TTS Engine
# ─────────────────────────────────────────────
class PiperTTS:
    """
    Piper TTS — natural sounding voices.
    Requires downloading voice models separately.
    Latency: ~0.5-1 sec on Raspberry Pi 5

    Download voices from:
    https://huggingface.co/rhasspy/piper-voices/tree/main

    Recommended models for Indian languages:
    - Hindi  : hi_IN-hindi-medium.onnx
    - English: en_US-lessac-medium.onnx
    """

    def __init__(self, voices_dir: str = None):
        self.voices_dir = voices_dir
        self._loaded_voices = {}
        self._available = self._check_available()

        if self._available:
            print("✅ Piper TTS ready")
            if voices_dir:
                self._scan_voices(voices_dir)
        else:
            print("⚠️  Piper TTS not available — will use eSpeak fallback")

    def _check_available(self) -> bool:
        try:
            from piper import PiperVoice
            return True
        except ImportError:
            return False

    def _scan_voices(self, voices_dir: str):
        """Scan directory for available .onnx voice files."""
        if not os.path.exists(voices_dir):
            return
        for f in os.listdir(voices_dir):
            if f.endswith(".onnx"):
                print(f"   Found voice model: {f}")

    def is_available_for(self, lang_code: str) -> bool:
        """Check if Piper has a voice for this language."""
        return lang_code in PIPER_VOICES and self._available

    def speak(
        self,
        text: str,
        lang_code: str,
        output_file: str = None,
    ) -> bool:
        """
        Speak using Piper TTS.
        Returns False if language not supported (caller falls back to eSpeak).
        """
        if not self.is_available_for(lang_code):
            return False

        try:
            from piper import PiperVoice

            model_path, config_path = PIPER_VOICES[lang_code]

            # Load voice (cached after first load)
            if lang_code not in self._loaded_voices:
                print(f"🔄 Loading Piper voice: {lang_code}")
                self._loaded_voices[lang_code] = PiperVoice.load(
                    model_path,
                    config_path=config_path,
                    use_cuda=False,
                )

            voice = self._loaded_voices[lang_code]

            if output_file:
                # Write directly to caller-specified file
                with wave.open(output_file, "wb") as wav_file:
                    voice.synthesize(text, wav_file)
            else:
                # Synthesize to buffer, play via sounddevice
                import io
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wav_file:
                    voice.synthesize(text, wav_file)
                buf.seek(0)
                data, samplerate = sf.read(buf)
                sd.play(data, samplerate)
                sd.wait()

            return True

        except Exception as e:
            print(f"⚠️  Piper failed: {e} — falling back to eSpeak")
            return False


# ─────────────────────────────────────────────
# Main TTS Engine — Auto Routing
# ─────────────────────────────────────────────
class TTSEngine:
    """
    Smart TTS engine that auto-selects the best
    available voice for each language.

    Routing priority:
    ─────────────────────────────────────────
    1. Piper TTS  — if voice model available (natural voice)
    2. eSpeak-NG  — fallback for all languages (robotic but reliable)
    ─────────────────────────────────────────

    Usage:
        tts = TTSEngine()
        tts.speak("வணக்கம்", "tam_Taml")
        tts.speak("नमस्ते", "hin_Deva")
        tts.speak("Hello", "eng_Latn")
        tts.speak("Hello", "eng_Latn", output_file="/tmp/out.wav")
    """

    def __init__(self, piper_voices_dir: str = None):
        print("\n🔊 Initializing TTS Engine...")
        self.espeak = ESpeakTTS()
        self.piper = PiperTTS(voices_dir=piper_voices_dir)
        print("✅ TTS Engine ready\n")

    def speak(
        self,
        text: str,
        lang_code: str = "eng_Latn",
        speed: int = 150,
        output_file: str = None,
        engine: str = "auto",       # "auto", "piper", "espeak"
    ) -> dict:
        """
        Speak text in the given language.

        Args:
            text        : text to speak
            lang_code   : IndicTrans2 language code
            speed       : speaking speed in WPM (eSpeak only)
            output_file : save to WAV instead of playing
            engine      : force specific engine or "auto"

        Returns:
            {
                "success"     : True/False,
                "engine_used" : "piper" or "espeak" or None,
                "lang_code"   : "tam_Taml",
                "time_taken"  : 0.45,
                "output_file" : "/tmp/out.wav" or None
            }
        """
        if not text or not text.strip():
            print("⚠️  Empty text — nothing to speak")
            return {
                "success": False,
                "engine_used": None,
                "lang_code": lang_code,
                "time_taken": 0,
                "output_file": None,
            }

        print(f"\n🔊 Speaking [{lang_code}]: {text[:50]}{'...' if len(text) > 50 else ''}")
        start = time.time()

        engine_used = None
        success = False

        # 1. Try Piper first (natural voice) if available
        if engine in ("auto", "piper") and self.piper.is_available_for(lang_code):
            success = self.piper.speak(text, lang_code, output_file)
            if success:
                engine_used = "piper"

        # 2. Fall back to eSpeak-NG
        if not success and engine in ("auto", "espeak"):
            success = self.espeak.speak(text, lang_code, speed, output_file)
            if success:
                engine_used = "espeak"

        elapsed = round(time.time() - start, 3)

        if success:
            print(f"✅ Spoken via {engine_used} in {elapsed}s")
        else:
            print(f"❌ TTS failed for lang: {lang_code}")
            print(f"   Engine: {engine_used} | Time: {elapsed}s | Success: False")

        return {
            "success": success,
            "engine_used": engine_used,
            "lang_code": lang_code,
            "time_taken": elapsed,
            "output_file": output_file,
        }

    def speak_translation_result(self, translation_result: dict) -> dict:
        """
        Convenience method — pass translation result directly.
        Works with output from TranslationEngine.translate()

        Args:
            translation_result: dict with "translation" and "tgt_lang" keys

        Returns:
            TTS result dict
        """
        text = translation_result.get("translation", "")
        lang = translation_result.get("tgt_lang", "eng_Latn")
        return self.speak(text, lang)


# ─────────────────────────────────────────────
# Full Pipeline — STT + Translation + TTS
# ─────────────────────────────────────────────
class FullPipeline:
    """
    Complete speech-to-speech translation pipeline.

    Flow:
    🎤 Mic → Whisper STT → IndicTrans2 → TTS 🔊

    Usage:
        pipeline = FullPipeline(target_lang="tam_Taml")
        pipeline.run_from_file("audio.wav")
        pipeline.run_from_mic(duration=5)
    """

    def __init__(self, target_lang: str = "tam_Taml"):
        import sys
        sys.path.insert(0, '/home/viswaa/projects/ai-translator')

        from backend.services.stt.whisper_stt import WhisperSTT
        from backend.services.translation.translator import TranslationEngine

        print("\n🚀 Loading Full Pipeline...")
        self.stt = WhisperSTT("base")
        self.translator = TranslationEngine()
        self.tts = TTSEngine()
        self.target_lang = target_lang
        print(f"✅ Full Pipeline ready — target language: {target_lang}\n")

    def run_from_file(self, audio_path: str) -> dict:
        """Run full pipeline on an audio file."""
        print(f"\n{'='*50}")
        print("  FULL PIPELINE: Speech → Translation → Speech")
        print(f"{'='*50}")

        print("\n📍 Step 1: Speech to Text")
        stt_result = self.stt.transcribe_file(audio_path)

        print("\n📍 Step 2: Translation")
        translated = self.translator.translate(
            stt_result["text"],
            tgt_lang=self.target_lang,
            src_lang=stt_result["indictrans_code"],
        )

        print("\n📍 Step 3: Text to Speech")
        tts_result = self.tts.speak(translated, self.target_lang)

        return {
            "original_text": stt_result["text"],
            "src_lang": stt_result["indictrans_code"],
            "translated_text": translated,
            "tgt_lang": self.target_lang,
            "tts_engine": tts_result["engine_used"],
            "success": tts_result["success"],
        }

    def run_from_mic(self, duration: int = 5) -> dict:
        """Record from mic, translate, speak output."""
        stt_result = self.stt.record_and_transcribe(duration=duration)

        translated = self.translator.translate(
            stt_result["text"],
            tgt_lang=self.target_lang,
            src_lang=stt_result["indictrans_code"],
        )

        tts_result = self.tts.speak(translated, self.target_lang)

        return {
            "original_text": stt_result["text"],
            "translated_text": translated,
            "tgt_lang": self.target_lang,
            "success": tts_result["success"],
        }


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────
if __name__ == "__main__":

    tts = TTSEngine()

    tests = [
        ("Hello, welcome to the AI translator.", "eng_Latn"),
        ("வணக்கம், இது AI மொழிபெயர்ப்பு சாதனம்.", "tam_Taml"),
        ("नमस्ते, यह AI अनुवाद उपकरण है।", "hin_Deva"),
        ("హలో, ఇది AI అనువాద పరికరం.", "tel_Telu"),
        ("ಹಲೋ, ಇದು AI ಅನುವಾದ ಸಾಧನ.", "kan_Knda"),
        ("ഹലോ, ഇത് AI വിവർത്തന ഉപകരണം.", "mal_Mlym"),
    ]

    print("=" * 50)
    print("  TTS ENGINE TESTS")
    print("=" * 50)

    for text, lang in tests:
        out = f"/tmp/tts_test_{lang}.wav"
        result = tts.speak(text, lang, output_file=out)
        print(f"   Engine: {result['engine_used']} | "
              f"Time: {result['time_taken']}s | "
              f"Success: {result['success']}")
        if result["success"]:
            from pathlib import Path
            size_kb = round(Path(out).stat().st_size / 1024, 1)
            print(f"   File  : {out} ({size_kb} KB)")
        print("-" * 50)
        time.sleep(0.3)

    print("\n✅ Test complete. Verify WAV files with:")
    print("   file /tmp/tts_test_*.wav")
    print("   aplay /tmp/tts_test_tam_Taml.wav   # if speakers available")