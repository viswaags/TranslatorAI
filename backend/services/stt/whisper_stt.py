"""
Whisper STT Engine
===================
- Uses faster-whisper (4x faster than openai-whisper on CPU)
- Supports live microphone recording
- Auto-detects spoken language
- Returns text + detected language code
- Feeds directly into TranslationEngine

Requirements:
    pip install faster-whisper sounddevice soundfile numpy
"""

import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


# ─────────────────────────────────────────────
# Language code mapping
# Whisper → IndicTrans2 language codes
# ─────────────────────────────────────────────
WHISPER_TO_INDICTRANS = {
    "en": "eng_Latn",
    "ta": "tam_Taml",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "mr": "mar_Deva",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
}


# ─────────────────────────────────────────────
# STT Engine
# ─────────────────────────────────────────────
class WhisperSTT:
    """
    Speech-to-Text engine using faster-whisper.

    Model sizes vs performance on Raspberry Pi 5 (CPU):
    ─────────────────────────────────────────────────────
    tiny   → ~40MB  | ~1-2 sec latency | lower accuracy
    base   → ~75MB  | ~2-3 sec latency | good accuracy   ← RECOMMENDED
    small  → ~250MB | ~5-8 sec latency | better accuracy
    medium → ~800MB | too slow for Pi  | NOT recommended
    ─────────────────────────────────────────────────────
    """

    def __init__(self, model_size: str = "base"):
        print(f"🔄 Loading Whisper model: {model_size}")
        # cpu + int8 = fastest on Raspberry Pi (no GPU)
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",        # INT8 quantization — faster on CPU
        )
        print(f"✅ Whisper ready: {model_size}")

    def transcribe_file(self, audio_path: str) -> dict:
        """
        Transcribe audio from a file.

        Args:
            audio_path: Path to .wav or .mp3 file

        Returns:
            {
                "text": "transcribed text",
                "language": "ta",               # Whisper language code
                "indictrans_code": "tam_Taml",  # IndicTrans2 code
                "confidence": 0.95              # language detection confidence
            }
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"📄 Transcribing file: {audio_path}")
        start = time.time()

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            best_of=5,
            language=None,          # None = auto-detect language
            task="transcribe",      # transcribe = keep original language
                                    # translate = force output to English
            vad_filter=True,        # removes silence automatically
            vad_parameters={
                "min_silence_duration_ms": 500,
            },
        )

        # Collect all segments into full text
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed = time.time() - start

        lang_code = info.language
        indictrans_code = WHISPER_TO_INDICTRANS.get(lang_code, "eng_Latn")

        print(f"🔍 Detected language : {lang_code} → {indictrans_code}")
        print(f"📝 Transcribed text  : {text}")
        print(f"⏱️  Time taken        : {elapsed:.2f}s")

        return {
            "text": text,
            "language": lang_code,
            "indictrans_code": indictrans_code,
            "confidence": round(info.language_probability, 3),
        }

    def record_and_transcribe(
        self,
        duration: int = 5,
        sample_rate: int = 16000,
        temp_path: str = "/tmp/recorded_audio.wav",
    ) -> dict:
        """
        Record from microphone and transcribe.

        Args:
            duration    : recording duration in seconds (default 5)
            sample_rate : 16000 Hz is optimal for Whisper
            temp_path   : temporary file path for recording

        Returns:
            Same dict as transcribe_file()
        """
        print(f"\n🎤 Recording for {duration} seconds...")
        print("   Speak now ▶")

        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,             # mono — sufficient for speech
            dtype=np.float32,
        )
        sd.wait()                   # wait until recording is complete
        print("   Recording done ⏹")

        # Save to temp file
        sf.write(temp_path, audio_data, sample_rate)

        # Transcribe
        return self.transcribe_file(temp_path)

    def record_until_silence(
        self,
        sample_rate: int = 16000,
        silence_threshold: float = 0.01,
        min_duration: int = 1,
        max_duration: int = 10,
        temp_path: str = "/tmp/recorded_audio.wav",
    ) -> dict:
        """
        Record until user stops speaking (silence detection).
        Better UX than fixed duration for real product use.

        Args:
            silence_threshold : volume below this = silence (0.0 to 1.0)
            min_duration      : minimum recording seconds
            max_duration      : maximum recording seconds (safety cap)

        Returns:
            Same dict as transcribe_file()
        """
        print(f"\n🎤 Listening... (speak now, auto-stops on silence)")
        print(f"   Max duration: {max_duration}s")

        frames = []
        silent_frames = 0
        speaking = False
        chunk_size = int(sample_rate * 0.1)     # 100ms chunks
        silence_chunks_needed = 15              # 1.5 sec silence to stop

        def callback(indata, frame_count, time_info, status):
            nonlocal silent_frames, speaking
            volume = float(np.sqrt(np.mean(indata ** 2)))
            frames.append(indata.copy())

            if volume > silence_threshold:
                speaking = True
                silent_frames = 0
            elif speaking:
                silent_frames += 1

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_size,
            callback=callback,
        ):
            start = time.time()
            while True:
                elapsed = time.time() - start
                if elapsed >= max_duration:
                    print("   ⏱️  Max duration reached")
                    break
                if speaking and silent_frames >= silence_chunks_needed:
                    if elapsed >= min_duration:
                        print("   🔇 Silence detected — stopping")
                        break
                time.sleep(0.1)

        if not frames:
            return {"text": "", "language": "en", "indictrans_code": "eng_Latn", "confidence": 0.0}

        audio_data = np.concatenate(frames, axis=0)
        sf.write(temp_path, audio_data, sample_rate)
        return self.transcribe_file(temp_path)


# ─────────────────────────────────────────────
# Full Pipeline — STT + Translation
# ─────────────────────────────────────────────
class SpeechTranslationPipeline:
    """
    Connects WhisperSTT → TranslationEngine.

    Usage:
        pipeline = SpeechTranslationPipeline(target_lang="tam_Taml")
        result = pipeline.run_from_mic(duration=5)
        print(result["translation"])
    """

    def __init__(self, target_lang: str = "eng_Latn", whisper_model: str = "base"):
        from backend.services.translation.translator import TranslationEngine  # your Phase 1 file

        self.stt = WhisperSTT(whisper_model)
        self.translator = TranslationEngine()
        self.target_lang = target_lang
        print(f"\n✅ Pipeline ready — translating to: {target_lang}")

    def run_from_file(self, audio_path: str) -> dict:
        """Transcribe file and translate."""
        stt_result = self.stt.transcribe_file(audio_path)
        return self._translate(stt_result)

    def run_from_mic(self, duration: int = 5) -> dict:
        """Record from mic and translate."""
        stt_result = self.stt.record_and_transcribe(duration=duration)
        return self._translate(stt_result)

    def run_from_mic_auto(self) -> dict:
        """Record with auto silence detection and translate."""
        stt_result = self.stt.record_until_silence()
        return self._translate(stt_result)

    def _translate(self, stt_result: dict) -> dict:
        """Internal: translate STT result."""
        text = stt_result["text"]
        src_lang = stt_result["indictrans_code"]

        if not text:
            print("⚠️  No speech detected")
            return {**stt_result, "translation": ""}

        print(f"\n🔁 Translating: {src_lang} → {self.target_lang}")
        translation = self.translator.translate(
            text,
            tgt_lang=self.target_lang,
            src_lang=src_lang,
        )
        print(f"✅ Translation: {translation}")

        return {
            "original_text": text,
            "src_lang": src_lang,
            "tgt_lang": self.target_lang,
            "translation": translation,
            "confidence": stt_result["confidence"],
        }


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    stt = WhisperSTT(model_size="base")

    if len(sys.argv) > 1:
        # Usage: python stt.py path/to/audio.wav
        audio_file = sys.argv[1]
        result = stt.transcribe_file(audio_file)
        print(f"\n✅ Result:")
        print(f"   Text     : {result['text']}")
        print(f"   Language : {result['language']} → {result['indictrans_code']}")
        print(f"   Confidence: {result['confidence']}")

    else:
        # Live mic test — records 5 seconds
        print("\n🎤 No file provided — testing live microphone (5 seconds)")
        result = stt.record_and_transcribe(duration=5)
        print(f"\n✅ Result:")
        print(f"   Text     : {result['text']}")
        print(f"   Language : {result['language']} → {result['indictrans_code']}")
        print(f"   Confidence: {result['confidence']}")