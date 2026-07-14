"""
IndicTrans2 Translation Engine
================================
- Uses distilled 200M models (primary layer)
- langdetect for language detection (works for all Indian languages)
- Pivot routing: Indic → EN → Indic for cross-language pairs
- Lazy model loading (RAM-friendly for Raspberry Pi)

Requirements:
    pip install transformers>=4.51 torch>=2.5 numpy>=2.1
    pip install sentencepiece sacremoses langdetect
    git clone https://github.com/VarunGumma/IndicTransToolkit
    cd IndicTransToolkit && pip install --editable . --use-pep517
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor
from langdetect import detect as langdetect_detect
from langdetect import DetectorFactory

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Makes langdetect deterministic (same input = same output every time)
DetectorFactory.seed = 0

# ─────────────────────────────────────────────
# Device — CPU on Raspberry Pi
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Running on: {DEVICE}")


# ─────────────────────────────────────────────
# Language Detector
# ─────────────────────────────────────────────
class LanguageDetectorWrapper:
    """
    Detects language using langdetect.
    Supports all major Indian languages reliably.
    Maps langdetect ISO codes → IndicTrans2 language codes.
    """

    LANG_CODE_MAP = {
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
        "or": "ory_Orya",
        "as": "asm_Beng",
    }

    def detect(self, text: str) -> str:
        """
        Returns IndicTrans2 language code.
        Falls back to eng_Latn if detection fails.
        """
        try:
            code = langdetect_detect(text)
            result = self.LANG_CODE_MAP.get(code, "eng_Latn")
            print(f"🔍 Detected: {code} → {result}")
            return result
        except Exception as e:
            print(f"⚠️  Detection failed ({e}) — defaulting to eng_Latn")
            return "eng_Latn"


# ─────────────────────────────────────────────
# Core Translator
# ─────────────────────────────────────────────
class IndicTranslator:
    """
    Loads ONE IndicTrans2 distilled model for a given direction.

    direction: "en-indic"  →  English to any Indian language
               "indic-en"  →  Any Indian language to English

    Memory note:
    - Each model uses ~500MB RAM after loading
    - On Raspberry Pi 5 (8GB), both can stay loaded simultaneously
    - Use unload() if you ever need to free memory
    """

    MODEL_MAP = {
        "en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
        "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M",
    }

    def __init__(self, direction: str):
        if direction not in self.MODEL_MAP:
            raise ValueError(
                f"direction must be 'en-indic' or 'indic-en', got: '{direction}'"
            )

        self.direction = direction
        self.model_name = self.MODEL_MAP[direction]

        print(f"\n🔄 Loading model  : {self.model_name}")

        # IndicProcessor handles all Indic script preprocessing and postprocessing
        self.processor = IndicProcessor(inference=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model.to(DEVICE)
        self.model.eval()

        print(f"✅ Model ready    : {self.model_name}")

    def translate(self, sentences: list, src_lang: str, tgt_lang: str) -> list:
        """
        Translate a batch of sentences.

        Args:
            sentences : list of input strings
            src_lang  : source language code e.g. "eng_Latn"
            tgt_lang  : target language code e.g. "tam_Taml"

        Returns:
            list of translated strings (same length as input)
        """
        if not sentences:
            return []

        # Step 1: Preprocess — normalizes text, handles script issues
        batch = self.processor.preprocess_batch(
            sentences,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            visualize=False,
        )

        # Step 2: Tokenize
        inputs = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(DEVICE)

        # Step 3: Generate
        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                num_beams=5,
                num_return_sequences=1,
                max_length=256,
            )

        # Step 4: Decode
        raw_output = self.tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Step 5: Postprocess — restores correct script and punctuation
        results = self.processor.postprocess_batch(raw_output, lang=tgt_lang)

        return results

    def translate_one(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate a single string. Wrapper around translate()."""
        return self.translate([text], src_lang, tgt_lang)[0]

    def unload(self):
        """Free this model from RAM."""
        del self.model
        del self.tokenizer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"🗑️  Unloaded: {self.model_name}")


# ─────────────────────────────────────────────
# Main Translation Engine
# ─────────────────────────────────────────────
class TranslationEngine:
    """
    Smart routing engine with lazy model loading.

    Routing logic:
    ──────────────────────────────────────────
    English  → Indic  :  en-indic model (direct)
    Indic    → English:  indic-en model (direct)
    Indic    → Indic  :  pivot via English (both models)
    ──────────────────────────────────────────

    Models are loaded on first use (lazy loading).
    Both models stay cached after first load.
    """

    EN = "eng_Latn"

    def __init__(self):
        print("\n🚀 Initializing TranslationEngine...")
        self.detector = LanguageDetectorWrapper()
        self._en_indic = None   # Loaded on first use
        self._indic_en = None   # Loaded on first use
        print("✅ TranslationEngine ready\n")

    def _get_en_indic(self) -> IndicTranslator:
        """Lazy load en→indic model."""
        if self._en_indic is None:
            self._en_indic = IndicTranslator("en-indic")
        return self._en_indic

    def _get_indic_en(self) -> IndicTranslator:
        """Lazy load indic→en model."""
        if self._indic_en is None:
            self._indic_en = IndicTranslator("indic-en")
        return self._indic_en

    def translate(self, text: str, tgt_lang: str, src_lang: str = None) -> str:
        """
        Translate a single string.

        Args:
            text     : input string
            tgt_lang : target language code e.g. "tam_Taml", "hin_Deva"
            src_lang : optional — auto-detected if not provided

        Returns:
            Translated string
        """
        # Auto-detect source language
        if src_lang is None:
            src_lang = self.detector.detect(text)
        else:
            print(f"📌 Source (manual): {src_lang}")

        print(f"🎯 Target         : {tgt_lang}")

        # Same language — no translation needed
        if src_lang == tgt_lang:
            print("⚡ Same language — returning original")
            return text

        # Route 1: English → Indic
        if src_lang == self.EN:
            print("➡️  Route: EN → Indic (direct)")
            return self._get_en_indic().translate_one(text, src_lang, tgt_lang)

        # Route 2: Indic → English
        if tgt_lang == self.EN:
            print("➡️  Route: Indic → EN (direct)")
            return self._get_indic_en().translate_one(text, src_lang, tgt_lang)

        # Route 3: Indic → Indic (pivot through English)
        print("🔁 Route: Indic → EN → Indic (pivot)")
        en_text = self._get_indic_en().translate_one(text, src_lang, self.EN)
        print(f"   Pivot (EN)     : {en_text}")
        return self._get_en_indic().translate_one(en_text, self.EN, tgt_lang)

    def translate_batch(self, texts: list, tgt_lang: str, src_lang: str = None) -> list:
        """
        Translate multiple strings efficiently (for OCR / document use).
        All texts must be in the same source language.

        Args:
            texts    : list of strings
            tgt_lang : target language code
            src_lang : optional — auto-detected from first string if not provided

        Returns:
            List of translated strings
        """
        if not texts:
            return []

        if src_lang is None:
            src_lang = self.detector.detect(texts[0])
            print(f"🔍 Detected (batch): {src_lang}")

        if src_lang == tgt_lang:
            return texts

        if src_lang == self.EN:
            return self._get_en_indic().translate(texts, src_lang, tgt_lang)

        if tgt_lang == self.EN:
            return self._get_indic_en().translate(texts, src_lang, tgt_lang)

        # Pivot: Indic → EN → Indic
        en_texts = self._get_indic_en().translate(texts, src_lang, self.EN)
        return self._get_en_indic().translate(en_texts, self.EN, tgt_lang)


# ─────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────
if __name__ == "__main__":

    engine = TranslationEngine()

    tests = [
        # (input_text, target_lang, description)
        ("Hello, how are you?",               "tam_Taml", "EN → Tamil"),
        ("Hello, how are you?",               "hin_Deva", "EN → Hindi"),
        ("Hello, how are you?",               "tel_Telu", "EN → Telugu"),
        ("நீங்கள் எப்படி இருக்கிறீர்கள்?",  "hin_Deva", "Tamil → Hindi (pivot)"),
        ("நீங்கள் எப்படி இருக்கிறீர்கள்?",  "eng_Latn", "Tamil → EN"),
        ("आप कैसे हैं?",                      "tam_Taml", "Hindi → Tamil (pivot)"),
        ("आप कैसे हैं?",                      "eng_Latn", "Hindi → EN"),
    ]

    print("\n" + "="*60)
    print("  TRANSLATION TESTS")
    print("="*60)

    for text, tgt, desc in tests:
        print(f"\n📌 {desc}")
        print(f"   Input  : {text}")
        result = engine.translate(text, tgt_lang=tgt)
        print(f"   Output : {result}")
        print("-"*60)

    # Batch test — simulates OCR document translation
    print("\n📄 BATCH TEST (OCR simulation — EN → Tamil)")
    batch_input = [
        "Please enter your full name.",
        "Date of birth:",
        "Residential address:",
        "Mobile number:",
    ]
    batch_output = engine.translate_batch(
        batch_input,
        tgt_lang="tam_Taml",
        src_lang="eng_Latn",
    )
    for original, translated in zip(batch_input, batch_output):
        print(f"   {original!r}")
        print(f"   → {translated!r}\n")