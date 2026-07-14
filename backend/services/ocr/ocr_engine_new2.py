# OCR Engine — Production Ready (Updated Clean Version)

from __future__ import annotations
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import requests

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LANGUAGE CONFIG
# ─────────────────────────────────────────────

TESSERACT_LANGS = {
    "eng_Latn": "eng",
    "tam_Taml": "tam",
    "hin_Deva": "hin",
    "tel_Telu": "tel",
    "kan_Knda": "kan",
    "mal_Mlym": "mal",
}

PADDLE_TO_INDIC = {
    "en": "eng_Latn",
    "ta": "tam_Taml",
    "hi": "hin_Deva",
}

INDIC_TO_PADDLE = {v: k for k, v in PADDLE_TO_INDIC.items()}

DEFAULT_CONF = 0.6

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ─────────────────────────────────────────────
# RESULT CLASS
# ─────────────────────────────────────────────

@dataclass
class OCRResult:
    text: str
    language: str
    confidence: float
    engine_used: str
    corrected: bool = False
    processing_time_ms: int = 0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def is_valid(self):
        return bool(self.text.strip()) and self.error is None

    def summary(self) -> str:
        """Human-readable one-line summary for logging/debugging."""
        status = "✅" if self.is_valid() else "❌"
        return (
            f"{status} Engine={self.engine_used} | Lang={self.language} "
            f"| Conf={self.confidence:.2f} | Time={self.processing_time_ms}ms "
            f"| Corrected={self.corrected} | Words={len(self.text.split())}"
        )


# ─────────────────────────────────────────────
# TEXT NORMALIZATION
# ─────────────────────────────────────────────

def normalize_indic_text(text: str, lang: str) -> str:
    """Normalize Unicode and fix script-specific issues."""
    if not text:
        return text

    text = unicodedata.normalize("NFC", text)

    if lang == "tam_Taml":
        # Fix common composite vowel representations
        text = text.replace("\u0BCA\u0BBE", "\u0BCA")   # ொ
        text = text.replace("\u0BCB\u0BBE", "\u0BCB")   # ோ
        text = text.replace("\u0BCC\u0BC1", "\u0BCC")   # ௌ

        # Fix vowel sign ordering (sign before consonant → after)
        text = re.sub(r"\u0BC6([\u0B95-\u0BB9])", r"\1" + "\u0BC6", text)  # ெ
        text = re.sub(r"\u0BC7([\u0B95-\u0BB9])", r"\1" + "\u0BC7", text)  # ே
        text = re.sub(r"\u0BC8([\u0B95-\u0BB9])", r"\1" + "\u0BC8", text)  # ை

        # Remove invalid vowel combinations
        text = text.replace("\u0BBF\u0BC8", "\u0BC8")  # ிை → ை

    # Strip null bytes and control characters (common in bad OCR output)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Collapse excessive blank lines (more than 2 in a row → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_english_noise(text: str) -> str:
    """Remove obvious OCR noise: lone symbols, random single chars, etc."""
    # Remove lines that are just punctuation/symbols with no letters
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped and re.search(r"[A-Za-z\u0B80-\u0BFF\u0900-\u097F]", stripped):
            cleaned.append(line)
        elif stripped and len(stripped) > 3:
            cleaned.append(line)
    return "\n".join(cleaned)


# ─────────────────────────────────────────────
# OCR ENGINE
# ─────────────────────────────────────────────

class OCREngine:

    def __init__(self, confidence_threshold=DEFAULT_CONF, use_llm=False):
        self.conf_threshold = confidence_threshold
        self.use_llm = use_llm
        self._paddle = None

    # ─────────────────────────────

    def extract(self, image_path, language_hint=None) -> OCRResult:
        start = time.time()
        image_path = Path(image_path)

        # ── Input validation ──
        if not image_path.exists():
            return OCRResult("", "", 0.0, "none", error=f"Image not found: {image_path}")

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return OCRResult(
                "", "", 0.0, "none",
                error=f"Unsupported file type: {image_path.suffix}. Supported: {SUPPORTED_EXTENSIONS}"
            )

        warnings = []

        # 1️⃣ PaddleOCR (primary)
        result = self._paddle_ocr(image_path, language_hint)

        # 2️⃣ Tesseract (fallback if Paddle failed or low confidence)
        if not result.is_valid() or result.confidence < self.conf_threshold:
            if result.is_valid():
                warnings.append(
                    f"Paddle confidence {result.confidence:.2f} below threshold "
                    f"{self.conf_threshold:.2f}, falling back to Tesseract."
                )
            fallback = self._tesseract_ocr(image_path, language_hint)
            if fallback.is_valid():
                result = fallback
            elif result.is_valid():
                warnings.append("Tesseract also failed; keeping Paddle result.")

        # 3️⃣ Noise cleanup (before normalization)
        if result.text:
            result.text = clean_english_noise(result.text)

        # 4️⃣ Unicode normalization
        if result.text:
            result.text = normalize_indic_text(result.text, result.language)

        # 5️⃣ LLM correction (optional)
        if self.use_llm and result.is_valid():
            corrected = self._llm_correct(result.text, result.language)
            if corrected and corrected != result.text:
                result.text = corrected
                result.corrected = True
            else:
                warnings.append("LLM correction returned no change or failed.")

        result.warnings = warnings
        result.processing_time_ms = int((time.time() - start) * 1000)
        logger.info(result.summary())
        return result

    # ─────────────────────────────
    # PADDLE OCR
    # ─────────────────────────────

    def _load_paddle(self, lang):
        """Lazy-load PaddleOCR; reuse if already loaded for same lang."""
        if self._paddle is None:
            from paddleocr import PaddleOCR
            logger.info(f"Loading PaddleOCR (lang={lang})")
            self._paddle = PaddleOCR(
                lang=lang,
                use_angle_cls=True,
                use_gpu=False,
                enable_mkldnn=True,
                cpu_threads=4,
                show_log=False,        # suppress verbose Paddle logs
            )
        return self._paddle

    def _paddle_ocr(self, image_path: Path, language_hint: Optional[str]) -> OCRResult:
        try:
            lang = INDIC_TO_PADDLE.get(language_hint, "en")
            reader = self._load_paddle(lang)

            result = reader.ocr(str(image_path), cls=True)

            if not result or not result[0]:
                return OCRResult("", "", 0.0, "paddle", error="No text detected")

            lines, confs = [], []
            for line in result[0]:
                if line and len(line) >= 2:
                    text, conf = line[1]
                    if text.strip():
                        lines.append(text.strip())
                        confs.append(conf)

            if not lines:
                return OCRResult("", "", 0.0, "paddle", error="Empty OCR result")

            text = "\n".join(lines)
            confidence = sum(confs) / len(confs)

            return OCRResult(
                text=text,
                language=self._detect_lang(text) or "eng_Latn",
                confidence=confidence,
                engine_used="paddle",
            )

        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")
            return OCRResult("", "", 0.0, "paddle", error=str(e))

    # ─────────────────────────────
    # TESSERACT OCR
    # ─────────────────────────────

    def _tesseract_ocr(self, image_path: Path, language_hint: Optional[str]) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image

            lang = TESSERACT_LANGS.get(language_hint, "eng+tam")
            image = Image.open(image_path)

            # Tesseract data gives per-word confidence
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                config="--oem 3 --psm 3",
                output_type=pytesseract.Output.DICT,
            )

            words, confs = [], []
            for i, word in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if conf > 0 and word.strip():
                    words.append(word.strip())
                    confs.append(conf / 100.0)

            text = " ".join(words)
            confidence = sum(confs) / len(confs) if confs else 0.0

            detected_lang = self._detect_lang(text) or language_hint or "eng_Latn"

            return OCRResult(
                text=text.strip(),
                language=detected_lang,
                confidence=confidence,
                engine_used="tesseract",
            )

        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
            return OCRResult("", "", 0.0, "tesseract", error=str(e))

    # ─────────────────────────────
    # LANGUAGE DETECTION
    # ─────────────────────────────

    def _detect_lang(self, text: str) -> Optional[str]:
        """Detect script from Unicode ranges."""
        if not text:
            return None
        if re.search(r"[\u0B80-\u0BFF]", text):
            return "tam_Taml"
        if re.search(r"[\u0900-\u097F]", text):
            return "hin_Deva"
        if re.search(r"[\u0C00-\u0C7F]", text):
            return "tel_Telu"
        if re.search(r"[\u0C80-\u0CFF]", text):
            return "kan_Knda"
        if re.search(r"[\u0D00-\u0D7F]", text):
            return "mal_Mlym"
        if re.search(r"[A-Za-z]", text):
            return "eng_Latn"
        return None

    # ─────────────────────────────
    # LLM CORRECTION (Ollama)
    # ─────────────────────────────



    def _llm_correct(self, text: str, lang: str = "tam_Taml") -> str:
        """Send OCR text to local Ollama model for correction."""
        try:
            # Minimal prompt — small models choke on long instructions
            if lang == "tam_Taml":
                prompt = (
                    "Fix Tamil OCR errors. Remove English noise. Keep correct Tamil words. "
                    "Output corrected Tamil text only, no explanation.\n\n"
                    "Input:\n" + text + "\n\nOutput:"
                )
            else:
                prompt = (
                    "Fix OCR errors. Remove noise. Keep correct words. "
                    "Output corrected text only, no explanation.\n\n"
                    "Input:\n" + text + "\n\nOutput:"
                )

            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:1.5b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 200,   # cap output tokens
                        "num_ctx": 512,       # smaller context window = much faster
                    },
                },
                timeout=60,   # 1.5B on CPU needs time; give it room
            )
            res.raise_for_status()
            corrected = res.json().get("response", "").strip()

            if corrected and len(corrected) >= len(text) * 0.4:
                logger.info("LLM correction applied (%d -> %d chars)", len(text), len(corrected))
                return corrected
            else:
                logger.warning("LLM output too short -- keeping original.")
                return text

        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not running -- skipping LLM correction.")
            return text
        except requests.exceptions.Timeout:
            logger.warning("Ollama timed out -- skipping LLM correction. Try reducing image text or use a faster model.")
            return text
        except Exception as e:
            logger.warning("LLM correction failed: %s", e)
            return text

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run OCR on an image file.")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--lang", default=None, help="Language hint (e.g. tam_Taml, eng_Latn)")
    parser.add_argument("--llm", action="store_true", help="Enable LLM correction via Ollama")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    args = parser.parse_args()

    engine = OCREngine(confidence_threshold=args.conf, use_llm=args.llm)
    res = engine.extract(args.image, args.lang)

    print("\n===== OCR RESULT =====")
    print(res.summary())
    if res.warnings:
        print("\nWarnings:")
        for w in res.warnings:
            print(f"  ⚠️  {w}")
    print("\nText:\n", res.text)
    if res.error:
        print("\nError:", res.error)