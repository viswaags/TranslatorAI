# OCR Engine — Production Ready (Final Clean Version)

from __future__ import annotations
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

    def is_valid(self):
        return bool(self.text.strip()) and self.error is None


# ─────────────────────────────────────────────
# TEXT NORMALIZATION
# ─────────────────────────────────────────────

def normalize_indic_text(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFC", text)

    if lang == "tam_Taml":
        text = text.replace("ொ", "ொ")
        text = text.replace("ோ", "ோ")
        text = text.replace("ௌ", "ௌ")

        text = re.sub(r"ெ([க-ஹ])", r"\1ெ", text)
        text = re.sub(r"ே([க-ஹ])", r"\1ே", text)
        text = re.sub(r"ை([க-ஹ])", r"\1ை", text)

        text = text.replace("ிை", "ை")

    return text


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

        if not image_path.exists():
            return OCRResult("", "", 0.0, "none", error="Image not found")

        # 1️⃣ Paddle OCR
        result = self._paddle_ocr(image_path, language_hint)

        # 2️⃣ fallback
        if not result.is_valid() or result.confidence < self.conf_threshold:
            result = self._tesseract_ocr(image_path, language_hint)

        # 3️⃣ normalization
        if result.text:
            result.text = normalize_indic_text(result.text, result.language)

        # 4️⃣ LLM correction
        if self.use_llm:
            result.text = self._llm_correct(result.text)
            result.corrected = True

        result.processing_time_ms = int((time.time() - start) * 1000)
        return result

    # ─────────────────────────────
    # PADDLE OCR
    # ─────────────────────────────

    def _load_paddle(self, lang):
        if self._paddle is None:
            from paddleocr import PaddleOCR
            logger.info(f"Loading PaddleOCR ({lang})")
            self._paddle = PaddleOCR(
                lang=lang,
                use_angle_cls=True,
                use_gpu=False,
                enable_mkldnn=True,
                cpu_threads=4
            )
        return self._paddle

    def _paddle_ocr(self, image_path, language_hint):
        try:
            lang = INDIC_TO_PADDLE.get(language_hint, "en")
            reader = self._load_paddle(lang)

            result = reader.ocr(str(image_path), cls=True)

            lines = []
            confs = []

            for line in result[0]:
                text, conf = line[1]
                lines.append(text)
                confs.append(conf)

            text = "\n".join(lines)
            confidence = sum(confs) / len(confs) if confs else 0.0

            return OCRResult(
                text=text,
                language=self._detect_lang(text) or "eng_Latn",
                confidence=confidence,
                engine_used="paddle"
            )

        except Exception as e:
            logger.warning(f"Paddle failed: {e}")
            return OCRResult("", "", 0.0, "paddle", error=str(e))

    # ─────────────────────────────
    # TESSERACT OCR
    # ─────────────────────────────

    def _tesseract_ocr(self, image_path, language_hint):
        try:
            import pytesseract
            from PIL import Image

            lang = TESSERACT_LANGS.get(language_hint, "eng+tam")
            image = Image.open(image_path)

            text = pytesseract.image_to_string(
                image,
                lang=lang,
                config="--oem 3 --psm 3"
            )

            return OCRResult(
                text=text.strip(),
                language=self._detect_lang(text) or language_hint or "eng_Latn",
                confidence=0.6,
                engine_used="tesseract"
            )

        except Exception as e:
            return OCRResult("", "", 0.0, "tesseract", error=str(e))

    # ─────────────────────────────
    # LANGUAGE DETECTION
    # ─────────────────────────────

    def _detect_lang(self, text):
        if re.search(r"[\u0B80-\u0BFF]", text):
            return "tam_Taml"
        if re.search(r"[A-Za-z]", text):
            return "eng_Latn"
        return None

    # ─────────────────────────────
    # LLM CORRECTION
    # ─────────────────────────────

    def _llm_correct(self, text):
        try:
            prompt = f"""
You are an expert Tamil text corrector.

Fix the OCR output by:
- correcting spelling mistakes
- fixing broken Tamil characters
- removing random English noise
- restoring proper Tamil words and grammar

Return ONLY the corrected Tamil text.
Do NOT explain anything.

Input:
{text}
"""

            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:1.5b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=20
            )

            return res.json().get("response", text)

        except Exception:
            return text


# ─────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--lang", default=None)
    args = parser.parse_args()

    engine = OCREngine(use_llm=True)  # 🔥 enabled
    res = engine.extract(args.image, args.lang)

    print("\n===== OCR RESULT =====")
    print("Engine:", res.engine_used)
    print("Lang:", res.language)
    print("Confidence:", res.confidence)
    print("Time:", res.processing_time_ms, "ms")
    print("\nText:\n", res.text)