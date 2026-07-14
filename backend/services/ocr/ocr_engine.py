"""
OCR Engine — Phase 4
====================
Offline multilingual OCR for Raspberry Pi 5 (CPU only, Python 3.10)

Pipeline:
    Layer 1  →  PaddleOCR           (primary, confidence-aware)
    Layer 2  →  Tesseract           (fallback if PaddleOCR fails / low confidence)
    Layer 3  →  qwen2.5:1.5b/Ollama (LLM correction — fixes garbled chars, spelling)
    Layer 4  →  IndicTrans2         (translation — handled by existing TranslationEngine)

Returns:
    OCRResult(text, language, confidence, engine_used)
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

# ── Environment flag required by PaddleOCR ────────────────────────────────────
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logger = logging.getLogger(__name__)

# ── Language mapping tables ────────────────────────────────────────────────────

# IndicTrans2 format  →  Tesseract lang code
TESSERACT_LANGS: dict[str, str] = {
    "eng_Latn": "eng",
    "tam_Taml": "tam",
    "hin_Deva": "hin",
    "tel_Telu": "tel",
    "kan_Knda": "kan",
    "mal_Mlym": "mal",
    "ben_Beng": "ben",
    "guj_Gujr": "guj",
    "mar_Deva": "mar",
    "pan_Guru": "pan",
    "ory_Orya": "ori",
}

# PaddleOCR language slug  →  IndicTrans2 format
PADDLE_TO_INDICTRANS: dict[str, str] = {
    "en":    "eng_Latn",
    "ta":    "tam_Taml",
    "hi":    "hin_Deva",
    "te":    "tel_Telu",
    "kn":    "kan_Knda",
    "ml":    "mal_Mlym",
    "bn":    "ben_Beng",
    "gu":    "guj_Gujr",
    "mr":    "mar_Deva",
    "pa":    "pan_Guru",
}

# PaddleOCR uses 2-letter codes for its lang= parameter
INDICTRANS_TO_PADDLE: dict[str, str] = {v: k for k, v in PADDLE_TO_INDICTRANS.items()}

# Devanagari-script languages share the same PaddleOCR model ("hi")
DEVANAGARI_LANGS: frozenset[str] = frozenset({"hin_Deva", "mar_Deva"})

# Languages that have Tesseract pack installed
TESSERACT_SUPPORTED: frozenset[str] = frozenset(TESSERACT_LANGS.keys())

# Default confidence threshold — below this, Tesseract fallback is triggered
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.60

# Ollama endpoint
OLLAMA_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "qwen2.5:1.5b"
OLLAMA_TIMEOUT: int = 60  # seconds — CPU inference is slow


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class OCRResult:
    """Returned by OCREngine.extract()"""
    text: str
    language: str                   # IndicTrans2 format, e.g. "tam_Taml"
    confidence: float               # 0.0–1.0
    engine_used: str                # "paddle" | "tesseract" | "paddle+tesseract"
    corrected: bool = False         # True if LLM correction was applied
    raw_text: str = ""              # pre-correction text (for debugging)
    processing_time_ms: int = 0
    error: Optional[str] = None

    def is_valid(self) -> bool:
        return bool(self.text.strip()) and self.error is None


# ── OCR Engine ─────────────────────────────────────────────────────────────────

class OCREngine:
    """
    Production OCR engine with layered fallback and LLM correction.

    Usage (lazy loading — models loaded on first call):
        engine = OCREngine()
        result = engine.extract("path/to/image.jpg")
        print(result.text, result.language)

    Usage (pre-load for a known language):
        engine = OCREngine()
        engine.warm_up("tam_Taml")
        result = engine.extract("image.jpg", language_hint="tam_Taml")
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        use_llm_correction: bool = True,
        ollama_url: str = OLLAMA_URL,
        ollama_model: str = OLLAMA_MODEL,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.use_llm_correction = use_llm_correction
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

        # Lazy-loaded — None until first use
        self._paddle_readers: dict[str, object] = {}   # lang_slug → PaddleOCR instance
        self._paddle_available: Optional[bool] = None  # None = not yet checked
        self._tesseract_available: Optional[bool] = None

        logger.info(
            "OCREngine initialised | confidence_threshold=%.2f | llm_correction=%s",
            confidence_threshold,
            use_llm_correction,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(
        self,
        image_path: str | Path,
        language_hint: Optional[str] = None,  # IndicTrans2 format, e.g. "tam_Taml"
    ) -> OCRResult:
        """
        Main entry point. Runs the full OCR pipeline on an image file.

        Args:
            image_path: Path to image (JPEG, PNG, BMP, TIFF supported).
            language_hint: Optional IndicTrans2 language code.
                           If None, PaddleOCR auto-detects.

        Returns:
            OCRResult with .text and .language in IndicTrans2 format.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error("Image not found: %s", image_path)
            return self._error_result(f"Image not found: {image_path}")

        t_start = time.monotonic()
        logger.info("OCR start | image=%s | hint=%s", image_path.name, language_hint)

        result = self._run_pipeline(image_path, language_hint)

        result.processing_time_ms = int((time.monotonic() - t_start) * 1000)
        logger.info(
            "OCR done  | engine=%s | lang=%s | conf=%.2f | corrected=%s | %dms",
            result.engine_used,
            result.language,
            result.confidence,
            result.corrected,
            result.processing_time_ms,
        )
        return result

    def warm_up(self, language: Optional[str] = None) -> None:
        """
        Pre-load models to avoid first-call delay.
        language: IndicTrans2 format, or None to load the multilingual PaddleOCR model.
        """
        logger.info("Warming up OCR models (lang=%s)…", language or "auto")
        paddle_lang = self._indictrans_to_paddle(language) if language else "en"
        self._get_paddle_reader(paddle_lang)
        logger.info("Warm-up complete.")

    def unload(self) -> None:
        """Release all loaded models to free RAM."""
        self._paddle_readers.clear()
        logger.info("OCR models unloaded.")

    # ── Internal pipeline ──────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        image_path: Path,
        language_hint: Optional[str],
    ) -> OCRResult:

        # --- Layer 1: PaddleOCR ---
        paddle_result = self._try_paddle(image_path, language_hint)

        if paddle_result.is_valid() and paddle_result.confidence >= self.confidence_threshold:
            logger.debug("PaddleOCR accepted (conf=%.2f)", paddle_result.confidence)
            return self._maybe_correct(paddle_result)

        # --- Layer 2: Tesseract fallback ---
        reason = (
            "low confidence (%.2f)" % paddle_result.confidence
            if paddle_result.is_valid()
            else "paddle failed"
        )
        logger.info("Falling back to Tesseract (%s)", reason)

        # Prefer detected language from Paddle even when falling back
        tess_lang_hint = paddle_result.language if paddle_result.language else language_hint
        tess_result = self._try_tesseract(image_path, tess_lang_hint)

        if tess_result.is_valid():
            # Combine: keep Paddle's language detection if Tesseract lacks it
            if not tess_result.language and paddle_result.language:
                tess_result.language = paddle_result.language
            # Mark both engines were used when Paddle gave partial results
            if paddle_result.is_valid():
                tess_result.engine_used = "paddle+tesseract"
            return self._maybe_correct(tess_result)

        # --- Both failed: return best available or error ---
        if paddle_result.is_valid():
            logger.warning("Tesseract failed — using low-confidence Paddle result")
            return self._maybe_correct(paddle_result)

        logger.error("All OCR engines failed for %s", image_path.name)
        return self._error_result("All OCR engines failed", language=language_hint or "")

    # ── Layer 1: PaddleOCR ─────────────────────────────────────────────────────

    def _try_paddle(
        self,
        image_path: Path,
        language_hint: Optional[str],
    ) -> OCRResult:
        if not self._is_paddle_available():
            return self._error_result("PaddleOCR not available")

        paddle_lang = self._indictrans_to_paddle(language_hint) if language_hint else "en"

        try:
            reader = self._get_paddle_reader(paddle_lang)

            # PaddleOCR 3.x deprecated .ocr() in favour of .predict().
            # We try predict() first; fall back to ocr() for 2.x installs.
            if hasattr(reader, "predict"):
                raw_results = reader.predict(str(image_path))
            else:
                raw_results = reader.ocr(str(image_path), cls=True)  # 2.x

            if not raw_results:
                logger.debug("PaddleOCR returned empty result")
                return self._error_result("PaddleOCR: no text detected")

            lines, confidences = self._parse_paddle_result(raw_results)
            if not lines:
                return self._error_result("PaddleOCR: could not parse result")

            text = "\n".join(lines)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            detected_lang = self._detect_language_from_text(text) or (
                PADDLE_TO_INDICTRANS.get(paddle_lang, "eng_Latn")
            )

            logger.debug(
                "PaddleOCR | lines=%d | conf=%.2f | lang=%s",
                len(lines), avg_conf, detected_lang,
            )
            return OCRResult(
                text=text,
                language=detected_lang,
                confidence=avg_conf,
                engine_used="paddle",
                raw_text=text,
            )

        except Exception as exc:
            logger.warning("PaddleOCR exception: %s", exc, exc_info=True)
            return self._error_result(f"PaddleOCR error: {exc}")

    def _parse_paddle_result(
        self, raw_results: list
    ) -> tuple[list[str], list[float]]:
        """
        Parse PaddleOCR output — handles both result formats:

        PaddleOCR 2.x (.ocr()):
            [ [ [bbox, (text, conf)], ... ] ]   ← list of pages, each page is list of items

        PaddleOCR 3.x (.predict()):
            Returns a generator/list of OCRResult objects, each with:
                .boxes  — bounding boxes
                .rec_texts  — list of strings
                .rec_scores — list of floats
            OR a flat dict-like object with 'rec_texts' / 'rec_scores' keys.
        """
        lines: list[str] = []
        confidences: list[float] = []

        for item in raw_results:
            # ── 3.x object with attributes ──────────────────────────────────
            if hasattr(item, "rec_texts") and hasattr(item, "rec_scores"):
                for text, score in zip(item.rec_texts, item.rec_scores):
                    if str(text).strip():
                        lines.append(str(text).strip())
                        confidences.append(float(score))
                continue

            # ── 3.x dict-like ────────────────────────────────────────────────
            if isinstance(item, dict) and "rec_texts" in item:
                for text, score in zip(item["rec_texts"], item.get("rec_scores", [])):
                    if str(text).strip():
                        lines.append(str(text).strip())
                        confidences.append(float(score) if score is not None else 0.5)
                continue

            # ── 2.x page (list of [bbox, (text, conf)]) ──────────────────────
            if isinstance(item, list):
                for entry in item:
                    try:
                        text_part = entry[1]
                        text = text_part[0] if isinstance(text_part, (list, tuple)) else str(text_part)
                        conf = float(text_part[1]) if isinstance(text_part, (list, tuple)) and len(text_part) > 1 else 0.5
                        if text.strip():
                            lines.append(text.strip())
                            confidences.append(conf)
                    except (IndexError, TypeError, ValueError) as exc:
                        logger.debug("Could not parse 2.x paddle entry %s: %s", entry, exc)
                continue

            logger.debug("Unrecognised PaddleOCR result item type: %s", type(item))

        return lines, confidences

    def _get_paddle_reader(self, lang_slug: str):
        """Return a cached (or newly created) PaddleOCR instance."""
        if lang_slug not in self._paddle_readers:
            from paddleocr import PaddleOCR  # lazy import
            import paddleocr as _poc
            logger.info("Loading PaddleOCR model (lang=%s)…", lang_slug)

            # PaddleOCR 3.x stripped many params (use_gpu, use_angle_cls, show_log).
            # Probe the real signature at runtime — works on 2.x and 3.x without pinning.
            import inspect
            _sig = inspect.signature(PaddleOCR.__init__).parameters
            kwargs: dict = {"lang": lang_slug}
            if "show_log" in _sig:
                kwargs["show_log"] = False              # 2.x only
            if "use_angle_cls" in _sig:
                kwargs["use_angle_cls"] = True          # 2.x param name
            elif "use_textline_orientation" in _sig:
                kwargs["use_textline_orientation"] = True  # 3.x replacement
            if "use_gpu" in _sig:
                kwargs["use_gpu"] = False               # 2.x only — 3.x auto-detects

            self._paddle_readers[lang_slug] = PaddleOCR(**kwargs)
            logger.info("PaddleOCR model loaded (lang=%s)", lang_slug)
        return self._paddle_readers[lang_slug]

    def _is_paddle_available(self) -> bool:
        if self._paddle_available is None:
            try:
                import paddleocr  # noqa: F401
                self._paddle_available = True
            except ImportError:
                logger.warning("PaddleOCR not installed — Layer 1 disabled")
                self._paddle_available = False
        return self._paddle_available

    # ── Layer 2: Tesseract ─────────────────────────────────────────────────────

    def _try_tesseract(
        self,
        image_path: Path,
        language_hint: Optional[str],
    ) -> OCRResult:
        if not self._is_tesseract_available():
            return self._error_result("Tesseract not available")

        # Step 1: resolve lang BEFORE running OCR.
        # If no hint, try to detect script from the image pixels first —
        # running Tesseract with the wrong lang pack produces garbage.
        resolved_hint = language_hint or self._detect_script_from_image(image_path)
        tess_lang = self._resolve_tesseract_lang(resolved_hint)
        logger.debug("Tesseract | lang=%s (hint=%s)", tess_lang, resolved_hint)

        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            # Tesseract config: OEM 3 (LSTM), PSM 3 (auto page segmentation)
            config = "--oem 3 --psm 3"
            raw_text = pytesseract.image_to_string(image, lang=tess_lang, config=config)

            if not raw_text.strip():
                logger.debug("Tesseract returned empty text")
                return self._error_result("Tesseract: no text detected")

            # pytesseract.image_to_data gives per-word confidence
            avg_conf = self._get_tesseract_confidence(image, tess_lang, config)
            # Language is whichever was resolved before OCR (already confirmed by script detection)
            _tess_to_indic = {v: k for k, v in TESSERACT_LANGS.items()}
            detected_lang = (
                self._detect_language_from_text(raw_text)
                or _tess_to_indic.get(tess_lang)
                or resolved_hint
                or "eng_Latn"
            )

            logger.debug(
                "Tesseract | chars=%d | conf=%.2f | lang=%s",
                len(raw_text), avg_conf, detected_lang,
            )
            return OCRResult(
                text=raw_text.strip(),
                language=detected_lang,
                confidence=avg_conf,
                engine_used="tesseract",
                raw_text=raw_text.strip(),
            )

        except Exception as exc:
            logger.warning("Tesseract exception: %s", exc, exc_info=True)
            return self._error_result(f"Tesseract error: {exc}")

    def _get_tesseract_confidence(self, image, lang: str, config: str) -> float:
        """Compute average word-level confidence from Tesseract data output."""
        try:
            import pytesseract
            data = pytesseract.image_to_data(
                image, lang=lang, config=config,
                output_type=pytesseract.Output.DICT,
            )
            confs = [
                c / 100.0
                for c in data.get("conf", [])
                if isinstance(c, (int, float)) and c >= 0
            ]
            return sum(confs) / len(confs) if confs else 0.5
        except Exception:
            return 0.5  # neutral default

    def _detect_script_from_image(self, image_path: Path) -> Optional[str]:
        """
        Pre-OCR script detection via a two-pass strategy:

        Pass 1 — Quick Tesseract eng run (~0.3s):
            Extract raw bytes with the English model (fast, no language switching).
            Run Unicode-range analysis on whatever bytes come back.
            For Indic scripts, Tesseract eng will output garbled chars that still
            fall within the correct Unicode block — enough to identify the script.

        Pass 2 — Pixel colour probe (fallback, no Tesseract):
            If Pass 1 yields no signal, sample the image for characteristic
            Indic script strokes using basic PIL analysis.

        Why NOT use Tesseract OSD (--psm 0)?
            OSD is trained primarily on Latin/CJK. For Indic scripts it commonly
            misidentifies Tamil as Arabic, Telugu as Cyrillic, etc.
            The Unicode-probe approach is both faster and more accurate here.

        Returns IndicTrans2 format string, or None if uncertain.
        """
        # ── Pass 1: quick eng OCR → Unicode probe ───────────────────────────
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            # Use eng model — fast and always available; we only care about
            # the Unicode codepoints in the output, not readable text.
            raw = pytesseract.image_to_string(img, lang="eng", config="--oem 3 --psm 3")
            detected = self._detect_language_from_text(raw)
            if detected and detected != "eng_Latn":
                logger.debug("Script probe (eng-pass): detected %s", detected)
                return detected
            # If unicode probe says Latin/nothing, check if text is mostly garbage
            # (high ratio of non-ASCII non-space) — means Indic content was present
            # but eng model couldn't decode it properly.
            non_ascii = sum(1 for c in raw if ord(c) > 127)
            total_chars = len(raw.replace(" ", "").replace("\n", ""))
            if total_chars > 10 and non_ascii / total_chars > 0.4:
                logger.debug("Script probe: high non-ASCII ratio — likely Indic, unknown script")
                # Can't determine which Indic — let Tesseract try all installed packs
                return self._probe_indic_packs(image_path)
        except Exception as exc:
            logger.debug("Script probe pass-1 failed (non-fatal): %s", exc)

        return None

    def _probe_indic_packs(self, image_path: Path) -> Optional[str]:
        """
        Try each installed Indic Tesseract language pack and return the one
        that gives the highest confidence score. Called only when script is
        clearly non-Latin but unidentifiable from Unicode ranges.
        Runs at most len(TESSERACT_LANGS)-1 Tesseract calls.
        """
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        best_lang: Optional[str] = None
        best_conf: float = 0.0
        indic_langs = {k: v for k, v in TESSERACT_LANGS.items() if k != "eng_Latn"}
        for indictrans_code, tess_code in indic_langs.items():
            try:
                data = pytesseract.image_to_data(
                    img, lang=tess_code, config="--oem 3 --psm 3",
                    output_type=pytesseract.Output.DICT,
                )
                confs = [c / 100.0 for c in data.get("conf", []) if isinstance(c, (int, float)) and c >= 0]
                avg = sum(confs) / len(confs) if confs else 0.0
                logger.debug("Indic probe: %s → conf=%.2f", tess_code, avg)
                if avg > best_conf:
                    best_conf = avg
                    best_lang = indictrans_code
            except Exception:
                pass
        if best_lang and best_conf > 0.3:
            logger.debug("Indic probe winner: %s (conf=%.2f)", best_lang, best_conf)
            return best_lang
        return None

    def _resolve_tesseract_lang(self, language_hint: Optional[str]) -> str:
        """
        Map an IndicTrans2 code to a Tesseract lang string.
        Falls back to "eng" if the language is not in TESSERACT_LANGS.
        """
        if not language_hint:
            return "eng"
        # Try exact match first
        if language_hint in TESSERACT_LANGS:
            return TESSERACT_LANGS[language_hint]
        # Try script-level fallback (e.g. "hin_Deva" → "hin")
        prefix = language_hint.split("_")[0]
        for key, val in TESSERACT_LANGS.items():
            if key.startswith(prefix):
                return val
        logger.debug("No Tesseract lang for %s — using eng", language_hint)
        return "eng"

    def _is_tesseract_available(self) -> bool:
        if self._tesseract_available is None:
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self._tesseract_available = True
            except Exception:
                logger.warning("Tesseract not available — Layer 2 disabled")
                self._tesseract_available = False
        return self._tesseract_available

    # ── Layer 3: LLM Correction (qwen2.5:1.5b via Ollama) ─────────────────────

    def _maybe_correct(self, result: OCRResult) -> OCRResult:
        """
        Apply LLM correction only when it's likely to help.

        Skip if:
          - LLM correction is disabled globally
          - Result is invalid or trivially short
          - Tesseract confidence is high (≥ 0.85) AND text looks clean
            (no suspicious char sequences) — saves ~25s on clean prints
        """
        if not self.use_llm_correction:
            return result
        if not result.is_valid():
            return result
        if len(result.text.strip()) < 5:
            return result

        # Fast path: skip LLM for high-confidence, clean-looking text
        if result.confidence >= 0.85 and not self._looks_garbled(result.text):
            logger.debug(
                "LLM correction skipped (conf=%.2f, text looks clean)", result.confidence
            )
            return result

        return self._correct_with_llm(result)

    # Patterns that suggest OCR garbling worth sending to LLM
    _GARBLE_RE = re.compile(
        r"[^\w\s,.\-:;/!?()\u0900-\u0DFF]"  # unexpected non-Indic, non-ASCII-punct symbols
        r"|[A-Za-z]{15,}"                     # implausibly long runs (merged words)
        r"|\d[A-Za-z]{3,}\d"                  # digits fused with letters
        r"|(.)\1{4,}",                         # 5+ repeated characters
        re.UNICODE,
    )

    def _looks_garbled(self, text: str) -> bool:
        """Return True if text contains patterns typical of OCR errors."""
        return bool(self._GARBLE_RE.search(text))

    def _correct_with_llm(self, result: OCRResult) -> OCRResult:
        """
        Send OCR text to qwen2.5:1.5b for cleanup.
        The LLM fixes character errors but does NOT translate.
        """
        prompt = self._build_correction_prompt(result.text, result.language)

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # deterministic — we want correction not creativity
                        # Cap output at ~1.5× input token estimate; avoids wasted inference
                        "num_predict": min(512, max(64, len(result.text) // 3)),
                    },
                },
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            corrected_text = data.get("response", "").strip()

            if not corrected_text:
                logger.warning("LLM correction returned empty response — keeping original")
                return result

            corrected_text = self._strip_llm_wrapper(corrected_text)

            if not corrected_text or corrected_text == result.text:
                logger.debug("LLM correction: no change")
                return result

            logger.info(
                "LLM correction applied | before=%d chars | after=%d chars",
                len(result.text), len(corrected_text),
            )
            result.raw_text = result.text
            result.text = corrected_text
            result.corrected = True
            return result

        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not running — skipping LLM correction")
        except requests.exceptions.Timeout:
            logger.warning("Ollama timed out after %ds — skipping correction", OLLAMA_TIMEOUT)
        except Exception as exc:
            logger.warning("LLM correction failed: %s", exc, exc_info=True)

        return result  # always return original on any failure

    def _build_correction_prompt(self, text: str, language: str) -> str:
        lang_name = self._language_display_name(language)
        return (
            f"You are an OCR text corrector. The following text was extracted from an image "
            f"using OCR and may contain character recognition errors, garbled characters, "
            f"or broken words. The text is in {lang_name}.\n\n"
            f"Your task:\n"
            f"1. Fix OCR errors (wrong characters, broken words, missing spaces).\n"
            f"2. Keep the SAME language — do NOT translate.\n"
            f"3. Do NOT add or remove meaning.\n"
            f"4. Return ONLY the corrected text. No explanations, no labels.\n\n"
            f"OCR text:\n{text}\n\n"
            f"Corrected text:"
        )

    @staticmethod
    def _strip_llm_wrapper(text: str) -> str:
        """Remove any markdown fences or labels the LLM might have added."""
        text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text)
        text = re.sub(r"^(corrected text|output|result)[:\s]*", "", text, flags=re.IGNORECASE)
        return text.strip()

    # ── Language detection helpers ─────────────────────────────────────────────

    def _detect_language_from_text(self, text: str) -> Optional[str]:
        """
        Unicode-range based language detection (no external library needed).
        Returns IndicTrans2 format or None if detection uncertain.
        """
        if not text.strip():
            return None

        # Count characters in each script's Unicode range
        counts: dict[str, int] = {
            "tam_Taml": len(re.findall(r"[\u0B80-\u0BFF]", text)),   # Tamil
            "hin_Deva": len(re.findall(r"[\u0900-\u097F]", text)),   # Devanagari
            "tel_Telu": len(re.findall(r"[\u0C00-\u0C7F]", text)),   # Telugu
            "kan_Knda": len(re.findall(r"[\u0C80-\u0CFF]", text)),   # Kannada
            "mal_Mlym": len(re.findall(r"[\u0D00-\u0D7F]", text)),   # Malayalam
            "ben_Beng": len(re.findall(r"[\u0980-\u09FF]", text)),   # Bengali
            "guj_Gujr": len(re.findall(r"[\u0A80-\u0AFF]", text)),   # Gujarati
            "pan_Guru": len(re.findall(r"[\u0A00-\u0A7F]", text)),   # Gurmukhi (Punjabi)
            "ory_Orya": len(re.findall(r"[\u0B00-\u0B7F]", text)),   # Odia
        }

        best_lang, best_count = max(counts.items(), key=lambda x: x[1])

        total_chars = len(re.sub(r"\s", "", text))
        if total_chars == 0:
            return None

        # Devanagari is shared by Hindi & Marathi — default to Hindi
        # (the translator will handle Marathi correctly via langdetect)
        if best_lang == "hin_Deva" and best_count > 0:
            # Attempt finer distinction between Hindi and Marathi via langdetect
            detected = self._langdetect_fallback(text)
            if detected == "mr":
                return "mar_Deva"
            return "hin_Deva"

        script_ratio = best_count / total_chars
        if script_ratio < 0.3:
            # Mostly Latin or ambiguous — check if it's English
            latin_count = len(re.findall(r"[A-Za-z]", text))
            if latin_count / total_chars > 0.5:
                return "eng_Latn"
            return None  # uncertain

        return best_lang if best_count > 0 else None

    @staticmethod
    def _langdetect_fallback(text: str) -> Optional[str]:
        """Use langdetect only when unicode heuristics are ambiguous."""
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return None

    @staticmethod
    def _indictrans_to_paddle(language: str) -> str:
        """Convert IndicTrans2 code to PaddleOCR lang slug."""
        if language in INDICTRANS_TO_PADDLE:
            return INDICTRANS_TO_PADDLE[language]
        # Devanagari scripts — use Hindi model
        if language in DEVANAGARI_LANGS:
            return "hi"
        return "en"  # safe default

    @staticmethod
    def _language_display_name(language: str) -> str:
        names = {
            "eng_Latn": "English",
            "tam_Taml": "Tamil",
            "hin_Deva": "Hindi",
            "tel_Telu": "Telugu",
            "kan_Knda": "Kannada",
            "mal_Mlym": "Malayalam",
            "ben_Beng": "Bengali",
            "guj_Gujr": "Gujarati",
            "mar_Deva": "Marathi",
            "pan_Guru": "Punjabi",
            "ory_Orya": "Odia",
        }
        return names.get(language, language)

    @staticmethod
    def _error_result(message: str, language: str = "") -> OCRResult:
        return OCRResult(
            text="",
            language=language,
            confidence=0.0,
            engine_used="none",
            error=message,
        )


# ── Pipeline helper: OCR → Translation ────────────────────────────────────────

class OCRTranslationPipeline:
    """
    Convenience wrapper: Image → OCR → IndicTrans2 translation.

    Example:
        from backend.services.ocr.ocr_engine import OCRTranslationPipeline
        pipeline = OCRTranslationPipeline(target_language="tam_Taml")
        translated = pipeline.run("sign.jpg")
        print(translated)
    """

    def __init__(
        self,
        target_language: str = "eng_Latn",
        ocr_engine: Optional[OCREngine] = None,
    ) -> None:
        self.target_language = target_language
        self.ocr_engine = ocr_engine or OCREngine()
        self._translator = None  # lazy

    def _get_translator(self):
        if self._translator is None:
            from backend.services.translation.translator import TranslationEngine
            self._translator = TranslationEngine()
        return self._translator

    def run(
        self,
        image_path: str | Path,
        language_hint: Optional[str] = None,
    ) -> str:
        """
        Extract text from image and translate to target_language.

        Returns translated text, or raises RuntimeError on failure.
        """
        result = self.ocr_engine.extract(image_path, language_hint)
        if not result.is_valid():
            raise RuntimeError(f"OCR failed: {result.error}")

        if result.language == self.target_language:
            logger.info("OCR language matches target — no translation needed")
            return result.text

        translator = self._get_translator()
        translated = translator.translate(
            text=result.text,
            src_lang=result.language,
            tgt_lang=self.target_language,
        )
        return translated


# ── CLI / quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="OCR Engine — Phase 4 test")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--lang", default=None, help="Language hint (IndicTrans2 format)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM correction")
    parser.add_argument(
        "--translate-to", default=None,
        help="Translate extracted text to this language (IndicTrans2 format)",
    )
    args = parser.parse_args()

    engine = OCREngine(use_llm_correction=not args.no_llm)
    result = engine.extract(args.image, language_hint=args.lang)

    print("\n" + "=" * 60)
    print(f"  Engine      : {result.engine_used}")
    print(f"  Language    : {result.language} ({OCREngine._language_display_name(result.language)})")
    print(f"  Confidence  : {result.confidence:.2%}")
    print(f"  LLM corrected: {result.corrected}")
    print(f"  Time        : {result.processing_time_ms}ms")
    print("=" * 60)
    print("\nExtracted text:\n")
    print(result.text)

    if args.translate_to:
        print("\n" + "-" * 60)
        print(f"Translating to {args.translate_to}…")
        try:
            pipeline = OCRTranslationPipeline(target_language=args.translate_to, ocr_engine=engine)
            translated = pipeline.run(args.image, language_hint=args.lang)
            print("\nTranslated text:\n")
            print(translated)
        except Exception as e:
            print(f"Translation failed: {e}", file=sys.stderr)
            sys.exit(1)

    if result.error:
        sys.exit(1)