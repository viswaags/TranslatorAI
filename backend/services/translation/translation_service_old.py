"""
TranslationService
==================
Primary  : IndicTrans2 distilled 200M models (ai4bharat)
Fallback : M2M-100 418M (facebook) — COMMENTED OUT, ready to enable

Routing:
  English  → Indic  : en-indic model (direct)
  Indic    → English : indic-en model (direct)
  Indic    → Indic   : pivot via English (both models)

All models are lazy-loaded on first use.
"""

import logging
import time
from typing import Optional

import torch

from ai_translator_api.core.config import settings

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EN = "eng_Latn"


class IndicTranslator:
    """
    Wraps one IndicTrans2 model for a given direction.
    direction: "en-indic" | "indic-en"
    """

    MODEL_MAP = {
        "en-indic": settings.INDICTRANS2_EN_INDIC,
        "indic-en": settings.INDICTRANS2_INDIC_EN,
    }

    def __init__(self, direction: str):
        if direction not in self.MODEL_MAP:
            raise ValueError(f"direction must be 'en-indic' or 'indic-en', got: {direction!r}")

        self.direction = direction
        self.model_name = self.MODEL_MAP[direction]

        logger.info("Loading IndicTrans2 model: %s", self.model_name)

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from IndicTransToolkit import IndicProcessor

        self.processor = IndicProcessor(inference=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model.to(DEVICE)
        self.model.eval()

        logger.info("✅ IndicTrans2 [%s] ready on %s", direction, DEVICE)

    def translate(self, sentences: list, src_lang: str, tgt_lang: str) -> list:
        if not sentences:
            return []

        batch = self.processor.preprocess_batch(
            sentences, src_lang=src_lang, tgt_lang=tgt_lang, visualize=False
        )

        inputs = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                num_beams=5,
                num_return_sequences=1,
                max_length=256,
            )

        raw_output = self.tokenizer.batch_decode(
            output_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return self.processor.postprocess_batch(raw_output, lang=tgt_lang)

    def translate_one(self, text: str, src_lang: str, tgt_lang: str) -> str:
        return self.translate([text], src_lang, tgt_lang)[0]

    def unload(self):
        del self.model
        del self.tokenizer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        logger.info("Unloaded IndicTrans2 [%s]", self.direction)


# ─────────────────────────────────────────────────────────────────────────────
# M2M-100 FALLBACK — uncomment to enable
# ─────────────────────────────────────────────────────────────────────────────
#
# ISO-639-1 codes used by M2M-100:
# M2M100_LANG_MAP = {
#     "eng_Latn": "en", "tam_Taml": "ta", "hin_Deva": "hi",
#     "tel_Telu": "te", "kan_Knda": "kn", "mal_Mlym": "ml",
#     "ben_Beng": "bn", "guj_Gujr": "gu", "mar_Deva": "mr",
#     "pan_Guru": "pa", "urd_Arab": "ur",
# }
#
# class M2M100Translator:
#     """
#     Fallback translator using facebook/m2m100_418M.
#     Supports direct Indic↔Indic without pivot.
#     Activate by setting TRANSLATION_PRIMARY=m2m100 in env.
#     """
#
#     def __init__(self):
#         from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
#         logger.info("Loading M2M-100: %s", settings.M2M100_MODEL)
#         self.tokenizer = M2M100Tokenizer.from_pretrained(settings.M2M100_MODEL)
#         self.model = M2M100ForConditionalGeneration.from_pretrained(settings.M2M100_MODEL)
#         self.model.to(DEVICE)
#         self.model.eval()
#         logger.info("✅ M2M-100 ready")
#
#     def translate_one(self, text: str, src_lang: str, tgt_lang: str) -> str:
#         src_iso = M2M100_LANG_MAP.get(src_lang, "en")
#         tgt_iso = M2M100_LANG_MAP.get(tgt_lang, "en")
#         self.tokenizer.src_lang = src_iso
#         inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
#         with torch.inference_mode():
#             generated = self.model.generate(
#                 **inputs,
#                 forced_bos_token_id=self.tokenizer.get_lang_id(tgt_iso),
#                 max_length=256,
#             )
#         return self.tokenizer.decode(generated[0], skip_special_tokens=True)
#
#     def unload(self):
#         del self.model, self.tokenizer
#         if DEVICE == "cuda":
#             torch.cuda.empty_cache()
# ─────────────────────────────────────────────────────────────────────────────


class TranslationService:
    """
    Smart routing translation service.
    Exposes the same interface regardless of backend model.
    """

    def __init__(self):
        self._en_indic: Optional[IndicTranslator] = None
        self._indic_en: Optional[IndicTranslator] = None
        # self._m2m100: Optional[M2M100Translator] = None  # fallback (uncomment to enable)
        logger.info("TranslationService ready (models load on first use)")

    # ── Lazy loaders ───────────────────────────────────────────────────────────

    def _get_en_indic(self) -> IndicTranslator:
        if self._en_indic is None:
            self._en_indic = IndicTranslator("en-indic")
        return self._en_indic

    def _get_indic_en(self) -> IndicTranslator:
        if self._indic_en is None:
            self._indic_en = IndicTranslator("indic-en")
        return self._indic_en

    # Uncomment to activate M2M-100 fallback:
    # def _get_m2m100(self) -> M2M100Translator:
    #     if self._m2m100 is None:
    #         self._m2m100 = M2M100Translator()
    #     return self._m2m100

    # ── Public API ─────────────────────────────────────────────────────────────

    def translate(
        self,
        text: str,
        tgt_lang: str,
        src_lang: Optional[str] = None,
    ) -> dict:
        """
        Translate a single string.

        Returns:
            {
                "translated_text": str,
                "src_lang": str,       # resolved source language
                "tgt_lang": str,
                "engine": str,         # "indictrans2" | "m2m100"
                "processing_ms": int,
            }
        """
        t0 = time.monotonic()

        if not src_lang:
            from ai_translator_api.utils.language_detector import LanguageDetector
            src_lang = LanguageDetector().detect(text)

        logger.info("Translate [%s → %s] | text_len=%d", src_lang, tgt_lang, len(text))

        # Same language — passthrough
        if src_lang == tgt_lang:
            return self._result(text, src_lang, tgt_lang, "passthrough", t0)

        try:
            # ── Primary: IndicTrans2 ──────────────────────────────────────────
            translated = self._indictrans2_route(text, src_lang, tgt_lang)
            return self._result(translated, src_lang, tgt_lang, "indictrans2", t0)

        except Exception as e:
            logger.warning("IndicTrans2 failed: %s — trying fallback", e)

            # ── Fallback: M2M-100 (uncomment block below to enable) ───────────
            # try:
            #     translated = self._get_m2m100().translate_one(text, src_lang, tgt_lang)
            #     logger.info("Fallback M2M-100 succeeded")
            #     return self._result(translated, src_lang, tgt_lang, "m2m100", t0)
            # except Exception as e2:
            #     logger.error("M2M-100 fallback also failed: %s", e2)
            #     raise RuntimeError(f"All translation engines failed: {e2}") from e2

            raise RuntimeError(f"Translation failed: {e}") from e

    def translate_batch(
        self,
        texts: list,
        tgt_lang: str,
        src_lang: Optional[str] = None,
    ) -> list:
        """Batch translate — all texts assumed same source language."""
        if not texts:
            return []

        if not src_lang:
            from ai_translator_api.utils.language_detector import LanguageDetector
            src_lang = LanguageDetector().detect(texts[0])

        if src_lang == tgt_lang:
            return texts

        return self._indictrans2_batch(texts, src_lang, tgt_lang)

    # ── Routing logic ──────────────────────────────────────────────────────────

    def _indictrans2_route(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if src_lang == EN:
            logger.debug("Route: EN → Indic (direct)")
            return self._get_en_indic().translate_one(text, src_lang, tgt_lang)

        if tgt_lang == EN:
            logger.debug("Route: Indic → EN (direct)")
            return self._get_indic_en().translate_one(text, src_lang, tgt_lang)

        # Indic → Indic: pivot via English
        logger.debug("Route: Indic → EN → Indic (pivot)")
        pivot = self._get_indic_en().translate_one(text, src_lang, EN)
        logger.debug("Pivot (EN): %s", pivot[:80])
        return self._get_en_indic().translate_one(pivot, EN, tgt_lang)

    def _indictrans2_batch(self, texts: list, src_lang: str, tgt_lang: str) -> list:
        if src_lang == EN:
            return self._get_en_indic().translate(texts, src_lang, tgt_lang)
        if tgt_lang == EN:
            return self._get_indic_en().translate(texts, src_lang, tgt_lang)
        # Pivot
        pivoted = self._get_indic_en().translate(texts, src_lang, EN)
        return self._get_en_indic().translate(pivoted, EN, tgt_lang)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _result(text: str, src: str, tgt: str, engine: str, t0: float) -> dict:
        return {
            "translated_text": text,
            "src_lang": src,
            "tgt_lang": tgt,
            "engine": engine,
            "processing_ms": int((time.monotonic() - t0) * 1000),
        }

    def unload(self):
        if self._en_indic:
            self._en_indic.unload()
        if self._indic_en:
            self._indic_en.unload()
        # if self._m2m100:
        #     self._m2m100.unload()
        self._en_indic = None
        self._indic_en = None
        logger.info("TranslationService unloaded")
