"""
Translation Service — Production (CPU Only + Quantized + Qwen Fallback)
=======================================================================

✔ CPU only (no CUDA)
✔ IndicTrans2 (primary engine)
✔ INT8 dynamic quantization
✔ Fast decoding (beam=1)
✔ Lazy loading (RAM efficient)
✔ Qwen fallback via Ollama
✔ Production-ready structure
"""
# -*- coding: utf-8 -*-

import time
import logging
from typing import Optional
import requests

import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE = "cpu"   # 🔥 FORCE CPU
EN = "eng_Latn"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────
# QWEN FALLBACK (OLLAMA)
# ─────────────────────────────────────────────
class QwenTranslator:
    def __init__(self, model: str = "qwen2.5:1.5b"):
        self.model = model

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        prompt = f"""
Translate the following text from {src_lang} to {tgt_lang}.
Return ONLY the translated sentence.

Text:
{text}
"""
        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60,
            )
            return res.json()["response"].strip()
        except Exception as e:
            logger.error(f"Qwen failed: {e}")
            raise


# ─────────────────────────────────────────────
# INDIC TRANSLATOR (CPU + QUANTIZED)
# ─────────────────────────────────────────────
class IndicTranslator:
    MODEL_MAP = {
        "en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
        "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M",
    }

    def __init__(self, direction: str):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from IndicTransToolkit import IndicProcessor

        self.direction = direction
        self.model_name = self.MODEL_MAP[direction]

        logger.info(f"🔄 Loading {self.model_name}")

        self.processor = IndicProcessor(inference=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        model.to(DEVICE)
        model.eval()

        # 🔥 INT8 QUANTIZATION (CPU BOOST)
        try:
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("⚡ Quantization applied (INT8)")
        except Exception as e:
            logger.warning(f"Quantization skipped: {e}")

        # 🔥 Torch compile (optional boost)
        try:
            model = torch.compile(model)
        except Exception:
            pass

        self.model = model

        logger.info(f"✅ Ready [{direction}]")

    def translate_one(self, text: str, src_lang: str, tgt_lang: str) -> str:
        batch = self.processor.preprocess_batch(
            [text], src_lang=src_lang, tgt_lang=tgt_lang, visualize=False
        )

        inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,   # 🔥 optimized
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                num_beams=1,       # 🔥 FAST
                do_sample=False,
                max_length=128,
                use_cache=True,
            )

        decoded = self.tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return self.processor.postprocess_batch(decoded, lang=tgt_lang)[0]


# ─────────────────────────────────────────────
# MAIN TRANSLATION SERVICE
# ─────────────────────────────────────────────
class TranslationService:

    def __init__(self):
        self._en_indic: Optional[IndicTranslator] = None
        self._indic_en: Optional[IndicTranslator] = None
        self._qwen = QwenTranslator()

        logger.info("🚀 TranslationService initialized")

    # ── Lazy loading ──────────────────────────

    def _get_en_indic(self):
        if self._en_indic is None:
            self._en_indic = IndicTranslator("en-indic")
        return self._en_indic

    def _get_indic_en(self):
        if self._indic_en is None:
            self._indic_en = IndicTranslator("indic-en")
        return self._indic_en

    # ── MAIN FUNCTION ─────────────────────────

    def translate(
        self,
        text: str,
        tgt_lang: str,
        src_lang: Optional[str] = None,
    ) -> dict:

        start = time.time()

        # Lazy import → faster startup
        if not src_lang:
            from langdetect import detect
            try:
                code = detect(text)
            except:
                code = "en"

            src_lang = {
                "en": "eng_Latn",
                "ta": "tam_Taml",
                "hi": "hin_Deva",
                "te": "tel_Telu",
                "kn": "kan_Knda",
                "ml": "mal_Mlym",
            }.get(code, "eng_Latn")

        if src_lang == tgt_lang:
            return self._result(text, src_lang, tgt_lang, "passthrough", start)

        try:
            # 🔥 FAST PATH (IndicTrans2)
            if src_lang == EN:
                result = self._get_en_indic().translate_one(text, src_lang, tgt_lang)

            elif tgt_lang == EN:
                result = self._get_indic_en().translate_one(text, src_lang, tgt_lang)

            else:
                pivot = self._get_indic_en().translate_one(text, src_lang, EN)
                result = self._get_en_indic().translate_one(pivot, EN, tgt_lang)

            # Basic quality check
            if len(result.strip()) < 2:
                raise ValueError("Low quality output")

            return self._result(result, src_lang, tgt_lang, "indictrans2", start)

        except Exception as e:
            logger.warning(f"⚠️ IndicTrans failed → fallback Qwen: {e}")

            # 🧠 QWEN FALLBACK
            try:
                qwen_result = self._qwen.translate(text, src_lang, tgt_lang)
                return self._result(qwen_result, src_lang, tgt_lang, "qwen", start)
            except Exception:
                raise RuntimeError("All translation methods failed")

    # ── HELPER ────────────────────────────────

    @staticmethod
    def _result(text, src, tgt, engine, start):
        return {
            "translated_text": text,
            "src_lang": src,
            "tgt_lang": tgt,
            "engine": engine,
            "processing_ms": int((time.time() - start) * 1000),
        }


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    service = TranslationService()

    res = service.translate(
        text="Hello, how are you?",
        tgt_lang="tam_Taml"
    )
    print(res["translated_text"].encode('utf-8').decode('utf-8'))
    print("\n✅ RESULT:")
    print(res)