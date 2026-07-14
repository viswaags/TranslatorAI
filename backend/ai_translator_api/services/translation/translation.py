"""
TranslationService
==================
Primary   : IndicTrans2 distilled 200M models (ai4bharat)
Fallback  : M2M-100 418M (facebook)
LLM Refine: Qwen2.5-0.5B-Instruct (last-resort, time-gated)

Fallback triggers:
  1. Hard error from IndicTrans2
  2. IndicTrans2 confidence < CONFIDENCE_THRESHOLD
     → Both models run → higher confidence score wins
     → Both low confidence + time budget OK → Qwen refines IndicTrans2 output
     → Both low confidence + no budget    → IndicTrans2 returned (best effort)

CPU-only — optimised for Raspberry Pi 4/5 but works on any laptop for dev/testing.

Environment variables (set in core/config.py / .env):
  INDICTRANS2_EN_INDIC  — HF model id or local path
  INDICTRANS2_INDIC_EN  — HF model id or local path
  M2M100_MODEL          — HF model id or local path
  QWEN_MODEL            — HF model id or local path  (default: Qwen/Qwen2.5-0.5B-Instruct)
  MAX_TOTAL_MS          — overall deadline per call   (default: 8000)
  QWEN_MIN_BUDGET_MS    — minimum ms needed for Qwen  (default: 2000)
  CONFIDENCE_THRESHOLD  — IndicTrans2 pass bar        (default: 0.50)
"""

import logging
import time
from typing import Optional

import torch

from core.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Runtime constants  (all tuneable via settings / env)
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cpu"
EN = "eng_Latn"

CONFIDENCE_THRESHOLD: float = getattr(settings, "CONFIDENCE_THRESHOLD", 0.50)
MAX_TOTAL_MS:         int   = getattr(settings, "MAX_TOTAL_MS",         8000)
QWEN_MIN_BUDGET_MS:   int   = getattr(settings, "QWEN_MIN_BUDGET_MS",   2000)

# Human-readable BCP-47-ish names used inside Qwen prompts
LANG_DISPLAY = {
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
    "urd_Arab": "Urdu",
}

M2M100_LANG_MAP = {
    "eng_Latn": "en", "tam_Taml": "ta", "hin_Deva": "hi",
    "tel_Telu": "te", "kan_Knda": "kn", "mal_Mlym": "ml",
    "ben_Beng": "bn", "guj_Gujr": "gu", "mar_Deva": "mr",
    "pan_Guru": "pa", "urd_Arab": "ur",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared confidence helper
# ─────────────────────────────────────────────────────────────────────────────

def _seq_confidence(
    sequences_scores: torch.Tensor,
    sequences: torch.Tensor,
    pad_id: int,
) -> list[float]:
    """
    Convert beam-search sequence log-probs → per-sequence confidence in [0, 1].
    Formula: exp(sum_log_prob / non_pad_tokens)  →  geometric-mean token prob.
    Near-zero overhead — scores are already computed during generate().
    """
    confs = []
    for score, seq in zip(sequences_scores, sequences):
        n_tokens = max((seq != pad_id).sum().item(), 1)
        per_token_lp = score.item() / n_tokens      # negative float
        confs.append(round(float(torch.exp(torch.tensor(per_token_lp))), 4))
    return confs


# ─────────────────────────────────────────────────────────────────────────────
# IndicTrans2  (Primary)
# ─────────────────────────────────────────────────────────────────────────────

class IndicTranslator:
    MODEL_MAP = {
        "en-indic": settings.INDICTRANS2_EN_INDIC,
        "indic-en": settings.INDICTRANS2_INDIC_EN,
    }

    def __init__(self, direction: str):
        if direction not in self.MODEL_MAP:
            raise ValueError(
                f"direction must be 'en-indic' or 'indic-en', got: {direction!r}"
            )

        self.direction  = direction
        self.model_name = self.MODEL_MAP[direction]
        logger.info("Loading IndicTrans2 [%s]: %s", direction, self.model_name)

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit import IndicProcessor

        self.processor = IndicProcessor(inference=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model.to(DEVICE).eval()
        logger.info("✅ IndicTrans2 [%s] ready on %s", direction, DEVICE.upper())

    # ------------------------------------------------------------------

    def translate(
        self,
        sentences: list[str],
        src_lang: str,
        tgt_lang: str,
        return_scores: bool = False,
    ) -> list[str] | tuple[list[str], list[float]]:
        if not sentences:
            return ([], []) if return_scores else []

        batch = self.processor.preprocess_batch(
            sentences, src_lang=src_lang, tgt_lang=tgt_lang, visualize=False
        )
        inputs = self.tokenizer(
            batch, padding="longest", truncation=True,
            max_length=256, return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                use_cache=True,
                num_beams=5,
                num_return_sequences=1,
                max_length=256,
                output_scores=return_scores,
                return_dict_in_generate=return_scores,
            )

        sequences = output.sequences if return_scores else output
        raw = self.tokenizer.batch_decode(
            sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        translated = self.processor.postprocess_batch(raw, lang=tgt_lang)

        if not return_scores:
            return translated

        confs = _seq_confidence(
            output.sequences_scores, sequences, self.tokenizer.pad_token_id
        )
        return translated, confs

    def translate_one(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        return_scores: bool = False,
    ) -> str | tuple[str, float]:
        result = self.translate([text], src_lang, tgt_lang, return_scores=return_scores)
        if return_scores:
            texts, scores = result
            return texts[0], scores[0]
        return result[0]

    def unload(self):
        del self.model, self.tokenizer
        logger.info("Unloaded IndicTrans2 [%s]", self.direction)


# ─────────────────────────────────────────────────────────────────────────────
# M2M-100  (Fallback)
# ─────────────────────────────────────────────────────────────────────────────

class M2M100Translator:
    def __init__(self):
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        logger.info("Loading M2M-100: %s", settings.M2M100_MODEL)
        self.tokenizer = M2M100Tokenizer.from_pretrained(settings.M2M100_MODEL)
        self.model = M2M100ForConditionalGeneration.from_pretrained(settings.M2M100_MODEL)
        self.model.to(DEVICE).eval()
        logger.info("✅ M2M-100 ready on %s", DEVICE.upper())

    # ------------------------------------------------------------------

    def translate_one(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        return_scores: bool = False,
    ) -> str | tuple[str, float]:
        src_iso = M2M100_LANG_MAP.get(src_lang, "en")
        tgt_iso = M2M100_LANG_MAP.get(tgt_lang, "en")

        self.tokenizer.src_lang = src_iso
        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.get_lang_id(tgt_iso),
                num_beams=5,
                max_length=256,
                output_scores=return_scores,
                return_dict_in_generate=return_scores,
            )

        sequences = output.sequences if return_scores else output
        translated = self.tokenizer.decode(sequences[0], skip_special_tokens=True)

        if not return_scores:
            return translated

        confs = _seq_confidence(
            output.sequences_scores, sequences, self.tokenizer.pad_token_id
        )
        return translated, confs[0]

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        return_scores: bool = False,
    ) -> list[str] | tuple[list[str], list[float]]:
        """M2M-100 has no batch processor — loops translate_one."""
        results = [
            self.translate_one(t, src_lang, tgt_lang, return_scores) for t in texts
        ]
        if return_scores:
            texts_out, scores = zip(*results)
            return list(texts_out), list(scores)
        return results

    def unload(self):
        del self.model, self.tokenizer
        logger.info("Unloaded M2M-100")


# ─────────────────────────────────────────────────────────────────────────────
# Qwen LLM  (Last-resort refinement, time-gated)
# ─────────────────────────────────────────────────────────────────────────────

class QwenRefiner:
    """
    Wraps Qwen2.5-0.5B-Instruct (or any small causal LM) as a translation
    refiner / last-resort engine.

    Designed to run ONLY when:
      (a) both seq2seq models produced low-confidence output, OR
      (b) both seq2seq models hard-errored,
    AND there is still enough wall-clock budget left.

    On a Raspberry Pi 4 (4 GB), 0.5B INT8 → ~3–6 s/sentence.
    On a laptop CPU it is ~1–2 s.  Adjust QWEN_MIN_BUDGET_MS accordingly.
    """

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = getattr(settings, "QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        logger.info("Loading Qwen refiner: %s", model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,   # CPU — float32 is safest
        ).to(DEVICE)
        self.model.eval()
        logger.info("✅ Qwen refiner ready on %s", DEVICE.upper())

    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        text: str,
        src_lang: str,
        tgt_lang: str,
        draft: Optional[str],
        detected_hint: Optional[str],
    ) -> str:
        """
        Build a tightly-scoped prompt.

        Parameters
        ----------
        text          : original source sentence
        src_lang      : BCP-47-ish code, e.g. "tam_Taml"
        tgt_lang      : BCP-47-ish code, e.g. "eng_Latn"
        draft         : low-confidence machine translation to refine (may be None)
        detected_hint : language name string from LanguageDetector, if available
                        e.g. "Tamil" — used to reinforce source language identity.
        """
        src_name = LANG_DISPLAY.get(src_lang, src_lang)
        tgt_name = LANG_DISPLAY.get(tgt_lang, tgt_lang)

        # If the language detector confirmed the source language, weave it in
        # so the model does not have to guess from the script alone.
        lang_note = ""
        if detected_hint and detected_hint.lower() != src_name.lower():
            # detector returned something slightly different — mention both
            lang_note = (
                f"Note: automatic language detection identified the source as "
                f'"{detected_hint}", which corresponds to {src_name}.\n'
            )
        elif detected_hint:
            lang_note = (
                f"Note: the source language has been automatically confirmed "
                f"as {detected_hint} by a language detector.\n"
            )

        if draft:
            # Refinement mode — we have a weak machine translation to improve
            prompt = (
                f"You are an expert translator specialising in Indian languages.\n"
                f"{lang_note}"
                f"\nTask: The sentence below was originally written in {src_name}. "
                f"A machine translation model produced a low-confidence draft "
                f"translation into {tgt_name}. Correct any errors, improve fluency, "
                f"and ensure the meaning is fully preserved.\n"
                f"\nSource ({src_name}):\n{text}\n"
                f"\nMachine draft ({tgt_name}):\n{draft}\n"
                f"\nProvide ONLY the corrected {tgt_name} translation. "
                f"Do not explain, do not add notes."
            )
        else:
            # Hard-fallback mode — no draft available, translate from scratch
            prompt = (
                f"You are an expert translator specialising in Indian languages.\n"
                f"{lang_note}"
                f"\nTask: Translate the following sentence from {src_name} to "
                f"{tgt_name} accurately and naturally.\n"
                f"\nSource ({src_name}):\n{text}\n"
                f"\nProvide ONLY the {tgt_name} translation. "
                f"Do not explain, do not add notes."
            )
        return prompt

    # ------------------------------------------------------------------

    def refine(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        draft: Optional[str] = None,
        detected_hint: Optional[str] = None,
    ) -> str:
        """
        Run Qwen to produce / refine a translation.

        Returns the translated string (stripped).
        Raises on model error — caller must catch.
        """
        prompt = self._build_prompt(text, src_lang, tgt_lang, draft, detected_hint)
        logger.debug("Qwen prompt:\n%s", prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,          # greedy — deterministic, faster on CPU
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip echoed prompt)
        new_tokens = out[0][prompt_len:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        logger.info("Qwen output: %r", result[:120])
        return result

    def unload(self):
        del self.model, self.tokenizer
        logger.info("Unloaded Qwen refiner")


# ─────────────────────────────────────────────────────────────────────────────
# TranslationService  (orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class TranslationService:
    """
    Full decision flow
    ──────────────────
    IndicTrans2
      ├── conf ≥ THRESHOLD ──► ✅ return it2           (M2M-100 + Qwen never load)
      │
      ├── conf < THRESHOLD ──► run M2M-100
      │       ├── m2m_conf > it2_conf ──► return M2M-100 ✅
      │       ├── it2_conf > m2m_conf ──► return IndicTrans2 ✅
      │       └── M2M-100 error + it2 exists
      │               ├── budget OK ──► 🤖 Qwen refine(it2_draft)
      │               │       ├── success ──► return Qwen result
      │               │       └── error   ──► return it2 best-effort
      │               └── no budget ──► return it2 best-effort (low_confidence=True)
      │
      └── IndicTrans2 hard error ──► run M2M-100
              ├── success ──► return M2M-100 ✅
              └── M2M-100 also errors
                      ├── budget OK ──► 🤖 Qwen translate from scratch
                      │       ├── success ──► return Qwen result
                      │       └── error   ──► raise RuntimeError
                      └── no budget ──► raise RuntimeError
    """

    def __init__(self):
        self._en_indic: Optional[IndicTranslator] = None
        self._indic_en: Optional[IndicTranslator] = None
        self._m2m100:   Optional[M2M100Translator] = None
        self._qwen:     Optional[QwenRefiner]      = None
        logger.info(
            "TranslationService ready (lazy loading) | "
            "threshold=%.2f | max_ms=%d | qwen_budget_ms=%d",
            CONFIDENCE_THRESHOLD, MAX_TOTAL_MS, QWEN_MIN_BUDGET_MS,
        )

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_en_indic(self) -> IndicTranslator:
        if self._en_indic is None:
            self._en_indic = IndicTranslator("en-indic")
        return self._en_indic

    def _get_indic_en(self) -> IndicTranslator:
        if self._indic_en is None:
            self._indic_en = IndicTranslator("indic-en")
        return self._indic_en

    def _get_m2m100(self) -> M2M100Translator:
        if self._m2m100 is None:
            self._m2m100 = M2M100Translator()
        return self._m2m100

    def _get_qwen(self) -> QwenRefiner:
        if self._qwen is None:
            self._qwen = QwenRefiner()
        return self._qwen

    # ── Time budget guard ─────────────────────────────────────────────────────

    def _qwen_budget_ok(self, t0: float) -> bool:
        """True when enough wall-clock time remains for a Qwen call."""
        elapsed_ms = (time.monotonic() - t0) * 1000
        remaining_ms = MAX_TOTAL_MS - elapsed_ms
        ok = remaining_ms >= QWEN_MIN_BUDGET_MS
        if not ok:
            logger.warning(
                "Skipping Qwen — only %.0f ms remaining (need %d ms)",
                remaining_ms, QWEN_MIN_BUDGET_MS,
            )
        return ok

    # ── Public API ────────────────────────────────────────────────────────────

    def translate(
        self,
        text: str,
        tgt_lang: str,
        src_lang: Optional[str] = None,
    ) -> dict:
        """
        Translate a single sentence.

        Returns
        -------
        dict with keys:
            translated_text : str
            src_lang        : str
            tgt_lang        : str
            engine          : "indictrans2" | "m2m100" | "qwen" | "passthrough"
            confidence      : float   (0.0 for qwen / passthrough)
            low_confidence  : bool    (True only on best-effort fallback)
            detected_lang   : str | None  (set when src_lang was auto-detected)
            processing_ms   : int
        """
        t0 = time.monotonic()
        detected_hint: Optional[str] = None   # human-readable detection result

        # ── Language detection ────────────────────────────────────────────────
        if not src_lang:
            from utils.language_detector import LanguageDetector
            detected = LanguageDetector().detect(text)   # returns e.g. "tam_Taml"
            src_lang = detected
            # Map to display name so Qwen prompt knows what was inferred
            detected_hint = LANG_DISPLAY.get(detected, detected)
            logger.info("Auto-detected src_lang=%s (%s)", src_lang, detected_hint)
        else:
            # src_lang was provided by caller — Qwen still benefits from display name
            detected_hint = LANG_DISPLAY.get(src_lang, src_lang)

        logger.info("Translate [%s → %s] | len=%d", src_lang, tgt_lang, len(text))

        if src_lang == tgt_lang:
            return self._result(
                text, src_lang, tgt_lang, "passthrough", t0, 1.0, False, detected_hint
            )

        # ── Step 1: IndicTrans2 ───────────────────────────────────────────────
        it2_text: Optional[str] = None
        it2_conf: float         = 0.0

        try:
            it2_text, it2_conf = self._indictrans2_route(text, src_lang, tgt_lang)
            logger.info("IndicTrans2 conf=%.4f", it2_conf)

            if it2_conf >= CONFIDENCE_THRESHOLD:
                # ✅ High confidence — return immediately
                return self._result(
                    it2_text, src_lang, tgt_lang, "indictrans2",
                    t0, it2_conf, False, detected_hint,
                )

            logger.warning(
                "IndicTrans2 low conf (%.4f < %.2f) — running M2M-100",
                it2_conf, CONFIDENCE_THRESHOLD,
            )

        except Exception as e_it2:
            logger.warning("IndicTrans2 hard error: %s — falling back to M2M-100", e_it2)

        # ── Step 2: M2M-100 ───────────────────────────────────────────────────
        try:
            m2m_text, m2m_conf = self._get_m2m100().translate_one(
                text, src_lang, tgt_lang, return_scores=True
            )
            logger.info("M2M-100 conf=%.4f", m2m_conf)

            # Hard fallback path (IndicTrans2 errored — no it2_text)
            if it2_text is None:
                return self._result(
                    m2m_text, src_lang, tgt_lang, "m2m100",
                    t0, m2m_conf, False, detected_hint,
                )

            # Comparison path — both ran
            if m2m_conf >= it2_conf:
                logger.info("M2M-100 wins (%.4f ≥ %.4f)", m2m_conf, it2_conf)
                return self._result(
                    m2m_text, src_lang, tgt_lang, "m2m100",
                    t0, m2m_conf, False, detected_hint,
                )
            else:
                logger.info("IndicTrans2 wins (%.4f > %.4f)", it2_conf, m2m_conf)
                return self._result(
                    it2_text, src_lang, tgt_lang, "indictrans2",
                    t0, it2_conf, False, detected_hint,
                )

        except Exception as e_m2m:
            logger.error("M2M-100 failed: %s", e_m2m)

        # ── Step 3: Qwen (last resort, time-gated) ────────────────────────────
        #
        # We land here in two scenarios:
        #   (A) it2_text exists  → both models low-conf, M2M-100 errored
        #   (B) it2_text is None → both models hard-errored
        #
        if self._qwen_budget_ok(t0):
            try:
                # Scenario A: refine the weak it2 draft
                # Scenario B: translate from scratch (draft=None)
                qwen_text = self._get_qwen().refine(
                    text       = text,
                    src_lang   = src_lang,
                    tgt_lang   = tgt_lang,
                    draft      = it2_text,          # None in scenario B
                    detected_hint = detected_hint,  # passes language-detector info
                )
                mode = "qwen_refine" if it2_text else "qwen"
                logger.info("Qwen (%s) succeeded", mode)
                return self._result(
                    qwen_text, src_lang, tgt_lang, mode,
                    t0, 0.0, False, detected_hint,
                )

            except Exception as e_qwen:
                logger.error("Qwen also failed: %s", e_qwen)

        # ── Total failure ─────────────────────────────────────────────────────
        if it2_text:
            # Scenario A best-effort: return IndicTrans2 low-conf output
            logger.warning("All engines exhausted — returning IndicTrans2 best effort")
            return self._result(
                it2_text, src_lang, tgt_lang, "indictrans2",
                t0, it2_conf, True, detected_hint,
            )

        # Scenario B: nothing at all — hard raise
        raise RuntimeError(
            f"All translation engines failed for [{src_lang} → {tgt_lang}]"
        )

    # ------------------------------------------------------------------

    def translate_batch(
        self,
        texts: list[str],
        tgt_lang: str,
        src_lang: Optional[str] = None,
    ) -> list[dict]:
        """
        Translate a list of sentences.
        Each item returns the same dict structure as translate().
        src_lang is detected once from the first sentence if not provided.
        """
        if not texts:
            return []

        if not src_lang:
            from utils.language_detector import LanguageDetector
            src_lang = LanguageDetector().detect(texts[0])
            logger.info(
                "Batch: auto-detected src_lang=%s from first sentence", src_lang
            )

        return [self.translate(t, tgt_lang, src_lang) for t in texts]

    # ── IndicTrans2 routing ───────────────────────────────────────────────────

    def _indictrans2_route(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> tuple[str, float]:
        """
        Route through correct IndicTrans2 model.
        Indic→Indic pivots via English; confidence = min of both hops.
        """
        if src_lang == EN:
            return self._get_en_indic().translate_one(
                text, src_lang, tgt_lang, return_scores=True
            )

        if tgt_lang == EN:
            return self._get_indic_en().translate_one(
                text, src_lang, tgt_lang, return_scores=True
            )

        # Indic → Indic (pivot)
        pivot, conf1 = self._get_indic_en().translate_one(
            text, src_lang, EN, return_scores=True
        )
        final, conf2 = self._get_en_indic().translate_one(
            pivot, EN, tgt_lang, return_scores=True
        )
        return final, min(conf1, conf2)

    # ── Result builder ────────────────────────────────────────────────────────

    @staticmethod
    def _result(
        text: str,
        src: str,
        tgt: str,
        engine: str,
        t0: float,
        confidence: float,
        low_confidence: bool,
        detected_hint: Optional[str],
    ) -> dict:
        return {
            "translated_text": text,
            "src_lang":        src,
            "tgt_lang":        tgt,
            # indictrans2 | m2m100 | qwen | qwen_refine | passthrough
            "engine":          engine,
            "confidence":      confidence,
            # True only when best-effort output is returned (both models failed)
            "low_confidence":  low_confidence,
            # Human-readable name of the detected/provided source language
            # useful for callers / API consumers to surface in UI
            "detected_lang":   detected_hint,
            "processing_ms":   int((time.monotonic() - t0) * 1000),
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def unload(self):
        if self._en_indic: self._en_indic.unload()
        if self._indic_en: self._indic_en.unload()
        if self._m2m100:   self._m2m100.unload()
        if self._qwen:     self._qwen.unload()
        self._en_indic = self._indic_en = self._m2m100 = self._qwen = None
        logger.info("TranslationService: all models unloaded")