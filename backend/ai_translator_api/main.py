"""
AI Translator — FastAPI Backend
================================
Full pipeline: Text | OCR | Speech  →  Translation  →  Text + TTS

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Docs:
    http://localhost:8000/docs
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from core.config import settings
from core.registry import ServiceRegistry
from api.routes import text, ocr, speech, health

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise the global service registry (lazy — models not loaded yet).
    Shutdown: gracefully unload all models to free RAM.
    """
    logger.info("🚀 AI Translator API starting up…")
    registry = ServiceRegistry.get_instance()
    app.state.registry = registry
    logger.info("✅ ServiceRegistry ready (models will load lazily on first use)")
    yield
    logger.info("🛑 Shutting down — unloading models…")
    registry.unload_all()
    logger.info("✅ Shutdown complete")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Translator API",
    description=(
        "Multilingual AI translation pipeline for Indian languages.\n\n"
        "**Input modes:** Text · OCR (image) · Speech (audio)\n\n"
        "**Output:** Translated text + synthesised speech (WAV/base64)\n\n"
        "**Translation:** IndicTrans2 (primary) | M2M-100 (fallback — commented)\n\n"
        "**TTS:** Indic Parler (primary) | eSpeak-NG (fallback)\n\n"
        "**OCR:** PaddleOCR → Tesseract → Qwen LLM correction"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(text.router,   prefix="/api/v1/text",   tags=["Text Pipeline"])
app.include_router(ocr.router,    prefix="/api/v1/ocr",    tags=["OCR Pipeline"])
app.include_router(speech.router, prefix="/api/v1/speech", tags=["Speech Pipeline"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "AI Translator API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "pipelines": {
            "text":   "/api/v1/text/translate",
            "ocr":    "/api/v1/ocr/extract-and-translate",
            "speech": "/api/v1/speech/transcribe-and-translate",
        },
    }
