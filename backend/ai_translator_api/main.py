"""
AI Translator — FastAPI Backend
================================
Full pipeline: Text | OCR | Speech  →  Translation  →  Text + TTS

Run:
    uvicorn ai_translator_api.main:app --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from ai_translator_api.api.error_handlers import register_exception_handlers
from ai_translator_api.api.routes import health, ocr, speech, text
from ai_translator_api.core.config import settings
from ai_translator_api.core.readiness import model_artifact_status
from ai_translator_api.core.registry import ServiceRegistry

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
    for service_name, artifact in model_artifact_status().items():
        log = logger.info if artifact.available else logger.warning
        log(
            "Startup readiness | service=%s | available=%s | location=%s | %s",
            service_name,
            artifact.available,
            artifact.location,
            artifact.info,
        )
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
        "**Output:** Translated text with optional Piper speech (WAV/base64)\n\n"
        "**Translation:** IndicTrans2 using local CTranslate2 models\n\n"
        "**Speech recognition:** Faster Whisper using a local model\n\n"
        "**OCR:** PaddleOCR using local detection and recognition models\n\n"
        "**Speech synthesis:** Piper using local voice models"
    ),
    version="2.0.0",
    lifespan=lifespan,
)
register_exception_handlers(app)

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
