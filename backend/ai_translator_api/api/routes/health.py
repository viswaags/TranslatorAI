"""
Health & Status Routes
======================
GET /health         — quick alive check
GET /health/status  — detailed service + model status
GET /health/languages — list all supported languages
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from models.schemas import LANGUAGE_NAMES, HealthResponse, ServiceStatus

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_class=JSONResponse, summary="Health check")
async def health(request: Request):
    """Quick liveness check — always returns 200 if the process is alive."""
    registry = request.app.state.registry
    loaded = registry.status()

    return {
        "status": "ok",
        "version": "2.0.0",
        "services_loaded": loaded,
    }


@router.get("/health/status", summary="Detailed service status")
async def detailed_status(request: Request):
    """
    Returns which AI models are currently loaded in memory.
    All services lazy-load on first use — 'loaded: false' is normal on startup.
    """
    registry = request.app.state.registry
    status_map = registry.status()

    services = {
        name: ServiceStatus(loaded=is_loaded)
        for name, is_loaded in status_map.items()
    }

    overall = "ok" if all(v for v in status_map.values()) else "partial"

    return HealthResponse(
        status=overall,
        version="2.0.0",
        services=services,
        supported_languages=[
            {"code": code, "name": name}
            for code, name in LANGUAGE_NAMES.items()
        ],
    )


@router.get("/health/languages", summary="List supported languages")
async def list_languages():
    """Returns all language codes and display names supported by the pipeline."""
    return {
        "languages": [
            {"code": code, "name": name}
            for code, name in LANGUAGE_NAMES.items()
        ]
    }
