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

from ai_translator_api.core.readiness import model_artifact_status
from ai_translator_api.models.schemas import HealthResponse, ServiceStatus
from ai_translator_api.utils.languages import LANGUAGE_NAMES

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_class=JSONResponse, summary="Health check")
async def health(request: Request):
    """Quick liveness check — always returns 200 if the process is alive."""
    registry = request.app.state.registry
    loaded = registry.status()
    artifacts = model_artifact_status()

    return {
        "status": "ok",
        "version": "2.0.0",
        "services_loaded": loaded,
        "models_available": {
            name: status.available for name, status in artifacts.items()
        },
    }


@router.get("/health/status", summary="Detailed service status")
async def detailed_status(request: Request):
    """
    Returns which AI models are currently loaded in memory.
    All services lazy-load on first use — 'loaded: false' is normal on startup.
    """
    registry = request.app.state.registry
    status_map = registry.status()
    artifacts = model_artifact_status()

    services = {}
    for name, is_loaded in status_map.items():
        artifact = artifacts.get(name)
        services[name] = ServiceStatus(
            loaded=is_loaded,
            available=artifact.available if artifact else True,
            model_name=artifact.location if artifact else None,
            info=artifact.info if artifact else "ready",
        )

    overall = (
        "ok" if all(status.available for status in artifacts.values())
        else "degraded"
    )

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
