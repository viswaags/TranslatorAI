"""Central FastAPI exception mapping for the backend."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from ai_translator_api.core.errors import (
    InferenceError,
    LifecycleError,
    ModelLoadError,
    ModelUnavailableError,
    OverloadError,
    ServiceError,
    UnsupportedInputError,
    ValidationError,
)

logger = logging.getLogger(__name__)


def _error_response(
    *,
    status_code: int,
    error_type: str,
    category: str,
    message: str,
    details: Any = None,
) -> JSONResponse:
    error = {
        "type": error_type,
        "category": category,
        "message": message,
    }
    if details is not None:
        error["details"] = details
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": error},
    )


def service_error_status(error: ServiceError) -> tuple[int, str]:
    """Return the stable HTTP status and category for a service failure."""
    if isinstance(error, UnsupportedInputError):
        return 400, "unsupported_input"
    if isinstance(error, ValidationError):
        return 422, "validation"
    if isinstance(error, ModelUnavailableError):
        return 503, "model_unavailable"
    if isinstance(error, ModelLoadError):
        return 500, "model_load"
    if isinstance(error, InferenceError):
        return 500, "inference"
    if isinstance(error, LifecycleError):
        return 503, "lifecycle"
    if isinstance(error, OverloadError):
        return 429, "overload"
    return 500, "service"


async def service_error_handler(
    request: Request, error: ServiceError
) -> JSONResponse:
    del request
    status_code, category = service_error_status(error)
    return _error_response(
        status_code=status_code,
        error_type=type(error).__name__,
        category=category,
        message=str(error),
    )


async def request_validation_error_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    del request
    return _error_response(
        status_code=422,
        error_type=type(error).__name__,
        category="validation",
        message="Request validation failed",
        details=error.errors(),
    )


async def http_error_handler(
    request: Request, error: HTTPException
) -> JSONResponse:
    del request
    detail = error.detail
    message = detail if isinstance(detail, str) else "HTTP request failed"
    return _error_response(
        status_code=error.status_code,
        error_type=type(error).__name__,
        category="http",
        message=message,
        details=None if isinstance(detail, str) else detail,
    )


async def unexpected_error_handler(
    request: Request, error: Exception
) -> JSONResponse:
    logger.error(
        "Unhandled API error | method=%s | path=%s | type=%s",
        request.method,
        request.url.path,
        type(error).__name__,
        exc_info=(type(error), error, error.__traceback__),
    )
    return _error_response(
        status_code=500,
        error_type="InternalServerError",
        category="internal",
        message="An unexpected internal error occurred",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install the backend's canonical exception handlers."""
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(
        RequestValidationError, request_validation_error_handler
    )
    app.add_exception_handler(HTTPException, http_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
