"""Shared, bounded, asynchronous upload persistence."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import anyio
from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadPolicy:
    """Validation and persistence limits for one upload class."""

    name: str
    upload_dir: Path
    max_size_bytes: int
    chunk_size_bytes: int
    allowed_extensions: frozenset[str]
    allowed_mime_types: Optional[frozenset[str]] = None

    def __post_init__(self) -> None:
        if self.max_size_bytes <= 0 or self.chunk_size_bytes <= 0:
            raise ValueError("Upload size and chunk size must be positive")
        extensions = frozenset(
            extension.lower()
            if extension.startswith(".")
            else f".{extension.lower()}"
            for extension in self.allowed_extensions
        )
        if not extensions:
            raise ValueError("At least one upload extension must be allowed")
        object.__setattr__(self, "allowed_extensions", extensions)
        if self.allowed_mime_types is not None:
            object.__setattr__(
                self,
                "allowed_mime_types",
                frozenset(
                    mime_type.lower()
                    for mime_type in self.allowed_mime_types
                ),
            )


def _validated_extension(upload: UploadFile, policy: UploadPolicy) -> str:
    filename = upload.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in policy.allowed_extensions:
        allowed = ", ".join(sorted(policy.allowed_extensions))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported {policy.name} extension: "
                f"{extension or '<none>'}. Allowed: {allowed}"
            ),
        )
    return extension


def _validate_mime_type(upload: UploadFile, policy: UploadPolicy) -> None:
    if policy.allowed_mime_types is None:
        return
    content_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in policy.allowed_mime_types:
        allowed = ", ".join(sorted(policy.allowed_mime_types))
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported {policy.name} MIME type: "
                f"{upload.content_type}. Allowed: {allowed}"
            ),
        )


async def _allocate_upload_path(
    directory: Path, extension: str
) -> tuple[Path, anyio.AsyncFile]:
    await anyio.Path(directory).mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        path = directory / f"{uuid.uuid4().hex}{extension}"
        try:
            output = await anyio.open_file(path, "xb")
            return path, output
        except FileExistsError:
            continue
    raise HTTPException(
        status_code=500,
        detail="Unable to allocate a unique temporary upload file",
    )


async def remove_upload(path: str | Path) -> None:
    """Remove a managed upload, tolerating prior removal."""
    try:
        await anyio.Path(path).unlink()
    except FileNotFoundError:
        return
    except OSError:
        logger.warning("Unable to remove temporary upload: %s", path)


async def save_upload(upload: UploadFile, policy: UploadPolicy) -> Path:
    """Persist an upload incrementally and enforce its limit while reading."""
    extension = _validated_extension(upload, policy)
    _validate_mime_type(upload, policy)
    path: Optional[Path] = None

    try:
        path, output = await _allocate_upload_path(
            policy.upload_dir, extension
        )
        total_size = 0
        async with output:
            while True:
                chunk = await upload.read(policy.chunk_size_bytes)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > policy.max_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"{policy.name.capitalize()} upload is too large "
                            f"(max {policy.max_size_bytes} bytes)"
                        ),
                    )
                await output.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=422,
                detail=f"{policy.name.capitalize()} upload is empty",
            )
        return path
    except BaseException:
        if path is not None:
            await remove_upload(path)
        raise


@asynccontextmanager
async def managed_upload(
    upload: UploadFile, policy: UploadPolicy
) -> AsyncIterator[str]:
    """Yield a validated temporary path and always remove it afterward."""
    path = await save_upload(upload, policy)
    try:
        yield str(path)
    finally:
        await remove_upload(path)

