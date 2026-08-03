"""Shared, model-neutral service lifecycle coordination."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Generic, Iterator, Optional, TypeVar


BackendT = TypeVar("BackendT")


class LifecycleCoordinator:
    """Coordinate operation admission with serialized unload operations."""

    def __init__(self, error_factory: Callable[[], Exception]):
        self._error_factory = error_factory
        self._condition = threading.Condition(threading.RLock())
        self._active_operations = 0
        self._draining = False
        self._unload_active = False
        self._pending_unloads = 0
        self._local = threading.local()

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Admit one operation, restoring accounting even when it fails."""
        with self._condition:
            if self._draining:
                raise self._error_factory()
            self._active_operations += 1
            self._local.operation_depth = (
                getattr(self._local, "operation_depth", 0) + 1
            )
        try:
            yield
        finally:
            with self._condition:
                self._local.operation_depth -= 1
                self._active_operations -= 1
                if self._active_operations == 0:
                    self._condition.notify_all()

    def unload(self, cleanup: Callable[[], None]) -> None:
        """Drain active work and run one serialized cleanup operation.

        Queued unloads keep the coordinator draining continuously, preventing
        new work from entering between concurrent cleanup requests.
        """
        with self._condition:
            if getattr(self._local, "operation_depth", 0):
                raise self._error_factory()
            self._pending_unloads += 1
            self._draining = True
            while self._unload_active:
                self._condition.wait()
            self._pending_unloads -= 1
            self._unload_active = True
            while self._active_operations:
                self._condition.wait()

        try:
            cleanup()
        finally:
            with self._condition:
                self._unload_active = False
                if self._pending_unloads == 0:
                    self._draining = False
                self._condition.notify_all()

    @property
    def active_operations(self) -> int:
        with self._condition:
            return self._active_operations

    @property
    def is_draining(self) -> bool:
        with self._condition:
            return self._draining


class LazyBackendSlot(Generic[BackendT]):
    """Own and safely publish one lazily constructed backend reference."""

    def __init__(
        self,
        factory: Callable[[], BackendT],
        disposer: Callable[[BackendT], None],
        initial: Optional[BackendT] = None,
    ):
        self._factory = factory
        self._disposer = disposer
        self._backend = initial
        self._lock = threading.RLock()

    def get(self) -> BackendT:
        """Return the published backend, creating it exactly once if needed."""
        with self._lock:
            if self._backend is None:
                backend = self._factory()
                self._backend = backend
            return self._backend

    def unload(self) -> None:
        """Dispose the backend, retaining it when disposal fails."""
        with self._lock:
            backend = self._backend
            if backend is None:
                return
            self._disposer(backend)
            self._backend = None

    @property
    def is_initialized(self) -> bool:
        with self._lock:
            return self._backend is not None
