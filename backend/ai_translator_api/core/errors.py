"""Shared backend error categories.

Subsystems retain their public exception names and inherit these categories so
transport layers can handle failures consistently without depending on a
specific inference engine.
"""


class ServiceError(Exception):
    """Base class for expected backend service failures."""


class ValidationError(ServiceError, ValueError):
    """A request or service input failed validation."""


class UnsupportedInputError(ValidationError):
    """The supplied input type, language, or selection is unsupported."""


class ModelUnavailableError(ServiceError):
    """Required local model artifacts or runtime dependencies are unavailable."""


class ModelLoadError(ServiceError):
    """Available model artifacts could not be loaded."""


class InferenceError(ServiceError):
    """A backend failed while processing a valid request."""


class LifecycleError(ServiceError):
    """A service cannot accept work during a lifecycle transition."""


class OverloadError(ServiceError):
    """A service has reached its configured work capacity."""

