"""Base exceptions for the financial news sentiment analysis system."""
from typing import Any, Optional


class BaseAppException(Exception):
    """Base exception class for all application-specific exceptions."""

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        details: Optional[Any] = None,
    ) -> None:
        """Initialize the base exception.

        Args:
            message: Human-readable error message
            component: Name of the component where the error occurred
            details: Additional error details (can be dict, list, etc.)
        """
        self.message = message
        self.component = component or "unknown"
        self.details = details
        super().__init__(self.message)


class ConfigurationError(BaseAppException):
    """Raised when there is an error in the configuration."""

    pass


class DatasetError(BaseAppException):
    """Raised when there is an error loading or processing a dataset."""

    pass


class ModelError(BaseAppException):
    """Raised when there is an error with model operations."""

    pass


class LiveDataError(BaseAppException):
    """Raised when there is an error fetching or processing live data."""

    pass


class ValidationError(BaseAppException):
    """Raised when there is an error validating input data."""

    pass


class APIError(BaseAppException):
    """Raised when there is an error in API operations."""

    pass


class CacheError(BaseAppException):
    """Raised when there is an error with the caching system."""

    pass
