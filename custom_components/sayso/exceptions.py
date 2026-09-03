"""Exceptions raised by the SaySo llama.cpp client."""

from __future__ import annotations


class SaySoError(Exception):
    """Base exception for SaySo client errors."""


class SaySoAuthError(SaySoError):
    """Raised when llama.cpp rejects the API key."""


class SaySoConnectionError(SaySoError):
    """Raised when llama.cpp is unreachable."""


class SaySoTimeoutError(SaySoError):
    """Raised when a llama.cpp request times out."""


class SaySoInvalidResponseError(SaySoError):
    """Raised when llama.cpp returns an invalid or unusable response."""


class SaySoHttpError(SaySoError):
    """Raised when llama.cpp returns an unexpected HTTP error."""

    def __init__(self, status: int, message: str | None = None) -> None:
        self.status = status
        super().__init__(message or f"HTTP {status}")


class SaySoModelNotFoundError(SaySoError):
    """Raised when the configured model is not available on llama.cpp."""


class SaySoInvalidToolEnvelopeError(SaySoError):
    """Raised when a compiled tool envelope fails the outer transport contract."""
