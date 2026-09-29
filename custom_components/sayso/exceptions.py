
from __future__ import annotations


class SaySoError(Exception):
    pass


class SaySoAuthError(SaySoError):
    pass


class SaySoConnectionError(SaySoError):
    pass


class SaySoTimeoutError(SaySoError):
    pass


class SaySoInvalidResponseError(SaySoError):
    pass


class SaySoHttpError(SaySoError):

    def __init__(self, status: int, message: str | None = None) -> None:
        self.status = status
        super().__init__(message or f"HTTP {status}")


class SaySoModelNotFoundError(SaySoError):
    pass


class SaySoInvalidToolEnvelopeError(SaySoError):
    pass


class SaySoModelLoadError(SaySoError):
    pass


class SaySoDependencyError(SaySoError):
    pass
