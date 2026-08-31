"""Health check endpoint for SaySo server."""

from __future__ import annotations

HEALTH_PATH = "/api/v1/health"


def health_status(*, authorization: str | None, token: str) -> int:
    """Return the HTTP status code for GET /api/v1/health."""

    if not authorization:
        return 401
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or credentials != token:
        return 401
    return 200
