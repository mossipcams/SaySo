"""Bearer token authentication for SaySo server endpoints."""

from __future__ import annotations

import hmac


def parse_bearer_token(authorization: str | None) -> str | None:
    """Return the bearer credential from an Authorization header."""

    if not authorization:
        return None
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials:
        return None
    return credentials


def bearer_token_valid(*, authorization: str | None, expected_token: str) -> bool:
    """Validate a bearer token using a constant-time comparison."""

    token = parse_bearer_token(authorization)
    if token is None:
        return False
    return hmac.compare_digest(token, expected_token)
