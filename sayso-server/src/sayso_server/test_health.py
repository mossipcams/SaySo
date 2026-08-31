"""Health endpoint auth tests."""

from __future__ import annotations

from sayso_server.health import health_status


def test_missing_bearer_returns_401() -> None:
    assert health_status(authorization=None, token="secret") == 401


def test_wrong_bearer_returns_401() -> None:
    assert health_status(authorization="Bearer wrong", token="secret") == 401


def test_malformed_bearer_returns_401() -> None:
    assert health_status(authorization="Token secret", token="secret") == 401


def test_valid_bearer_returns_200() -> None:
    assert health_status(authorization="Bearer secret", token="secret") == 200
