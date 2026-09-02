"""Pytest fixtures for SaySo tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import ClientTimeout
from homeassistant.core import HomeAssistant

from custom_components.sayso.client import LlamaCppClient


@pytest.fixture
def mock_session() -> aiohttp.ClientSession:
    """Provide a mock aiohttp session for HTTP-boundary tests."""
    session = MagicMock(spec=aiohttp.ClientSession)
    session.post = MagicMock()
    return session


@pytest.fixture
def llama_client(mock_session: aiohttp.ClientSession) -> LlamaCppClient:
    """Provide a LlamaCppClient backed by a mock session."""
    return LlamaCppClient(
        mock_session,
        "http://127.0.0.1:8080/v1",
        api_key="test-key",
        timeout=30,
    )


@pytest.fixture
def configure_post(
    mock_session: aiohttp.ClientSession,
) -> Callable[..., AsyncMock]:
    """Configure mock_session.post to return a context-managed response."""

    def _configure(
        *,
        status: int = 200,
        json_body: Any = None,
        text_body: str | None = None,
        side_effect: Exception | None = None,
    ) -> AsyncMock:
        response = AsyncMock()
        response.status = status
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)

        if side_effect is not None:
            mock_session.post.side_effect = side_effect
            return mock_session.post

        if json_body is not None:
            response.json = AsyncMock(return_value=json_body)
        elif text_body is not None:
            response.json = AsyncMock(
                side_effect=aiohttp.ContentTypeError(
                    response.request_info,
                    response.history,
                    message="invalid json",
                )
            )
            response.text = AsyncMock(return_value=text_body)
        else:
            response.json = AsyncMock(return_value={})

        context_manager = AsyncMock()
        context_manager.__aenter__ = AsyncMock(return_value=response)
        context_manager.__aexit__ = AsyncMock(return_value=False)
        mock_session.post.return_value = context_manager
        return mock_session.post

    return _configure


@pytest.fixture
async def hass(
    hass: HomeAssistant,
) -> AsyncGenerator[HomeAssistant, None]:
    """Expose Home Assistant for tests that need from_hass()."""
    yield hass
