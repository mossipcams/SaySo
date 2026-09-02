"""Tests for SaySo diagnostics."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components.diagnostics import REDACTED
from homeassistant.const import CONF_API_KEY, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.sayso.client import LlamaCppClient
from custom_components.sayso.const import DOMAIN
from custom_components.sayso.diagnostics import async_get_config_entry_diagnostics
from tests.test_config_flow import (
    BASE_URL,
    MODEL_ID,
    _complete_model_step,
    _complete_user_step,
    _start_user_step,
)


@pytest.fixture
def mock_llama_client() -> Any:
    """Patch llama.cpp connectivity checks during setup."""
    with patch.object(
        LlamaCppClient,
        "list_models",
        new=AsyncMock(return_value=[MODEL_ID]),
    ), patch.object(
        LlamaCppClient,
        "validate_model",
        new=AsyncMock(return_value=None),
    ):
        yield


async def _create_entry_with_api_key(
    hass: HomeAssistant,
    api_key: str,
) -> Any:
    result = await _start_user_step(hass)
    result = await _complete_user_step(
        hass, result["flow_id"], api_key=api_key
    )
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    entry = hass.config_entries.async_entries(DOMAIN)[0]
    await hass.async_block_till_done()
    return entry


async def test_api_key_redacted_from_diagnostics(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test API keys are redacted from config entry diagnostics."""
    secret = "super-secret-api-key"
    entry = await _create_entry_with_api_key(hass, secret)

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)
    serialized = json.dumps(diagnostics)

    assert diagnostics["entry"]["data"][CONF_API_KEY] == REDACTED
    assert secret not in serialized
    assert diagnostics["connectivity"]["api_key_configured"] is True
    assert diagnostics["connectivity"]["reachable"] is True
    assert diagnostics["connectivity"]["models"] == [MODEL_ID]
    assert diagnostics["runtime"]["loaded"] is True
    assert diagnostics["runtime"]["model"] == MODEL_ID


async def test_diagnostics_without_api_key(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test diagnostics when no API key is configured."""
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    entry = hass.config_entries.async_entries(DOMAIN)[0]
    await hass.async_block_till_done()

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    assert CONF_API_KEY not in diagnostics["entry"]["data"]
    assert diagnostics["connectivity"]["api_key_configured"] is False
    assert diagnostics["entry"]["data"][CONF_URL] == BASE_URL
