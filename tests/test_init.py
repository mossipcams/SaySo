"""Tests for SaySo setup and unload."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.setup import async_setup_component

from custom_components.sayso.client import LlamaCppClient
from custom_components.sayso.const import (
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    DOMAIN,
)
from tests.test_config_flow import (
    BASE_URL,
    MODEL_ID,
    _complete_model_step,
    _complete_user_step,
    _start_user_step,
)


@pytest.fixture(autouse=True)
async def setup_llm(hass: HomeAssistant) -> None:
    """Load the LLM integration for config-flow helpers."""
    assert await async_setup_component(hass, "llm", {})


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


async def _create_entry(hass: HomeAssistant) -> Any:
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    return hass.config_entries.async_entries(DOMAIN)[0]


async def test_setup_and_unload_entry(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test config entry setup stores runtime data and unload clears it."""
    entry = await _create_entry(hass)
    await hass.async_block_till_done()

    assert entry.state.value == "loaded"
    assert entry.runtime_data is not None
    assert entry.runtime_data.model == MODEL_ID
    assert entry.runtime_data.client.base_url == BASE_URL

    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state.value == "not_loaded"


async def test_options_reload_updates_runtime_data(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    """Test changing options reloads runtime configuration."""
    entry = await _create_entry(hass)
    await hass.async_block_till_done()
    assert entry.state.value == "loaded"

    result = await hass.config_entries.options.async_init(entry.entry_id)
    new_options = dict(entry.options)
    new_options[CONF_TIMEOUT] = 60
    new_options[CONF_TEMPERATURE] = 0.25
    new_options[CONF_MODEL] = MODEL_ID

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], new_options
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY

    await hass.async_block_till_done()
    assert entry.state.value == "loaded"
    assert entry.runtime_data.client._timeout == 60
    assert entry.runtime_data.temperature == 0.25
