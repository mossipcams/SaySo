"""Tests for SaySo config and options flows."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch
import pytest
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_MODEL, CONF_URL
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.setup import async_setup_component

from custom_components.sayso.client import LlamaCppClient
from custom_components.sayso.const import (
    CONF_MAX_OUTPUT_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DOMAIN,
)
from custom_components.sayso.exceptions import (
    SaySoAuthError,
    SaySoConnectionError,
    SaySoInvalidResponseError,
    SaySoModelNotFoundError,
)

BASE_URL = "http://127.0.0.1:8080/v1"
MODEL_ID = "test-model"


@pytest.fixture(autouse=True)
async def setup_llm(hass: HomeAssistant) -> None:
    """Load the LLM integration so API selectors can be populated."""
    assert await async_setup_component(hass, "llm", {})


@pytest.fixture
def mock_list_models() -> Any:
    """Patch LlamaCppClient.list_models."""
    with patch.object(
        LlamaCppClient,
        "list_models",
        new=AsyncMock(return_value=[MODEL_ID]),
    ) as mocked:
        yield mocked


@pytest.fixture
def mock_validate_model() -> Any:
    """Patch LlamaCppClient.validate_model."""
    with patch.object(
        LlamaCppClient,
        "validate_model",
        new=AsyncMock(return_value=None),
    ) as mocked:
        yield mocked


async def _start_user_step(hass: HomeAssistant) -> dict[str, Any]:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"
    return result


async def _complete_user_step(
    hass: HomeAssistant,
    flow_id: str,
    *,
    base_url: str = BASE_URL,
    api_key: str | None = None,
) -> dict[str, Any]:
    user_input: dict[str, Any] = {CONF_URL: base_url}
    if api_key is not None:
        user_input[CONF_API_KEY] = api_key
    return await hass.config_entries.flow.async_configure(flow_id, user_input)


async def _complete_model_step(
    hass: HomeAssistant,
    flow_id: str,
    *,
    model: str = MODEL_ID,
) -> dict[str, Any]:
    return await hass.config_entries.flow.async_configure(
        flow_id, {CONF_MODEL: model}
    )


async def test_successful_config_flow(
    hass: HomeAssistant,
    mock_list_models: AsyncMock,
    mock_validate_model: AsyncMock,
) -> None:
    """Test a successful config flow creates one entry per endpoint and model."""
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "model"

    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == f"{MODEL_ID} @ 127.0.0.1:8080"
    assert result["data"] == {CONF_URL: BASE_URL}
    assert result["options"][CONF_MODEL] == MODEL_ID
    assert result["options"][CONF_TIMEOUT] == DEFAULT_TIMEOUT
    assert result["options"][CONF_TEMPERATURE] == DEFAULT_TEMPERATURE
    assert result["options"][CONF_MAX_OUTPUT_TOKENS] == DEFAULT_MAX_OUTPUT_TOKENS
    assert result["options"][CONF_MAX_TOOL_ITERATIONS] == DEFAULT_MAX_TOOL_ITERATIONS
    assert result["options"]["prompt"] == DEFAULT_SYSTEM_PROMPT

    entry = hass.config_entries.async_entries(DOMAIN)[0]
    assert entry.unique_id == f"{BASE_URL}|{MODEL_ID}"
    mock_list_models.assert_awaited()
    mock_validate_model.assert_awaited()


async def test_duplicate_entry_prevention(
    hass: HomeAssistant,
    mock_list_models: AsyncMock,
    mock_validate_model: AsyncMock,
) -> None:
    """Test duplicate endpoint and model combinations are rejected."""
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY

    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_unreachable_server(hass: HomeAssistant) -> None:
    """Test unreachable llama.cpp surfaces a connection error."""
    with patch.object(
        LlamaCppClient,
        "list_models",
        new=AsyncMock(side_effect=SaySoConnectionError("llama.cpp is unreachable")),
    ):
        result = await _start_user_step(hass)
        result = await _complete_user_step(hass, result["flow_id"])

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": "cannot_connect"}


async def test_invalid_api_key(hass: HomeAssistant) -> None:
    """Test invalid API key surfaces an authentication error."""
    with patch.object(
        LlamaCppClient,
        "list_models",
        new=AsyncMock(side_effect=SaySoAuthError("llama.cpp rejected the API key")),
    ):
        result = await _start_user_step(hass)
        result = await _complete_user_step(
            hass, result["flow_id"], api_key="secret-key"
        )

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": "invalid_auth"}


async def test_invalid_response(hass: HomeAssistant) -> None:
    """Test invalid llama.cpp responses surface a response error."""
    with patch.object(
        LlamaCppClient,
        "list_models",
        new=AsyncMock(
            side_effect=SaySoInvalidResponseError("llama.cpp returned invalid JSON")
        ),
    ):
        result = await _start_user_step(hass)
        result = await _complete_user_step(hass, result["flow_id"])

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"] == {"base": "invalid_response"}


async def test_missing_model(
    hass: HomeAssistant,
    mock_list_models: AsyncMock,
) -> None:
    """Test selecting a missing model surfaces a model error."""
    with patch.object(
        LlamaCppClient,
        "validate_model",
        new=AsyncMock(
            side_effect=SaySoModelNotFoundError("Model 'missing' is not available")
        ),
    ):
        result = await _start_user_step(hass)
        result = await _complete_user_step(hass, result["flow_id"])
        result = await _complete_model_step(hass, result["flow_id"], model="missing")

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "model"
    assert result["errors"] == {"base": "model_not_found"}
    mock_list_models.assert_awaited()


async def test_api_key_redacted_from_logged_errors(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test API keys are redacted from config-flow debug logs."""
    secret = "super-secret-key"
    with patch.object(
        LlamaCppClient,
        "list_models",
        new=AsyncMock(
            side_effect=SaySoAuthError(f"Bearer {secret} rejected")
        ),
    ):
        result = await _start_user_step(hass)
        await _complete_user_step(hass, result["flow_id"], api_key=secret)

    assert secret not in caplog.text
    assert "***" in caplog.text


async def test_options_flow_updates_and_reloads(
    hass: HomeAssistant,
    mock_list_models: AsyncMock,
    mock_validate_model: AsyncMock,
) -> None:
    """Test options updates reload the config entry."""
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY

    entry = hass.config_entries.async_entries(DOMAIN)[0]
    await hass.async_block_till_done()
    assert entry.state.value == "loaded"
    assert entry.runtime_data.model == MODEL_ID
    assert entry.runtime_data.client._timeout == DEFAULT_TIMEOUT

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.FORM

    new_options = dict(entry.options)
    new_options[CONF_TIMEOUT] = 45
    new_options[CONF_TEMPERATURE] = 0.5

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], new_options
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY

    await hass.async_block_till_done()
    assert entry.state.value == "loaded"
    assert entry.options[CONF_TIMEOUT] == 45
    assert entry.runtime_data.client._timeout == 45
    assert entry.runtime_data.temperature == 0.5
