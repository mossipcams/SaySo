"""Config flow tests for SaySo using pytest-homeassistant-custom-component."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.sayso.config_flow import (
    CannotConnect,
    InvalidAuth,
    probe_connection,
)
from custom_components.sayso.const import CONF_TOKEN, CONF_URL, DOMAIN

PROBE_PATH = "custom_components.sayso.config_flow.probe_connection"


async def _init_user_flow(hass: HomeAssistant) -> dict:
    return await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_USER},
    )


@pytest.mark.asyncio
async def test_initial_step_shows_form(hass: HomeAssistant) -> None:
    result = await _init_user_flow(hass)

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"


@pytest.mark.asyncio
async def test_empty_token_shows_error_and_does_not_create_entry(hass: HomeAssistant) -> None:
    result = await _init_user_flow(hass)

    with patch(PROBE_PATH) as mock_probe:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_URL: "http://127.0.0.1:8765",
                CONF_TOKEN: "",
            },
        )

    mock_probe.assert_not_called()
    assert result["type"] == FlowResultType.FORM
    assert CONF_TOKEN in result["errors"]


@pytest.mark.asyncio
async def test_invalid_url_shows_error_and_does_not_create_entry(hass: HomeAssistant) -> None:
    result = await _init_user_flow(hass)

    with patch(PROBE_PATH) as mock_probe:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_URL: "not-a-url",
                CONF_TOKEN: "secret-token",
            },
        )

    mock_probe.assert_not_called()
    assert result["type"] == FlowResultType.FORM
    assert CONF_URL in result["errors"]


@pytest.mark.asyncio
async def test_invalid_token_shows_error_and_does_not_create_entry(hass: HomeAssistant) -> None:
    result = await _init_user_flow(hass)

    with patch(PROBE_PATH, side_effect=InvalidAuth()) as mock_probe:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_URL: "http://127.0.0.1:8765",
                CONF_TOKEN: "bad-token",
            },
        )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "invalid_auth"
    mock_probe.assert_awaited_once_with("http://127.0.0.1:8765", "bad-token")


@pytest.mark.asyncio
async def test_unreachable_server_shows_error_and_does_not_create_entry(hass: HomeAssistant) -> None:
    result = await _init_user_flow(hass)

    with patch(PROBE_PATH, side_effect=CannotConnect()) as mock_probe:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_URL: "http://127.0.0.1:8765",
                CONF_TOKEN: "good-token",
            },
        )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"]["base"] == "cannot_connect"
    mock_probe.assert_awaited_once()


@pytest.mark.asyncio
async def test_valid_credentials_create_entry(hass: HomeAssistant) -> None:
    result = await _init_user_flow(hass)

    with patch(PROBE_PATH, return_value=None) as mock_probe:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_URL: "http://127.0.0.1:8765",
                CONF_TOKEN: "good-token",
            },
        )

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "SaySo"
    assert result["data"] == {
        CONF_URL: "http://127.0.0.1:8765",
        CONF_TOKEN: "good-token",
    }
    mock_probe.assert_awaited_once_with("http://127.0.0.1:8765", "good-token")


@pytest.mark.asyncio
async def test_probe_connection_http_401_raises_invalid_auth() -> None:
    error = HTTPError(
        "http://127.0.0.1:8765/api/v1/health",
        401,
        "Unauthorized",
        {},
        BytesIO(b""),
    )

    with patch("custom_components.sayso.config_flow.urlopen", side_effect=error):
        with pytest.raises(InvalidAuth):
            await probe_connection("http://127.0.0.1:8765", "bad-token")


@pytest.mark.asyncio
async def test_probe_connection_connect_failure_raises_cannot_connect() -> None:
    with patch(
        "custom_components.sayso.config_flow.urlopen",
        side_effect=URLError("connection refused"),
    ):
        with pytest.raises(CannotConnect):
            await probe_connection("http://127.0.0.1:8765", "good-token")
