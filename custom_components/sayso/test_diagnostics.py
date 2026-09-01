"""Diagnostics and secret redaction tests for SaySo."""

from __future__ import annotations

import json

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import CONF_TOKEN, CONF_URL, DOMAIN
from custom_components.sayso.diagnostics import async_get_config_entry_diagnostics

SECRET_TOKEN = "super-secret-access-token-xyz"


def _secret_scan(payload: dict, secret: str) -> None:
    """Fail if a secret appears anywhere in the serialized diagnostics payload."""

    serialized = json.dumps(payload)
    assert secret not in serialized


@pytest.mark.asyncio
async def test_config_entry_diagnostics_includes_health_exposure_and_protocol(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: SECRET_TOKEN},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id) is True

    entity_reg = er.async_get(hass)
    light = entity_reg.async_get_or_create(
        "light",
        "test",
        "kitchen",
        suggested_object_id="kitchen",
    )
    hass.states.async_set(light.entity_id, "on")
    await hass.async_block_till_done()

    diag = await async_get_config_entry_diagnostics(hass, entry)

    assert "health" in diag
    assert "connected" in diag["health"]
    assert "exposure" in diag
    assert "mode" in diag["exposure"]
    assert "entity_count" in diag["exposure"]
    assert "protocol" in diag
    assert "api_version" in diag["protocol"]
    assert "connected" in diag["protocol"]

    _secret_scan(diag, SECRET_TOKEN)
