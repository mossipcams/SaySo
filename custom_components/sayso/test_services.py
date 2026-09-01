"""Service and translation metadata tests for SaySo."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.helpers.service import _SERVICES_SCHEMA
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import CONF_TOKEN, CONF_URL, DOMAIN, MSG_GRAPH_SNAPSHOT

COMPONENT_DIR = Path(__file__).parent
SERVICES_PATH = COMPONENT_DIR / "services.yaml"
STRINGS_PATH = COMPONENT_DIR / "strings.json"
TRANSLATIONS_PATH = COMPONENT_DIR / "translations" / "en.json"

SERVICE_SYNC_HOME_GRAPH = "sync_home_graph"


def test_services_yaml_parses_with_home_assistant_schema() -> None:
    services = yaml.safe_load(SERVICES_PATH.read_text(encoding="utf-8"))
    parsed = _SERVICES_SCHEMA(services)
    assert SERVICE_SYNC_HOME_GRAPH in parsed


def test_english_strings_include_sync_home_graph_service() -> None:
    strings = json.loads(STRINGS_PATH.read_text(encoding="utf-8"))
    translations = json.loads(TRANSLATIONS_PATH.read_text(encoding="utf-8"))

    for source in (strings, translations):
        service_strings = source["services"][SERVICE_SYNC_HOME_GRAPH]
        assert service_strings["name"]
        assert service_strings["description"]

    assert strings["services"] == translations["services"]


@pytest.mark.asyncio
async def test_setup_registers_sync_home_graph_service(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "good-token"},
    )
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id) is True
    assert hass.services.has_service(DOMAIN, SERVICE_SYNC_HOME_GRAPH)


@pytest.mark.asyncio
async def test_sync_home_graph_sends_graph_snapshot(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "good-token"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id) is True

    coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    for _ in range(50):
        if coordinator.connected:
            break
        await asyncio.sleep(0.01)

    ws = coordinator._ws  # noqa: SLF001 — test inspects outbound socket
    assert ws is not None
    initial_graph_messages = [
        message
        for message in ws.sent
        if json.loads(message).get("type") == MSG_GRAPH_SNAPSHOT
    ]
    assert initial_graph_messages

    await hass.services.async_call(
        DOMAIN,
        SERVICE_SYNC_HOME_GRAPH,
        blocking=True,
    )

    graph_messages = [
        message
        for message in ws.sent
        if json.loads(message).get("type") == MSG_GRAPH_SNAPSHOT
    ]
    assert len(graph_messages) == len(initial_graph_messages) + 1


@pytest.mark.asyncio
async def test_unload_removes_sync_home_graph_service(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "good-token"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id) is True
    assert hass.services.has_service(DOMAIN, SERVICE_SYNC_HOME_GRAPH)

    assert await hass.config_entries.async_unload(entry.entry_id) is True
    assert not hass.services.has_service(DOMAIN, SERVICE_SYNC_HOME_GRAPH)


@pytest.mark.asyncio
async def test_reload_entry_reregisters_service(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "good-token"},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id) is True
    assert hass.services.has_service(DOMAIN, SERVICE_SYNC_HOME_GRAPH)

    assert await hass.config_entries.async_reload(entry.entry_id) is True
    assert entry.state is ConfigEntryState.LOADED
    assert hass.services.has_service(DOMAIN, SERVICE_SYNC_HOME_GRAPH)
    assert entry.entry_id in hass.data.get(DOMAIN, {})
