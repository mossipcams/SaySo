"""Registry-to-contract Home Graph snapshot tests."""

from __future__ import annotations

import enum
import json
from datetime import datetime
from pathlib import Path

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers.entity_registry import RegistryEntryDisabler
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import CONF_TOKEN, CONF_URL, DOMAIN, get_entry_options
from custom_components.sayso.snapshot import build_home_graph_snapshot, _serialize_state
from sayso_server.home_graph import HomeGraphSnapshot

FIXTURES = Path(__file__).resolve().parents[2] / "evals" / "fixtures"


@pytest.mark.asyncio
async def test_registry_snapshot_includes_floors_aliases_disabled_and_excluded(
    hass: HomeAssistant,
) -> None:
    """Snapshot must include floors, aliases, and disabled entities in exposed areas."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret"},
        options={
            "domain_allowlist": [],
            "action_allowlist": [],
            "exposure_mode": "area",
            "area_ids": ["living_room"],
            "entity_ids": [],
        },
    )
    entry.add_to_hass(hass)
    options = get_entry_options(entry)

    floor_reg = fr.async_get(hass)
    ground = floor_reg.async_create(
        "Ground Floor",
        aliases={"downstairs", "first floor"},
    )

    area_reg = ar.async_get(hass)
    living_room = area_reg.async_create(
        "Living Room",
        floor_id=ground.floor_id,
        aliases={"lounge", "family room"},
    )
    garage = area_reg.async_create("Garage", floor_id=ground.floor_id)

    entity_reg = er.async_get(hass)
    kitchen_light = entity_reg.async_get_or_create(
        "light",
        "test",
        "kitchen_light",
        suggested_object_id="kitchen",
        original_name="Kitchen Light",
    )
    entity_reg.async_update_entity(
        kitchen_light.entity_id,
        area_id=living_room.id,
        aliases=["kitchen lamp", "overhead"],
    )

    disabled_switch = entity_reg.async_get_or_create(
        "switch",
        "test",
        "disabled_outlet",
        suggested_object_id="disabled_outlet",
        original_name="Disabled Outlet",
        disabled_by=RegistryEntryDisabler.INTEGRATION,
    )
    entity_reg.async_update_entity(
        disabled_switch.entity_id,
        area_id=living_room.id,
    )

    garage_sensor = entity_reg.async_get_or_create(
        "sensor",
        "test",
        "garage_temp",
        suggested_object_id="garage_temp",
        original_name="Garage Temperature",
    )
    entity_reg.async_update_entity(
        garage_sensor.entity_id,
        area_id=garage.id,
    )

    evening_scene = entity_reg.async_get_or_create(
        "scene",
        "test",
        "evening",
        suggested_object_id="evening",
        original_name="Evening",
    )
    entity_reg.async_update_entity(
        evening_scene.entity_id,
        area_id=living_room.id,
    )

    bedtime_script = entity_reg.async_get_or_create(
        "script",
        "test",
        "bedtime",
        suggested_object_id="bedtime",
        original_name="Bedtime",
    )
    entity_reg.async_update_entity(
        bedtime_script.entity_id,
        area_id=living_room.id,
    )

    hass.states.async_set(kitchen_light.entity_id, "on", {"brightness": 200})
    hass.states.async_set(disabled_switch.entity_id, "off")
    hass.states.async_set(garage_sensor.entity_id, "72", {"unit_of_measurement": "°F"})
    hass.states.async_set(evening_scene.entity_id, "scening")
    hass.states.async_set(bedtime_script.entity_id, "off")

    snapshot = build_home_graph_snapshot(
        hass,
        home_id="registry-test-home",
        sequence=7,
        options=options,
    )

    expected = json.loads((FIXTURES / "registry_snapshot.json").read_text())
    expected["entities"] = [
        entity
        for entity in expected["entities"]
        if entity["entity_id"] != garage_sensor.entity_id
    ]
    assert snapshot == expected

    validated = HomeGraphSnapshot.model_validate(snapshot)
    assert validated.version == 1
    assert validated.sequence == 7
    assert len(validated.floors) == 1
    assert {entity.entity_id for entity in validated.entities} == {
        "light.kitchen",
        "switch.disabled_outlet",
    }
    assert {scene.entity_id for scene in validated.scenes} == {"scene.evening"}
    assert {script.entity_id for script in validated.scripts} == {"script.bedtime"}


@pytest.mark.asyncio
async def test_snapshot_state_attributes_are_json_serializable(
    hass: HomeAssistant,
) -> None:
    """Datetime, set, and enum attributes must not abort snapshot send."""

    class SampleMode(enum.Enum):
        AUTO = "auto"

    hass.states.async_set(
        "sensor.sample",
        "ok",
        {
            "updated_at": datetime(2026, 3, 10, 12, 30, 0),
            "tags": {"alpha", "beta"},
            "mode": SampleMode.AUTO,
        },
    )

    serialized = _serialize_state(hass.states.get("sensor.sample"))
    json.dumps(serialized)
    assert serialized["attributes"]["updated_at"] == "2026-03-10T12:30:00"
    assert serialized["attributes"]["tags"] == ["alpha", "beta"]
    assert serialized["attributes"]["mode"] == "auto"


@pytest.mark.asyncio
async def test_snapshot_omits_bulky_forecast_attribute(hass: HomeAssistant) -> None:
    """Weather forecast arrays must not bloat the Home Graph snapshot."""

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret"},
        options={
            "domain_allowlist": [],
            "action_allowlist": [],
            "exposure_mode": "all",
            "area_ids": [],
            "entity_ids": [],
        },
    )
    entry.add_to_hass(hass)
    options = get_entry_options(entry)

    entity_reg = er.async_get(hass)
    weather = entity_reg.async_get_or_create(
        "weather",
        "test",
        "home",
        suggested_object_id="home",
        original_name="Home Weather",
    )
    hass.states.async_set(
        weather.entity_id,
        "sunny",
        {
            "temperature": 72,
            "forecast": [{"datetime": "2026-03-10T12:00:00", "temperature": 70}],
        },
    )

    snapshot = build_home_graph_snapshot(
        hass,
        home_id="weather-home",
        sequence=1,
        options=options,
    )
    entity = next(item for item in snapshot["entities"] if item["entity_id"] == weather.entity_id)
    assert "forecast" not in entity["state"]["attributes"]
    assert entity["state"]["attributes"]["temperature"] == 72
    json.dumps(snapshot)
