"""Offline checks for the Home Assistant home fetcher."""

from __future__ import annotations

import json

import pytest

from scripts.fetch_ha_home import DEFAULT_DOMAINS, _strip_email, build_home

STATES = [
    {
        "entity_id": "light.kitchen_light",
        "state": "on",
        "attributes": {
            "friendly_name": "Kitchen light",
            "supported_color_modes": ["color_temp", "hs"],
        },
    },
    {
        "entity_id": "switch.pond_aerator",
        "state": "off",
        "attributes": {"friendly_name": "Pond aerator", "device_class": "outlet"},
    },
    {
        "entity_id": "button.thermostat_restart",
        "state": "unknown",
        "attributes": {"friendly_name": "Thermostat Restart"},
    },
    {"entity_id": "sensor.cpu_temp", "state": "41", "attributes": {}},
    {
        "entity_id": "light.ghost",
        "state": "unavailable",
        "attributes": {"friendly_name": "Ghost", "restored": True},
    },
    {
        "entity_id": "todo.mattlivesay15_gmail_com_grocery",
        "state": "0",
        "attributes": {"friendly_name": "mattlivesay15@gmail.com Grocery"},
    },
    {
        "entity_id": "media_player.matt_s_kitchen_echo_dot",
        "state": "idle",
        "attributes": {"friendly_name": "Matt's Kitchen Echo Dot"},
    },
    {
        "entity_id": "button.rocky_room_cam_pan_left",
        "state": "unknown",
        "attributes": {"friendly_name": "Rocky room cam Pan left"},
    },
    {
        "entity_id": "timer.pizza",
        "state": "active",
        "attributes": {"friendly_name": "Pizza Timer", "remaining": "0:12:30"},
    },
]
AREA_ROWS = [
    ["light.kitchen_light", "Kitchen", "Main Floor"],
    ["switch.pond_aerator", None, None],
    ["button.thermostat_restart", "Hallway", "Main Floor"],
    ["timer.pizza", "Kitchen", "Main Floor"],
]


@pytest.fixture
def home():
    return build_home(STATES, AREA_ROWS, home_id="test_home")


def test_keeps_only_controllable_domains(home):
    assert {e["entity_id"] for e in home["entities"]} == {
        "light.kitchen_light",
        "switch.pond_aerator",
        "media_player.matt_s_kitchen_echo_dot",
        "todo.grocery",
    }
    assert "button" not in DEFAULT_DOMAINS


def test_buttons_come_back_when_asked():
    home = build_home(STATES, AREA_ROWS, domains=DEFAULT_DOMAINS + ("button",))
    assert "button.thermostat_restart" in {e["entity_id"] for e in home["entities"]}


def test_light_features_derive_from_color_modes(home):
    light = home["entities"][0]
    assert set(light["capabilities"]) == {"on", "off", "brightness", "color", "color_temp"}
    assert light["state"] == "on"
    assert light["area"] == "Kitchen"


def test_entity_ids_and_areas_come_from_home_assistant(home):
    outlet = next(e for e in home["entities"] if e["entity_id"] == "switch.pond_aerator")
    # A slugged name would be switch.pond_aerator too; the point is HA owns it.
    assert outlet["entity_id"] == "switch.pond_aerator"
    assert outlet["area"] == "Unassigned"
    assert outlet["device_class"] == "outlet"
    assert home["sayso_entity_area"] == "Kitchen"


def test_active_timers_convert_to_seconds(home):
    assert home["active_timers"] == [
        {"name": "Pizza Timer", "remaining_seconds": 750, "area": "Kitchen"}
    ]


def test_shape_matches_generate_home():
    import random

    from generators.homes import generate_home

    synthetic = generate_home(0, 8, random.Random(0))
    real = build_home(STATES, AREA_ROWS)
    assert real.keys() == synthetic.keys()
    assert real["entities"][0].keys() == synthetic["entities"][0].keys()


def test_emails_are_stripped_from_names_and_ids(home):
    grocery = next(e for e in home["entities"] if e["capability"] == "todo_lists")
    assert grocery["name"] == "Grocery"
    assert grocery["entity_id"] == "todo.grocery"
    assert "@" not in json.dumps(home)


def test_possessives_and_other_names_survive(home):
    echo = next(e for e in home["entities"] if e["capability"] == "media_players")
    assert echo["name"] == "Matt's Kitchen Echo Dot"
    assert echo["entity_id"] == "media_player.matt_s_kitchen_echo_dot"


def test_an_all_email_name_falls_back_to_the_object_id():
    assert _strip_email("a@b.com", "todo.a_b_com_shopping_list") == (
        "Shopping List",
        "todo.shopping_list",
    )


def test_keep_entity_overrides_the_domain_filter():
    home = build_home(
        STATES, AREA_ROWS, keep_entities=frozenset({"button.rocky_room_cam_pan_left"})
    )
    ids = {e["entity_id"] for e in home["entities"]}
    assert "button.rocky_room_cam_pan_left" in ids
    assert "button.thermostat_restart" not in ids
