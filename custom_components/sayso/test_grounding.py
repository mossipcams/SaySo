from __future__ import annotations

import pytest

from custom_components.sayso.const import ERROR_NOT_UNDERSTOOD
from custom_components.sayso.grounding import (
    GroundingEntity,
    GroundingHome,
    clarify,
    is_grounded,
)


def _entity(entity_id: str, name: str, domain: str, area=None, device_id=None):
    return GroundingEntity(
        entity_id=entity_id,
        name=name,
        aliases=(),
        domain=domain,
        area=area,
        device_id=device_id,
    )


@pytest.fixture
def home() -> GroundingHome:
    entities = (
        _entity(
            "light.living_room_lights", "Living room lights", "light", "Living Room"
        ),
        _entity("switch.corner_lamp", "Corner lamp", "switch", "Living Room"),
        _entity("switch.shelf_lamp", "Shelf lamp", "switch", "Living Room"),
        _entity("media_player.tv", "TV", "media_player", "Living Room"),
        _entity("media_player.samsung_tv", "Samsung Tv", "media_player", "Living Room"),
        _entity(
            "media_player.sayso_satellite",
            "SaySo Satellite",
            "media_player",
            "Living Room",
            device_id="sat1",
        ),
        _entity("light.nightstand", "Nightstand", "light", "Bedroom"),
        _entity(
            "switch.humidifier_plugin",
            "Humidifier Plugin",
            "switch",
            "Bedroom",
        ),
        _entity("light.kitchen_light", "Kitchen light", "light", "Kitchen"),
        _entity("light.counter_switch_1234", "Counter Switch-1234", "light", "Kitchen"),
        _entity("light.outdoor_lights", "Outdoor Lights", "light", "Outside"),
        _entity("media_player.party_time", "Party Time", "media_player"),
        _entity("light.guest_lamp", "Guest lamp", "light", "Guest Room"),
    )
    return GroundingHome(
        entities=entities,
        areas={
            "Living Room": (),
            "Bedroom": (),
            "Kitchen": (),
            "Outside": (),
            "Guest Room": (),
        },
        satellite_area="Living Room",
        satellite_device_id="sat1",
    )


@pytest.mark.parametrize(
    ("tool_name", "args", "utterance"),
    [
        (
            "HassTurnOn",
            {"name": "Living room lights"},
            "turn on the living room lights",
        ),
        ("HassTurnOff", {"name": "Living room lights"}, "Turn the living lights off."),
        ("HassTurnOn", {"name": "Living room lights"}, "The room lights on."),
        (
            "media_player__HassSetVolumeRelative",
            {"name": "TV"},
            "turn the tv volume up",
        ),
        ("media_player__HassMediaPause", {"name": "TV"}, "Pause the TV."),
        (
            "HassTurnOn",
            {"area": "Living Room", "domain": ["light"]},
            "turn on the lights",
        ),
        (
            "HassTurnOff",
            {"area": "Bedroom", "domain": ["light"]},
            "turn off the bedroom lights",
        ),
        ("homeassistant__GetLiveContext", {}, "is the tv on"),
        ("intent__HassStartTimer", {"minutes": 10}, "set a ten minute timer"),
        (
            "HassTurnOff",
            {"area": "Guest Room", "domain": ["light"]},
            "turn off the guest room lights",
        ),
        (
            "HassTurnOff",
            {"domain": ["light"], "name": "Counter Switch-1234"},
            "turn the kitchen lights off",
        ),
    ],
)
def test_is_grounded_true(tool_name, args, utterance, home):
    assert is_grounded(tool_name, args, utterance, home) is True


@pytest.mark.parametrize(
    ("tool_name", "args", "utterance", "speech"),
    [
        (
            "HassTurnOn",
            {"domain": ["switch"], "name": "Bedroom Lampshade"},
            "turn the bedroom lights on",
            "Did you mean the Nightstand?",
        ),
        (
            "HassTurnOff",
            {"domain": ["light"], "name": "Counter Switch-1234"},
            "turn off the liver and lights",
            "Did you mean the Living room lights?",
        ),
        (
            "media_player__HassSetVolumeRelative",
            {"name": "SaySo"},
            "Turn the bedroom volume up.",
            "The bedroom has no media players available.",
        ),
        (
            "media_player__HassMediaUnpause",
            {"name": "SaySo"},
            "unpause the tv",
            "Did you mean the TV?",
        ),
        (
            "HassTurnOn",
            {"domain": ["light"], "name": "Wick"},
            "turn the wick and lights on",
            None,
        ),
        (
            "intent__HassTurnOn",
            {"domain": ["light"], "name": "Lamp"},
            "turn on the hospital lamp",
            None,
        ),
        ("HassTurnOff", {"area": "Kitchen"}, "turn off the lights", None),
    ],
)
def test_not_grounded_and_clarify_speech(tool_name, args, utterance, speech, home):
    assert is_grounded(tool_name, args, utterance, home) is False
    if speech is None:
        speech = ERROR_NOT_UNDERSTOOD
    result = clarify(tool_name, args, utterance, home)
    assert result.speech == speech


def test_satellite_never_offered(home):
    result = clarify(
        "media_player__HassMediaUnpause", {"name": "SaySo"}, "unpause the tv", home
    )
    assert all(e.device_id != "sat1" for e in result.options)
    assert result.options == (home.entities[3],)


def test_not_grounded_unnamed_entity(home):
    assert (
        is_grounded("HassTurnOn", {"name": "Wick"}, "turn the wick and lights on", home)
        is False
    )
    result = clarify(
        "HassTurnOn", {"name": "Wick"}, "turn the wick and lights on", home
    )
    assert result.speech == ERROR_NOT_UNDERSTOOD
    assert result.options == ()
