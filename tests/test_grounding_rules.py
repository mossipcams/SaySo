from __future__ import annotations

import pytest

from custom_components.sayso.core.grounding import (
    GroundingEntity,
    GroundingHome,
    is_grounded,
)


def _e(
    name: str,
    domain: str,
    area: str | None,
    aliases: tuple[str, ...] = (),
) -> GroundingEntity:
    return GroundingEntity(
        entity_id=f"{domain}.{name.casefold().replace(' ', '_')}",
        name=name,
        aliases=aliases,
        domain=domain,
        area=area,
        device_id=None,
    )


def _home(
    entities: list[GroundingEntity],
    satellite: str,
    extra_areas: tuple[str, ...] = (),
) -> GroundingHome:
    areas = {e.area for e in entities if e.area} | {satellite, *extra_areas}
    return GroundingHome(
        entities=tuple(entities),
        areas={area: () for area in sorted(areas)},
        satellite_area=satellite,
        satellite_device_id=None,
    )


@pytest.mark.parametrize(
    ("tool", "args", "utterance", "home"),
    [
        pytest.param(
            "media_player__HassMediaPlayerMute",
            {"domain": ["media_player"], "name": "Sofa TV"},
            "Mute the TV.",
            _home([_e("Sofa TV", "media_player", "Living Room")], "Kitchen"),
            id="A-mute-sole-tv",
        ),
        pytest.param(
            "media_player__HassSetVolumeRelative",
            {"name": "Family TV", "volume_step": "down"},
            "Turn the TV down a bit.",
            _home([_e("Family TV", "media_player", "Living Room")], "Living Room"),
            id="A-volume-sole-tv",
        ),
        pytest.param(
            "intent__HassTurnOn",
            {"domain": ["fan"], "name": "Bedroom Fan"},
            "Turn on the fan.",
            _home([_e("Bedroom Fan", "fan", "Bedroom")], "Kitchen"),
            id="A-turn-on-sole-fan",
        ),
        pytest.param(
            "vacuum__HassVacuumReturnToBase",
            {"domain": ["vacuum"], "name": "Roomba"},
            "Dock the vacuum.",
            _home([_e("Roomba", "vacuum", "Kitchen")], "Bedroom"),
            id="A-dock-sole-vacuum",
        ),
        pytest.param(
            "fan__HassFanSetSpeed",
            {"domain": ["fan"], "name": "Bedroom Tower Fan", "percentage": 25},
            "Set the fan to 25%.",
            _home(
                [
                    _e("Bedroom Tower Fan", "fan", "Bedroom"),
                    _e("Living Room Fan", "fan", "Living Room"),
                ],
                "Bedroom",
            ),
            id="B-satellite-room-fan",
        ),
        pytest.param(
            "intent__HassTurnOn",
            {"area": "Bedroom", "domain": ["fan"], "name": "Bedroom Tower Fan"},
            "Switch the fan on.",
            _home(
                [
                    _e("Bedroom Tower Fan", "fan", "Bedroom"),
                    _e("Living Room Fan", "fan", "Living Room"),
                ],
                "Bedroom",
            ),
            id="B-switch-fan-on",
        ),
        pytest.param(
            "media_player__HassSetVolume",
            {
                "domain": ["media_player"],
                "name": "Living Room Television",
                "volume_level": 30,
            },
            "Set the volume on the television to 30%.",
            _home(
                [_e("Living Room Television", "media_player", "Living Room")], "Bedroom"
            ),
            id="C-television-partial-noun",
        ),
        pytest.param(
            "vacuum__HassVacuumStart",
            {"domain": ["vacuum"], "name": "Downstairs Robot Vacuum"},
            "Start the vacuum.",
            _home([_e("Downstairs Robot Vacuum", "vacuum", "Kitchen")], "Kitchen"),
            id="C-vacuum-partial-name",
        ),
        pytest.param(
            "intent__HassTurnOn",
            {"domain": ["light"], "name": "Living Room Floor Lamp"},
            "Turn on the floor lamp.",
            _home(
                [
                    _e("Living Room Floor Lamp", "light", "Living Room"),
                    _e("Kitchen Light", "light", "Kitchen"),
                    _e("Nightstand", "light", "Bedroom"),
                ],
                "Bedroom",
            ),
            id="C-floor-lamp-partial-name",
        ),
        pytest.param(
            "intent__HassTurnOff",
            {
                "area": "Living Room",
                "domain": ["light"],
                "name": "Sofa Reading Lamp",
            },
            "Switch off the lamp.",
            _home(
                [
                    _e(
                        "Sofa Reading Lamp",
                        "light",
                        "Living Room",
                        aliases=("reading light", "sofa lamp"),
                    ),
                    _e("Living Room Ceiling Light", "light", "Living Room"),
                ],
                "Living Room",
            ),
            id="N-lamp-satellite-narrowing",
        ),
        pytest.param(
            "intent__HassTurnOn",
            {
                "area": "Living Room",
                "domain": ["light"],
                "name": "Living Room Corner Lamp",
            },
            "Switch on the lamp.",
            _home(
                [
                    _e(
                        "Living Room Corner Lamp",
                        "light",
                        "Living Room",
                        aliases=("reading light",),
                    ),
                    _e("Living Room TV Backlight", "light", "Living Room"),
                ],
                "Living Room",
            ),
            id="N-lamp-corner-lamp",
        ),
        pytest.param(
            "vacuum__HassVacuumCleanArea",
            {"area": "Hallway", "name": "Downstairs Robot Vacuum"},
            "Have the vacuum clean the hallway.",
            _home(
                [_e("Downstairs Robot Vacuum", "vacuum", "Kitchen")],
                "Kitchen",
                extra_areas=("Hallway",),
            ),
            id="R-clean-area-hallway",
        ),
        pytest.param(
            "vacuum__HassVacuumCleanArea",
            {"area": "Bathroom", "name": "Downstairs Robot Vacuum"},
            "Vacuum the bathroom.",
            _home(
                [_e("Downstairs Robot Vacuum", "vacuum", "Kitchen")],
                "Kitchen",
                extra_areas=("Bathroom",),
            ),
            id="R-clean-area-bathroom",
        ),
        pytest.param(
            "intent__HassTurnOff",
            {
                "area": "Living Room",
                "domain": ["media_player"],
                "name": "Lounge Television",
            },
            "Turn off librarian TV.",
            _home(
                [
                    _e("Bedroom TV", "media_player", "Bedroom"),
                    _e("Lounge Television", "media_player", "Living Room"),
                ],
                "Living Room",
            ),
            id="misheard-librarian-tv",
        ),
        pytest.param(
            "intent__HassTurnOff",
            {"domain": ["light"], "name": "Living Room Ceiling Light"},
            "Turn off the librarian lights.",
            _home(
                [_e("Living Room Ceiling Light", "light", "Living Room")], "Living Room"
            ),
            id="misheard-librarian-lights",
        ),
    ],
)
def test_rules_ground_expected_call(tool, args, utterance, home):
    assert is_grounded(tool, args, utterance, home) is True


@pytest.mark.parametrize(
    ("tool", "args", "utterance", "home"),
    [
        pytest.param(
            "intent__HassTurnOn",
            {"domain": ["fan"], "name": "Bedroom Ceiling Fan"},
            "Turn on the attic fan.",
            _home([_e("Bedroom Ceiling Fan", "fan", "Bedroom")], "Bedroom"),
            id="modifier-attic-fan",
        ),
        pytest.param(
            "intent__HassTurnOn",
            {"domain": ["media_player"], "name": "Family TV"},
            "Turn on TV time.",
            _home([_e("Family TV", "media_player", "Living Room")], "Living Room"),
            id="modifier-tv-time",
        ),
        pytest.param(
            "intent__HassTurnOn",
            {"domain": ["light"], "name": "Nightstand Lamp"},
            "Turn on Jamie's bedside lamp.",
            _home(
                [
                    _e("Nightstand Lamp", "light", "Bedroom"),
                    _e("Ceiling Light", "light", "Bedroom"),
                ],
                "Bedroom",
            ),
            id="possessive-bedside-lamp",
        ),
        pytest.param(
            "intent__HassTurnOn",
            {"area": "Kitchen", "name": "Bedroom Fan"},
            "Turn on the kitchen fan.",
            _home([_e("Bedroom Fan", "fan", "Bedroom")], "Kitchen"),
            id="room-guard-kitchen-fan",
        ),
        pytest.param(
            "intent__HassTurnOn",
            {"area": "Kitchen", "domain": ["fan"], "name": "Bedroom Fan"},
            "Turn on the fan in the kitchen.",
            _home([_e("Bedroom Fan", "fan", "Bedroom")], "Kitchen"),
            id="room-guard-fan-in-kitchen",
        ),
        pytest.param(
            "intent__HassTurnOff",
            {"area": "Living Room", "domain": ["light"], "name": "Sofa Reading Lamp"},
            "Switch off the lamp.",
            _home(
                [
                    _e("Sofa Reading Lamp", "light", "Living Room"),
                    _e("Corner Lamp", "light", "Living Room"),
                ],
                "Living Room",
            ),
            id="N-two-satellite-lamps-clarify",
        ),
        pytest.param(
            "intent__HassTurnOn",
            {"domain": ["fan"], "name": "Bedroom Fan"},
            "Turn on the fan in the attic.",
            _home([_e("Bedroom Fan", "fan", "Bedroom")], "Kitchen"),
            id="whole-utterance-fan-in-attic",
        ),
        pytest.param(
            "intent__HassTurnOff",
            {"domain": ["media_player"], "name": "Living Room TV"},
            "Turn off the TV in the garage.",
            _home([_e("Living Room TV", "media_player", "Living Room")], "Living Room"),
            id="whole-utterance-tv-in-garage",
        ),
    ],
)
def test_rules_leave_call_ungrounded(tool, args, utterance, home):
    assert is_grounded(tool, args, utterance, home) is False
