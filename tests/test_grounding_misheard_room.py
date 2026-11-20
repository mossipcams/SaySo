from __future__ import annotations

from custom_components.sayso.core.grounding import (
    GroundingEntity,
    GroundingHome,
    is_grounded,
)
from evals.cases import load_home
from evals.runner import _grounding_home

HOUSEHOLD_TWO = _grounding_home(load_home("realistic_eval_household_2"))

TURN_OFF_LOUNGE = {
    "domain": ["media_player"],
    "area": "Living Room",
    "name": "Lounge Television",
}

LOUNGE_TV = GroundingEntity(
    "media_player.lounge_television",
    "Lounge Television",
    (),
    "media_player",
    "Living Room",
    None,
)
CORNER_TV = GroundingEntity(
    "media_player.corner_television",
    "Corner Television",
    (),
    "media_player",
    "Living Room",
    None,
)
LIBRARIAN_SPEAKER = GroundingEntity(
    "media_player.librarian_speaker",
    "Librarian Speaker",
    (),
    "media_player",
    "Bedroom",
    None,
)
LIVING_FAN = GroundingEntity(
    "fan.living_room_fan", "Living Room Fan", (), "fan", "Living Room", None
)


def _home(*entities: GroundingEntity) -> GroundingHome:
    return GroundingHome(
        entities=entities,
        areas={"Bedroom": (), "Living Room": ()},
        satellite_area="Living Room",
        satellite_device_id=None,
    )


def test_misheard_room_word_grounds_satellite_tv() -> None:
    assert (
        is_grounded(
            "intent__HassTurnOff",
            TURN_OFF_LOUNGE,
            "Turn off librarian TV.",
            HOUSEHOLD_TWO,
        )
        is True
    )


def test_place_word_before_light_does_not_ground() -> None:
    args = {"domain": ["light"], "name": "Living Room Ceiling Light"}
    assert (
        is_grounded(
            "intent__HassTurnOff", args, "Turn off the patio lights.", HOUSEHOLD_TWO
        )
        is False
    )


def test_place_word_before_fan_does_not_ground() -> None:
    home = _home(LIVING_FAN)
    args = {"domain": ["fan"], "name": "Living Room Fan"}
    assert (
        is_grounded("intent__HassTurnOn", args, "Turn on the attic fan.", home) is False
    )


def test_token_after_device_word_does_not_ground() -> None:
    assert (
        is_grounded(
            "intent__HassTurnOn", TURN_OFF_LOUNGE, "Turn on TV time.", HOUSEHOLD_TWO
        )
        is False
    )


def test_two_unexplained_words_do_not_ground() -> None:
    assert (
        is_grounded(
            "intent__HassTurnOff",
            TURN_OFF_LOUNGE,
            "Turn off librarian reading TV.",
            HOUSEHOLD_TWO,
        )
        is False
    )


def test_satellite_room_with_two_tvs_does_not_ground() -> None:
    home = _home(LOUNGE_TV, CORNER_TV)
    assert (
        is_grounded(
            "intent__HassTurnOff", TURN_OFF_LOUNGE, "Turn off librarian TV.", home
        )
        is False
    )


def test_misheard_word_matching_other_candidate_does_not_ground() -> None:
    home = _home(LOUNGE_TV, LIBRARIAN_SPEAKER)
    assert (
        is_grounded(
            "intent__HassTurnOff", TURN_OFF_LOUNGE, "Turn off librarian TV.", home
        )
        is False
    )
