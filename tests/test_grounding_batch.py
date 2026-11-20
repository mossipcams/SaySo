from __future__ import annotations

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


HOME = GroundingHome(
    entities=(
        _e("Morgan's Bedside Lamp", "light", "Bedroom", ("bedside lamp",)),
        _e("Bedroom Tower Fan", "fan", "Bedroom", ("bedroom fan",)),
        _e("Living Room Fan", "fan", "Living Room"),
    ),
    areas={"Bedroom": (), "Living Room": ()},
    satellite_area="Bedroom",
    satellite_device_id=None,
)

LAMP = ("intent__HassTurnOn", {"domain": ["light"], "name": "Morgan's Bedside Lamp"})
FAN = ("intent__HassTurnOff", {"domain": ["fan"], "name": "Bedroom Tower Fan"})


def test_multi_action_utterance_grounds_both_calls() -> None:
    utterance = "Turn on Morgan's bedside lamp and turn off the fan."
    assert is_grounded(*LAMP, utterance, HOME, others=[FAN])
    assert is_grounded(*FAN, utterance, HOME, others=[LAMP])


def test_unexplained_word_in_a_two_call_batch_still_rejects() -> None:
    utterance = "Turn on Morgan's bedside lamp and turn off the attic fan."
    assert not is_grounded(*FAN, utterance, HOME, others=[LAMP])
    assert not all(
        is_grounded(*call, utterance, HOME, others=[other])
        for call, other in ((LAMP, FAN), (FAN, LAMP))
    )


def test_multi_action_fan_without_batch_context_is_unchanged() -> None:
    utterance = "Turn on Morgan's bedside lamp and turn off the fan."
    assert not is_grounded(*FAN, utterance, HOME)
    assert not is_grounded(*FAN, utterance, HOME, others=())


def test_single_call_grounding_is_identical_with_empty_batch() -> None:
    cases = [
        (
            "intent__HassTurnOff",
            {"domain": ["fan"], "name": "Bedroom Tower Fan"},
            "Turn off the fan.",
        ),
        (
            "intent__HassTurnOff",
            {"domain": ["fan"], "name": "Bedroom Tower Fan"},
            "Turn off the attic fan.",
        ),
        (
            "intent__HassTurnOn",
            {"domain": ["light"], "name": "Morgan's Bedside Lamp"},
            "Turn on Morgan's bedside lamp.",
        ),
    ]
    for tool, args, utterance in cases:
        assert is_grounded(tool, args, utterance, HOME) is is_grounded(
            tool, args, utterance, HOME, others=()
        )
