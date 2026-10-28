from __future__ import annotations

from typing import Any

from evals.outcomes import Expectation, Outcome, ResponseType, Turn
from evals.scorer import classify, score

DEN_HOME = {
    "home_id": "grounding_den_lamp_alias",
    "entities": [
        {
            "entity_id": "light.corner_lamp",
            "name": "Corner Lamp",
            "aliases": ["Corner Lamp", "mood light"],
            "domain": "light",
            "area": "Den",
        },
        {
            "entity_id": "light.ceiling_light",
            "name": "Ceiling Light",
            "aliases": ["Ceiling Light"],
            "domain": "light",
            "area": "Den",
        },
    ],
}

SHARED_HOME = {
    "home_id": "shared_alias",
    "entities": [
        {
            "entity_id": "light.desk_lamp",
            "name": "Desk Lamp",
            "aliases": ["Desk Lamp", "lamp"],
            "domain": "light",
            "area": "Office",
        },
        {
            "entity_id": "light.floor_lamp",
            "name": "Floor Lamp",
            "aliases": ["Floor Lamp", "lamp"],
            "domain": "light",
            "area": "Living Room",
        },
    ],
}

SCRIPT_HOME = {
    "home_id": "script_home",
    "entities": [
        {
            "entity_id": "script.studio_away_script",
            "name": "Studio Away Script",
            "aliases": ["Studio Away Script"],
            "domain": "script",
            "area": None,
        },
        {
            "entity_id": "script.away_mode",
            "name": "Away Mode",
            "aliases": ["Away Mode"],
            "domain": "script",
            "area": None,
        },
        {
            "entity_id": "light.corner_lamp",
            "name": "Corner Lamp",
            "aliases": ["Corner Lamp"],
            "domain": "light",
            "area": "Den",
        },
    ],
}


def _turn_on(name: str, **extra: Any) -> dict[str, Any]:
    return {"name": "intent__HassTurnOn", "arguments": {"name": name, **extra}}


def _expect(*calls: dict[str, Any]) -> Expectation:
    return Expectation(
        category="action",
        response_type=ResponseType.ACTION,
        calls=tuple(calls),
    )


def _classify(
    expected: tuple[dict[str, Any], ...],
    actual: tuple[dict[str, Any], ...],
    home: dict | None,
) -> Outcome:
    turn = Turn(raw={"calls": list(actual)}, calls=list(actual))
    return classify(_expect(*expected), turn, home=home)


def test_alias_name_matches_expected_entity_with_home() -> None:
    expected = {"name": "intent__HassTurnOn", "arguments": {"name": "Corner Lamp", "domain": ["light"]}}
    actual = _turn_on("Mood Light", domain=["light"])
    assert _classify((expected,), (actual,), DEN_HOME) == Outcome.VALID_TOOL_CALL


def test_alias_name_fails_without_home() -> None:
    expected = {"name": "intent__HassTurnOn", "arguments": {"name": "Corner Lamp", "domain": ["light"]}}
    actual = _turn_on("Mood Light", domain=["light"])
    assert _classify((expected,), (actual,), None) == Outcome.UNEXPECTED_TOOL_CALL


def test_alias_match_is_case_insensitive() -> None:
    expected = {"name": "intent__HassTurnOn", "arguments": {"name": "corner lamp", "domain": ["light"]}}
    actual = _turn_on("MOOD LIGHT", domain=["light"])
    assert _classify((expected,), (actual,), DEN_HOME) == Outcome.VALID_TOOL_CALL


def test_alias_match_still_requires_same_entity() -> None:
    expected = {"name": "intent__HassTurnOn", "arguments": {"name": "Corner Lamp", "domain": ["light"]}}
    actual = _turn_on("Ceiling Light", domain=["light"])
    assert _classify((expected,), (actual,), DEN_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_shared_alias_does_not_match_either_entity() -> None:
    desk = {"name": "intent__HassTurnOn", "arguments": {"name": "Desk Lamp", "domain": ["light"]}}
    floor = {"name": "intent__HassTurnOn", "arguments": {"name": "Floor Lamp", "domain": ["light"]}}
    shared = _turn_on("lamp", domain=["light"])
    assert _classify((desk,), (shared,), SHARED_HOME) == Outcome.UNEXPECTED_TOOL_CALL
    assert _classify((floor,), (shared,), SHARED_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_expected_shared_alias_does_not_match_named_entity() -> None:
    shared = {"name": "intent__HassTurnOn", "arguments": {"name": "lamp", "domain": ["light"]}}
    desk = _turn_on("Desk Lamp", domain=["light"])
    assert _classify((shared,), (desk,), SHARED_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_script_call_matches_hass_turn_on_with_script_domain() -> None:
    expected = {"name": "studio_away_script", "arguments": {}}
    for extra in ({}, {"domain": ["script"]}, {"domain": "script"}):
        actual = _turn_on("Studio Away Script", **extra)
        assert _classify((expected,), (actual,), SCRIPT_HOME) == Outcome.VALID_TOOL_CALL


def test_script_call_matches_case_insensitively() -> None:
    expected = {"name": "studio_away_script", "arguments": {}}
    actual = _turn_on("studio away script", domain=["script"])
    assert _classify((expected,), (actual,), SCRIPT_HOME) == Outcome.VALID_TOOL_CALL


def test_script_call_without_home_fails() -> None:
    expected = {"name": "studio_away_script", "arguments": {}}
    actual = _turn_on("Studio Away Script", domain=["script"])
    assert _classify((expected,), (actual,), None) == Outcome.UNEXPECTED_TOOL_CALL


def test_script_call_rejects_non_script_domain() -> None:
    expected = {"name": "studio_away_script", "arguments": {}}
    actual = _turn_on("Studio Away Script", domain=["light"])
    assert _classify((expected,), (actual,), SCRIPT_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_script_call_rejects_extra_arguments() -> None:
    expected = {"name": "studio_away_script", "arguments": {}}
    actual = _turn_on("Studio Away Script", domain=["script"], area="Den")
    assert _classify((expected,), (actual,), SCRIPT_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_script_call_rejects_wrong_script() -> None:
    expected = {"name": "studio_away_script", "arguments": {}}
    actual = _turn_on("Away Mode", domain=["script"])
    assert _classify((expected,), (actual,), SCRIPT_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_script_call_rejects_non_script_entity() -> None:
    expected = {"name": "studio_away_script", "arguments": {}}
    actual = _turn_on("Corner Lamp", domain=["light"])
    assert _classify((expected,), (actual,), SCRIPT_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_script_call_rejects_wrong_tool_name() -> None:
    expected = {"name": "away_mode", "arguments": {}}
    actual = _turn_on("Studio Away Script", domain=["script"])
    assert _classify((expected,), (actual,), SCRIPT_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_script_form_expected_matches_hass_turn_on_actual() -> None:
    expected = _turn_on("Studio Away Script", domain=["script"])
    actual = _turn_on("Studio Away Script", domain=["script"])
    assert _classify((expected,), (actual,), SCRIPT_HOME) == Outcome.VALID_TOOL_CALL


def test_alias_equivalence_counts_each_call() -> None:
    expected = (
        {"name": "intent__HassTurnOn", "arguments": {"name": "Corner Lamp", "domain": ["light"]}},
        {"name": "intent__HassTurnOn", "arguments": {"name": "Ceiling Light", "domain": ["light"]}},
    )
    actual = (
        _turn_on("Mood Light", domain=["light"]),
        _turn_on("Mood Light", domain=["light"]),
    )
    assert _classify(expected, actual, DEN_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_alias_equivalence_accepts_reordered_multi_call() -> None:
    expected = (
        {"name": "intent__HassTurnOn", "arguments": {"name": "Corner Lamp", "domain": ["light"]}},
        {"name": "intent__HassTurnOn", "arguments": {"name": "Ceiling Light", "domain": ["light"]}},
    )
    actual = (
        _turn_on("Ceiling Light", domain=["light"]),
        _turn_on("Mood Light", domain=["light"]),
    )
    assert _classify(expected, actual, DEN_HOME) == Outcome.VALID_MULTI_TOOL_CALL


def test_alias_equivalence_keeps_domain_strict() -> None:
    expected = {"name": "intent__HassTurnOn", "arguments": {"name": "Corner Lamp", "domain": ["light"]}}
    actual = _turn_on("Mood Light", domain=["switch"])
    assert _classify((expected,), (actual,), DEN_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_alias_equivalence_keeps_other_arguments_strict() -> None:
    expected = {"name": "intent__HassTurnOn", "arguments": {"name": "Corner Lamp", "domain": ["light"]}}
    actual = _turn_on("Mood Light", domain=["light"], area="Den")
    assert _classify((expected,), (actual,), DEN_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_alias_equivalence_keeps_tool_name_strict() -> None:
    expected = {"name": "HassTurnOff", "arguments": {"name": "Corner Lamp", "domain": ["light"]}}
    actual = _turn_on("Mood Light", domain=["light"])
    assert _classify((expected,), (actual,), DEN_HOME) == Outcome.UNEXPECTED_TOOL_CALL


def test_score_passes_alias_call_with_home() -> None:
    expected = _expect(
        {"name": "intent__HassTurnOn", "arguments": {"name": "Corner Lamp", "domain": ["light"]}}
    )
    actual = _turn_on("Mood Light", domain=["light"])
    turn = Turn(raw={"calls": [actual]}, calls=[actual])
    result = score("case", expected, turn, home=DEN_HOME)
    assert result.passed
    assert result.flags == []
