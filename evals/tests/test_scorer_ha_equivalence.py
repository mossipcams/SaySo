from __future__ import annotations

from typing import Any

from evals.cases import case_index, load_home
from evals.outcomes import Expectation, Outcome, ResponseType, Turn
from evals.runner import render_case
from evals.scorer import classify, is_refusal, score

LIVE_CONTEXT = "homeassistant__GetLiveContext"
VOLUME_RELATIVE = "media_player__HassSetVolumeRelative"
HOME = load_home("realistic_eval_household_2")
TIMERS_HOME_CASE = "realistic_eval_20260908_timers_04"


def _lookup(**arguments: Any) -> dict[str, Any]:
    return {"name": LIVE_CONTEXT, "arguments": arguments}


def _status_expectation(name: str = "Breakfast Bar Pendants") -> Expectation:
    return Expectation(
        category="status",
        response_type=ResponseType.STATUS,
        calls=({"name": LIVE_CONTEXT, "arguments": {"name": name, "domain": "light"}},),
    )


def _action_expectation(
    *calls: dict[str, Any],
    target_area: str | None = None,
    target_area_source: str | None = None,
) -> Expectation:
    return Expectation(
        category="ordinary",
        response_type=ResponseType.ACTION,
        calls=tuple(calls),
        target_area=target_area,
        target_area_source=target_area_source,
    )


def _classify_turn(
    expectation: Expectation,
    actual: tuple[dict[str, Any], ...],
    home: dict | None = HOME,
) -> Outcome:
    return classify(expectation, Turn(raw=None, calls=list(actual)), home=home)


def _score_turn(
    expectation: Expectation,
    actual: tuple[dict[str, Any], ...],
    home: dict | None = HOME,
):
    return score("case", expectation, Turn(raw=None, calls=list(actual)), home=home)


def test_status_lookup_by_entity_name_passes() -> None:
    actual = (_lookup(name="Breakfast Bar Pendants"),)
    assert _classify_turn(_status_expectation(), actual) == Outcome.VALID_TOOL_CALL


def test_status_lookup_by_alias_passes() -> None:
    actual = (_lookup(name="prep lights"),)
    assert (
        _classify_turn(_status_expectation("Kitchen Worktop Lights"), actual)
        == Outcome.VALID_TOOL_CALL
    )


def test_status_lookup_alias_match_is_case_insensitive() -> None:
    actual = (_lookup(name="PREP LIGHTS"),)
    assert (
        _classify_turn(_status_expectation("Kitchen Worktop Lights"), actual)
        == Outcome.VALID_TOOL_CALL
    )


def test_status_lookup_without_arguments_passes() -> None:
    assert (
        _classify_turn(_status_expectation(), (_lookup(),)) == Outcome.VALID_TOOL_CALL
    )


def test_status_lookup_with_matching_domain_and_area_passes() -> None:
    for domain in ("light", ["light"], ["switch", "light"]):
        actual = (
            _lookup(name="Breakfast Bar Pendants", domain=domain, area="kitchen"),
        )
        assert _classify_turn(_status_expectation(), actual) == Outcome.VALID_TOOL_CALL


def test_status_lookup_naming_other_entity_fails() -> None:
    actual = (_lookup(name="Kitchen Pantry Light"),)
    assert _classify_turn(_status_expectation(), actual) == Outcome.UNEXPECTED_TOOL_CALL


def test_status_lookup_with_wrong_domain_fails() -> None:
    actual = (_lookup(name="Breakfast Bar Pendants", domain=["switch"]),)
    assert _classify_turn(_status_expectation(), actual) == Outcome.UNEXPECTED_TOOL_CALL


def test_status_lookup_with_wrong_area_fails() -> None:
    actual = (_lookup(name="Breakfast Bar Pendants", area="Bedroom"),)
    assert _classify_turn(_status_expectation(), actual) == Outcome.UNEXPECTED_TOOL_CALL


def test_status_lookup_with_unknown_argument_fails() -> None:
    actual = (_lookup(name="Breakfast Bar Pendants", floor="Ground"),)
    assert _classify_turn(_status_expectation(), actual) == Outcome.UNEXPECTED_TOOL_CALL


def test_status_lookup_with_arguments_fails_without_home() -> None:
    actual = (_lookup(name="Breakfast Bar Pendants"),)
    assert (
        _classify_turn(_status_expectation(), actual, None)
        == Outcome.UNEXPECTED_TOOL_CALL
    )


def test_status_action_call_still_fails_and_flags_state_change() -> None:
    expected = _status_expectation()
    actual = (
        _lookup(name="Breakfast Bar Pendants"),
        {
            "name": "intent__HassTurnOff",
            "arguments": {"name": "Breakfast Bar Pendants"},
        },
    )
    result = _score_turn(expected, actual)
    assert not result.passed
    assert "status_state_change" in result.flags
    assert result.outcome == Outcome.UNEXPECTED_TOOL_CALL


def test_status_lookup_score_passes_without_flags() -> None:
    result = _score_turn(
        _status_expectation("Kitchen Worktop Lights"), (_lookup(name="prep lights"),)
    )
    assert result.passed
    assert result.flags == []


def _satellite_expectation(*calls: dict[str, Any]) -> Expectation:
    return _action_expectation(
        *calls, target_area="Kitchen", target_area_source="satellite"
    )


def _pendants_off(**extra: Any) -> dict[str, Any]:
    arguments = {
        "name": "Breakfast Bar Pendants",
        "domain": ["light"],
        "area": "Kitchen",
        **extra,
    }
    return {"name": "intent__HassTurnOff", "arguments": arguments}


def test_satellite_area_omitted_matches_expected_area() -> None:
    expected = _satellite_expectation(_pendants_off())
    actual = {
        "name": "intent__HassTurnOff",
        "arguments": {"name": "Breakfast Bar Pendants", "domain": ["light"]},
    }
    assert _classify_turn(expected, (actual,)) == Outcome.VALID_TOOL_CALL


def test_satellite_area_omitted_still_requires_other_arguments() -> None:
    expected = _satellite_expectation(_pendants_off())
    actual = {
        "name": "intent__HassTurnOff",
        "arguments": {"name": "Breakfast Bar Pendants", "domain": ["switch"]},
    }
    assert _classify_turn(expected, (actual,)) == Outcome.UNEXPECTED_TOOL_CALL


def test_satellite_area_omitted_fails_when_expected_area_is_not_satellite() -> None:
    expected = _satellite_expectation(_pendants_off(area="Bedroom"))
    actual = {
        "name": "intent__HassTurnOff",
        "arguments": {"name": "Breakfast Bar Pendants", "domain": ["light"]},
    }
    assert _classify_turn(expected, (actual,)) == Outcome.UNEXPECTED_TOOL_CALL


def test_explicit_area_case_requires_area() -> None:
    expected = _action_expectation(
        _pendants_off(), target_area="Kitchen", target_area_source="explicit"
    )
    actual = {
        "name": "intent__HassTurnOff",
        "arguments": {"name": "Breakfast Bar Pendants", "domain": ["light"]},
    }
    assert _classify_turn(expected, (actual,)) == Outcome.UNEXPECTED_TOOL_CALL


def test_wrong_explicit_area_fails_and_flags_wrong_target_area() -> None:
    expected = _satellite_expectation(_pendants_off())
    actual = {
        "name": "intent__HassTurnOff",
        "arguments": {
            "name": "Breakfast Bar Pendants",
            "domain": ["light"],
            "area": "Bedroom",
        },
    }
    result = _score_turn(expected, (actual,))
    assert not result.passed
    assert result.outcome == Outcome.UNEXPECTED_TOOL_CALL
    assert "wrong_target_area" in result.flags


def test_render_case_threads_target_area_source() -> None:
    cases = case_index()
    assert (
        render_case(cases[TIMERS_HOME_CASE])["expectation"].target_area_source
        == "explicit"
    )
    satellite = next(
        case
        for case in cases.values()
        if render_case(case)["expectation"].target_area_source == "satellite"
    )
    assert render_case(satellite)["expectation"].target_area_source == "satellite"


def _volume(step: Any) -> dict[str, Any]:
    return {
        "name": VOLUME_RELATIVE,
        "arguments": {"name": "Living Room TV", "volume_step": step},
    }


def _volume_expectation(step: Any) -> Expectation:
    return _action_expectation(_volume(step))


def test_relative_volume_down_matches_negative_step() -> None:
    assert (
        _classify_turn(_volume_expectation("down"), (_volume(-10),))
        == Outcome.VALID_TOOL_CALL
    )


def test_relative_volume_up_matches_positive_step() -> None:
    assert (
        _classify_turn(_volume_expectation("up"), (_volume(25),))
        == Outcome.VALID_TOOL_CALL
    )


def test_relative_volume_opposite_direction_fails() -> None:
    assert (
        _classify_turn(_volume_expectation("down"), (_volume(5),))
        == Outcome.UNEXPECTED_TOOL_CALL
    )
    assert (
        _classify_turn(_volume_expectation("up"), (_volume(-5),))
        == Outcome.UNEXPECTED_TOOL_CALL
    )


def test_relative_volume_zero_matches_neither_direction() -> None:
    assert (
        _classify_turn(_volume_expectation("down"), (_volume(0),))
        == Outcome.UNEXPECTED_TOOL_CALL
    )
    assert (
        _classify_turn(_volume_expectation("up"), (_volume(0),))
        == Outcome.UNEXPECTED_TOOL_CALL
    )


def _clarification(
    text: str,
    candidates: tuple[str, ...] = ("Jamie's Bedside Lamp", "Robin's Bedside Lamp"),
    home: dict | None = HOME,
) -> Outcome:
    expected = Expectation(
        category="ambiguity",
        response_type=ResponseType.CLARIFICATION,
        clarification_candidates=candidates,
    )
    return classify(expected, Turn(raw=None, text=text), home=home)


def test_clarification_naming_distinguishing_words_passes() -> None:
    assert (
        _clarification("Which one, Jamie's or Robin's?") == Outcome.VALID_CLARIFICATION
    )


def test_clarification_naming_exclusive_alias_and_name_word_passes() -> None:
    assert (
        _clarification(
            "Which one, sofa lamp or Robin's?",
            ("Sofa Reading Lamp", "Robin's Bedside Lamp"),
        )
        == Outcome.VALID_CLARIFICATION
    )


def test_clarification_with_only_shared_terms_fails() -> None:
    assert _clarification("Which bedside lamp?") == Outcome.WRONG_RESPONSE_TYPE
    assert (
        _clarification("Which one, the bedside lamp or my bedside light?")
        == Outcome.WRONG_RESPONSE_TYPE
    )


def test_clarification_naming_one_candidate_fails() -> None:
    assert _clarification("Do you mean Jamie's?") == Outcome.WRONG_RESPONSE_TYPE


def test_clarification_without_question_fails() -> None:
    assert _clarification("Jamie's or Robin's.") == Outcome.WRONG_RESPONSE_TYPE


def test_satellite_area_only_call_without_area_fails() -> None:
    expected = _satellite_expectation(
        {
            "name": "intent__HassTurnOff",
            "arguments": {"domain": ["light"], "area": "Kitchen"},
        }
    )
    actual = {"name": "intent__HassTurnOff", "arguments": {"domain": ["light"]}}
    assert _classify_turn(expected, (actual,)) == Outcome.UNEXPECTED_TOOL_CALL


def test_satellite_area_only_call_with_correct_area_passes() -> None:
    expected = _satellite_expectation(
        {
            "name": "intent__HassTurnOff",
            "arguments": {"domain": ["light"], "area": "Kitchen"},
        }
    )
    actual = {
        "name": "intent__HassTurnOff",
        "arguments": {"domain": ["light"], "area": "Kitchen"},
    }
    assert _classify_turn(expected, (actual,)) == Outcome.VALID_TOOL_CALL


LAMP_HOME = load_home("realistic_eval_household_1")


def test_clarification_without_cue_words_passes() -> None:
    assert (
        _clarification("Jamie's or Robin's bedside lamp?")
        == Outcome.VALID_CLARIFICATION
    )
    assert (
        _clarification("Do you want Jamie's or Robin's?") == Outcome.VALID_CLARIFICATION
    )
    assert (
        _clarification(
            "The floor lamp or the table lamp?",
            ("Living Room Floor Lamp", "Living Room Table Lamp"),
            LAMP_HOME,
        )
        == Outcome.VALID_CLARIFICATION
    )


def test_clarification_with_cue_and_distinguishing_words_still_passes() -> None:
    assert (
        _clarification(
            "Do you mean the floor lamp or the table lamp?",
            ("Living Room Floor Lamp", "Living Room Table Lamp"),
            LAMP_HOME,
        )
        == Outcome.VALID_CLARIFICATION
    )


def test_clarification_naming_only_shared_words_with_question_fails() -> None:
    assert _clarification("The lamp or the other lamp?") == Outcome.WRONG_RESPONSE_TYPE


def test_natural_refusal_phrasings_are_refusals() -> None:
    for text in (
        "I couldn't find a garage heater.",
        "I could not find a garage heater.",
        "I can't find a garage heater.",
        "I cannot find a garage heater.",
        "I don't see a garage heater in your home.",
        "I do not see a garage heater in your home.",
        "Sorry, there isn't a garage heater.",
        "There aren't any garage heaters here.",
        "Hmm, I don't know of any garage heater.",
        "No device by that name is available.",
    ):
        assert is_refusal(text), text


def test_affirmative_confirmations_are_not_refusals() -> None:
    for text in ("Done, the heater is on.", "Turned off the lights."):
        assert not is_refusal(text), text
