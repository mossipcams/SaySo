from __future__ import annotations

from typing import Any

import pytest

from evals.cases import case_index, load_home
from evals.outcomes import Expectation, Outcome, ResponseType, Turn
from evals.runner import render_case
from evals.scorer import classify, score

LOCK_HOME = {
    "home_id": "lock_domain",
    "entities": [
        {
            "entity_id": "lock.front_entrance",
            "name": "Front Entrance Lock",
            "aliases": ["Front Door"],
            "domain": "lock",
            "area": None,
        },
    ],
}


def _turn_on(name: str, **extra: Any) -> dict[str, Any]:
    return {"name": "intent__HassTurnOn", "arguments": {"name": name, **extra}}


def _clarify(candidates: tuple[str, ...]) -> Expectation:
    return Expectation(
        category="clarification",
        response_type=ResponseType.CLARIFICATION,
        clarification_candidates=candidates,
    )


def _classify(expectation: Expectation, text: str) -> Outcome:
    turn = Turn(raw={"text": text}, text=text)
    return classify(expectation, turn)


def test_clarification_offering_wrong_candidates_fails() -> None:
    expectation = _clarify(("Jamie's Bedside Lamp", "Robin's Bedside Lamp"))
    text = "Did you mean the Bedroom Desk Fan or the Bedroom Window Shade?"
    assert _classify(expectation, text) == Outcome.WRONG_RESPONSE_TYPE


def test_clarification_offering_expected_candidates_passes() -> None:
    expectation = _clarify(("Jamie's Bedside Lamp", "Robin's Bedside Lamp"))
    text = (
        "Did you mean Jamie's bedside lamp or Robin's bedside lamp? "
        "Which one should I turn off?"
    )
    assert _classify(expectation, text) == Outcome.VALID_CLARIFICATION


def test_clarification_without_candidates_keeps_existing_behaviour() -> None:
    expectation = _clarify(())
    assert (
        _classify(expectation, "Which lamp did you mean?")
        == Outcome.VALID_CLARIFICATION
    )
    assert (
        _classify(expectation, "Turning the lamp off.") == Outcome.WRONG_RESPONSE_TYPE
    )


def test_redundant_correct_domain_is_accepted() -> None:
    expected = _turn_on("Front Entrance Lock")
    actual = _turn_on("Front Entrance Lock", domain=["lock"])
    expectation = Expectation(
        category="action",
        response_type=ResponseType.ACTION,
        calls=(expected,),
    )
    turn = Turn(raw={"calls": [actual]}, calls=[actual])
    assert classify(expectation, turn, home=LOCK_HOME) == Outcome.VALID_TOOL_CALL


def test_wrong_domain_still_fails() -> None:
    expected = _turn_on("Front Entrance Lock")
    actual = _turn_on("Front Entrance Lock", domain=["light"])
    expectation = Expectation(
        category="action",
        response_type=ResponseType.ACTION,
        calls=(expected,),
    )
    turn = Turn(raw={"calls": [actual]}, calls=[actual])
    assert classify(expectation, turn, home=LOCK_HOME) == Outcome.UNEXPECTED_TOOL_CALL


@pytest.mark.parametrize("name", ["FRONT ENTRANCE LOCK", "front door"])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_redundant_domain_without_entity_id(name: str, ambiguous: bool) -> None:
    entity = {
        "name": "Front Entrance Lock",
        "aliases": ["Front Door", "front door"],
        "domain": "lock",
        "area": "Entryway",
    }
    home = {"entities": [entity]}
    if ambiguous:
        home["entities"].append({**entity, "area": "Kitchen"})
    expectation = Expectation(
        category="action",
        response_type=ResponseType.ACTION,
        calls=(_turn_on(name),),
    )
    turn = Turn(raw=None, calls=[_turn_on(name, domain=["lock"])])
    outcome = classify(expectation, turn, home=home)
    assert outcome == (
        Outcome.UNEXPECTED_TOOL_CALL if ambiguous else Outcome.VALID_TOOL_CALL
    )


@pytest.mark.parametrize("domain, passed", [("lock", True), ("light", False)])
def test_real_home_multi_action_redundant_domain(domain: str, passed: bool) -> None:
    case = case_index()["realistic_eval_20260908_multi_action_10"]
    home = load_home(case.household)
    for name in ("Front Entrance Lock", "Taylor's Bedside Lamp"):
        entity = next(entity for entity in home["entities"] if entity["name"] == name)
        assert "entity_id" not in entity
    expectation = render_case(case)["expectation"]
    turn = Turn(
        raw=None,
        calls=[
            _turn_on("Front Entrance Lock", domain=[domain]),
            {
                "name": "intent__HassTurnOff",
                "arguments": {"name": "Taylor's Bedside Lamp", "domain": ["light"]},
            },
        ],
    )
    result = score(case.id, expectation, turn, home=home)
    assert result.passed is passed
    assert result.outcome == (
        Outcome.VALID_MULTI_TOOL_CALL if passed else Outcome.UNEXPECTED_TOOL_CALL
    )
    assert result.flags == ([] if passed else ["wrong_domain"])
