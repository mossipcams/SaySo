from __future__ import annotations

from evals.cases import load_gates

from evals.cases import load_gates
from evals.outcomes import Expectation, ResponseType, Turn
from evals.scorer import score

HOME = {
    "entities": [
        {"name": "TV", "domain": "media_player", "aliases": []},
        {"name": "Living room lights", "domain": "switch", "aliases": ["living room"]},
    ],
    "context_entities": [
        {"name": "TV Active app", "domain": "sensor"},
        {"name": "Person One", "domain": "person"},
    ],
}


def make_case(call: dict, response_type: ResponseType) -> tuple[Expectation, Turn]:
    expectation = Expectation(
        category="media", response_type=response_type, calls=(call,)
    )
    turn = Turn(raw={"calls": [call]}, calls=[dict(call)])
    return expectation, turn


def test_wrong_domain_fires_for_action_call() -> None:
    call = {"name": "HassTurnOn", "arguments": {"name": "TV", "domain": ["sensor"]}}
    expectation, turn = make_case(call, ResponseType.ACTION)
    result = score("c1", expectation, turn, home=HOME)
    assert "wrong_domain" in result.flags


def test_wrong_domain_fires_for_status_call() -> None:
    call = {"name": "GetLiveContext", "arguments": {"name": "TV", "domain": ["sensor"]}}
    expectation, turn = make_case(call, ResponseType.STATUS)
    result = score("c2", expectation, turn, home=HOME)
    assert "wrong_domain" in result.flags


def test_no_wrong_domain_when_domain_matches() -> None:
    call = {
        "name": "HassTurnOn",
        "arguments": {"name": "TV", "domain": ["media_player"]},
    }
    expectation, turn = make_case(call, ResponseType.ACTION)
    result = score("c3", expectation, turn, home=HOME)
    assert "wrong_domain" not in result.flags


def test_no_wrong_domain_when_absent() -> None:
    call = {"name": "HassTurnOn", "arguments": {"name": "TV"}}
    expectation, turn = make_case(call, ResponseType.ACTION)
    result = score("c4", expectation, turn, home=HOME)
    assert "wrong_domain" not in result.flags


def test_context_entity_targeted_fires() -> None:
    call = {"name": "HassTurnOn", "arguments": {"name": "Person One"}}
    expectation, turn = make_case(call, ResponseType.ACTION)
    result = score("c5", expectation, turn, home=HOME)
    assert "context_entity_targeted" in result.flags


def test_no_flags_when_home_is_none() -> None:
    call = {"name": "HassTurnOn", "arguments": {"name": "TV", "domain": ["sensor"]}}
    expectation, turn = make_case(call, ResponseType.ACTION)
    result = score("c6", expectation, turn)
    assert result.flags == []


def test_flagged_case_fails() -> None:
    call = {"name": "HassTurnOn", "arguments": {"name": "TV", "domain": ["sensor"]}}
    expectation, turn = make_case(call, ResponseType.ACTION)
    result = score("c7", expectation, turn, home=HOME)
    assert result.passed is False


def test_context_suite_gates() -> None:
    gates = load_gates("context")
    assert gates["min_overall_pass_rate"] == 1.0
    for suite in ("promotion", "smoke", "context"):
        flags = set(load_gates(suite)["zero_tolerance_flags"])
        assert {"wrong_domain", "context_entity_targeted"} <= flags, suite
