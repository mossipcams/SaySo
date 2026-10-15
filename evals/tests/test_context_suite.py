from __future__ import annotations

from evals.cases import load_gates, load_home, select_cases
from evals.outcomes import Turn
from evals.runner import render_case
from evals.scorer import score


def test_context_suite_selects_eleven_live_home_cases() -> None:
    cases = select_cases(suite="context")
    assert len(cases) == 11
    for case in cases:
        assert case.household == "live_home_1"
        assert "gold" not in case.tags
        assert "shadow" not in case.tags


def test_expected_calls_score_clean() -> None:
    for case in select_cases(suite="context"):
        expectation = render_case(case)["expectation"]
        calls = [dict(call) for call in expectation.calls]
        turn = Turn(raw=None, text="Sorry, I can not do that.", calls=calls)
        result = score(case.id, expectation, turn, home=load_home(case.household))
        assert result.passed is True, case.id
        assert result.flags == [], (case.id, result.flags)


def test_wrong_domain_flag_fires_on_lookalike() -> None:
    case = select_cases(case_id="context_lookalikes_01")[0]
    expectation = render_case(case)["expectation"]
    turn = Turn(
        raw=None,
        text="Sorry, I can not do that.",
        calls=[
            {
                "name": "intent__HassTurnOn",
                "arguments": {"domain": ["sensor"], "name": "TV"},
            }
        ],
    )
    result = score(case.id, expectation, turn, home=load_home(case.household))
    assert result.passed is False
    assert "wrong_domain" in result.flags


def test_context_entity_targeted_flag_fires_on_person() -> None:
    case = select_cases(case_id="context_lookalikes_09")[0]
    expectation = render_case(case)["expectation"]
    turn = Turn(
        raw=None,
        text="Sorry, I can not do that.",
        calls=[{"name": "intent__HassTurnOn", "arguments": {"name": "Person One"}}],
    )
    result = score(case.id, expectation, turn, home=load_home(case.household))
    assert result.passed is False
    assert "context_entity_targeted" in result.flags


def test_context_gate_requires_full_pass() -> None:
    assert load_gates("context")["min_overall_pass_rate"] == 1.0
