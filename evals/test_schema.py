"""Evaluation JSONL case schema validation tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from evals.schema import EvalCase, EvalSchemaError, load_eval_cases_jsonl, parse_eval_case


def _valid_action_case(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "case_id": "simple-001",
        "category": "simple_control",
        "home": "eval-home",
        "origin": "area_living_room",
        "turns": ["Turn off the lights"],
        "expected_control_plan": {
            "outcome": "action",
            "intent": "turn off the lights",
            "domain": "light",
            "scope": {"kind": "current_area"},
            "state": "off",
        },
        "expected_candidate_entities": ["light.living_room_ceiling"],
        "expected_resolved_entities": ["light.living_room_ceiling"],
        "expected_outcome": "valid_action",
        "execution_allowed": True,
    }
    base.update(overrides)
    return base


def test_valid_action_case_round_trips() -> None:
    case = EvalCase.model_validate(_valid_action_case())
    assert case.case_id == "simple-001"
    assert case.home == "eval-home"
    assert case.origin == "area_living_room"
    assert case.turns == ["Turn off the lights"]
    assert case.expected_outcome == "valid_action"
    assert case.expected_resolved_entities == ["light.living_room_ceiling"]


def test_incomplete_expected_outcome_missing_field_fails_fast() -> None:
    payload = _valid_action_case()
    del payload["expected_outcome"]
    with pytest.raises(ValidationError):
        EvalCase.model_validate(payload)


def test_incomplete_valid_action_with_empty_plan_fails_fast() -> None:
    with pytest.raises(ValidationError, match="expected_control_plan"):
        EvalCase.model_validate(_valid_action_case(expected_control_plan={}))


def test_incomplete_valid_action_with_empty_resolved_targets_fails_fast() -> None:
    with pytest.raises(ValidationError, match="expected_resolved_entities"):
        EvalCase.model_validate(_valid_action_case(expected_resolved_entities=[]))


def test_valid_action_requires_resolved_targets_in_candidate_set() -> None:
    with pytest.raises(ValidationError, match="expected_candidate_entities"):
        EvalCase.model_validate(
            _valid_action_case(
                expected_candidate_entities=["light.kitchen"],
                expected_resolved_entities=["light.living_room_ceiling"],
            ),
        )


def test_expected_outcome_must_match_control_plan_outcome() -> None:
    with pytest.raises(ValidationError, match="expected_outcome"):
        EvalCase.model_validate(
            _valid_action_case(
                expected_control_plan={
                    "outcome": "clarification",
                    "intent": "turn off the lights",
                    "reason": "ambiguous lamp",
                },
                expected_outcome="valid_action",
                expected_resolved_entities=[],
            ),
        )


def test_clarification_case_allows_empty_target_sets() -> None:
    case = EvalCase.model_validate(
        {
            "case_id": "ambiguity-001",
            "category": "ambiguity",
            "home": "eval-home",
            "origin": "area_living_room",
            "turns": ["Turn on the lamp"],
            "expected_control_plan": {
                "outcome": "clarification",
                "intent": "turn on the lamp",
                "reason": "multiple lamps match",
            },
            "expected_candidate_entities": ["light.living_room_lamp", "light.bedroom_lamp"],
            "expected_resolved_entities": [],
            "expected_outcome": "clarification",
        },
    )
    assert case.expected_outcome == "clarification"


def test_empty_turns_fail_fast() -> None:
    with pytest.raises(ValidationError, match="turns"):
        EvalCase.model_validate(_valid_action_case(turns=[]))


def test_missing_home_or_origin_fail_fast() -> None:
    payload = _valid_action_case()
    del payload["home"]
    with pytest.raises(ValidationError):
        EvalCase.model_validate(payload)

    payload = _valid_action_case()
    del payload["origin"]
    with pytest.raises(ValidationError):
        EvalCase.model_validate(payload)


def test_parse_eval_case_rejects_invalid_control_plan() -> None:
    payload = _valid_action_case(
        expected_control_plan={
            "outcome": "action",
            "intent": "turn off the lights",
            "domain": "light",
            "state": "off",
            "value": 50,
        },
    )
    with pytest.raises(EvalSchemaError, match="line 3"):
        parse_eval_case(json.dumps(payload), line_number=3)


def test_load_eval_cases_jsonl_skips_blank_lines_and_fails_on_invalid_row() -> None:
    valid = json.dumps(_valid_action_case())
    text = f"{valid}\n\n{valid}\nnot-json\n"
    with pytest.raises(EvalSchemaError, match="line 4"):
        load_eval_cases_jsonl(text)


def test_load_eval_cases_jsonl_returns_all_valid_cases() -> None:
    valid = json.dumps(_valid_action_case(case_id="simple-001"))
    second = json.dumps(_valid_action_case(case_id="simple-002"))
    cases = load_eval_cases_jsonl(f"{valid}\n{second}\n")
    assert [case.case_id for case in cases] == ["simple-001", "simple-002"]
