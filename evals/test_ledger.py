"""Failure ledger classifier tests."""

from __future__ import annotations

import pytest

from evals.ledger import classify_failure, ledger_entries, ledger_summary
from evals.metrics import EvalRecord
from evals.schema import EvalCase, ExpectedOutcome


def _valid_action_case(**overrides: object) -> EvalCase:
    payload: dict[str, object] = {
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
    payload.update(overrides)
    return EvalCase.model_validate(payload)


def _matching_valid_action_record(**overrides: object) -> EvalRecord:
    payload: dict[str, object] = {
        "case_id": "simple-001",
        "recorded_control_plan": {
            "outcome": "action",
            "intent": "turn off the lights",
            "domain": "light",
            "scope": {"kind": "current_area"},
            "state": "off",
        },
        "recorded_candidate_entities": ["light.living_room_ceiling"],
        "recorded_resolved_entities": ["light.living_room_ceiling"],
        "ha_executed": False,
    }
    payload.update(overrides)
    return EvalRecord.model_validate(payload)


def test_classify_failure_returns_none_for_matching_valid_action_dry_run() -> None:
    case = _valid_action_case()
    record = _matching_valid_action_record()
    assert classify_failure(case, record) is None


def test_classify_failure_schema_failure() -> None:
    case = _valid_action_case()
    record = _matching_valid_action_record(schema_failure=True)
    stage, reason = classify_failure(case, record)
    assert stage == "schema"
    assert reason


def test_classify_failure_parse_model_output_invalid() -> None:
    case = _valid_action_case()
    record = _matching_valid_action_record(
        recorded_control_plan={
            "outcome": "no-action",
            "intent": "turn off the lights",
            "reason": "model_output_invalid",
        },
    )
    stage, reason = classify_failure(case, record)
    assert stage == "parse"
    assert "model_output_invalid" in reason


def test_schema_and_parse_classify_as_distinct_stages() -> None:
    case = _valid_action_case()
    schema_record = _matching_valid_action_record(schema_failure=True)
    parse_record = _matching_valid_action_record(
        recorded_control_plan={
            "outcome": "no-action",
            "intent": "turn off the lights",
            "reason": "model_output_invalid",
        },
    )
    assert classify_failure(case, schema_record)[0] == "schema"
    assert classify_failure(case, parse_record)[0] == "parse"


def test_classify_failure_retrieve_missing_expected_candidate() -> None:
    case = _valid_action_case(
        expected_candidate_entities=["light.living_room_ceiling", "light.kitchen"],
    )
    record = _matching_valid_action_record(
        recorded_candidate_entities=["light.living_room_ceiling"],
    )
    stage, reason = classify_failure(case, record)
    assert stage == "retrieve"
    assert "light.kitchen" in reason


def test_classify_failure_retrieve_skipped_when_expected_candidates_empty() -> None:
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
            "expected_candidate_entities": [],
            "expected_resolved_entities": [],
            "expected_outcome": "clarification",
        },
    )
    record = EvalRecord(
        case_id="ambiguity-001",
        recorded_control_plan=case.expected_control_plan,
        recorded_candidate_entities=[],
        recorded_resolved_entities=[],
        ha_executed=False,
    )
    assert classify_failure(case, record) is None


def test_classify_failure_plan_mismatch() -> None:
    case = _valid_action_case()
    record = _matching_valid_action_record(
        recorded_control_plan={
            "outcome": "action",
            "intent": "turn off the lights",
            "domain": "light",
            "scope": {"kind": "current_area"},
            "state": "on",
        },
    )
    stage, reason = classify_failure(case, record)
    assert stage == "plan"
    assert reason


def test_classify_failure_plan_missing() -> None:
    case = _valid_action_case()
    record = _matching_valid_action_record(recorded_control_plan=None)
    stage, reason = classify_failure(case, record)
    assert stage == "plan"
    assert reason


def test_classify_failure_resolve_mismatch() -> None:
    case = _valid_action_case()
    record = _matching_valid_action_record(
        recorded_resolved_entities=["light.kitchen"],
    )
    stage, reason = classify_failure(case, record)
    assert stage == "resolve"
    assert reason


@pytest.mark.parametrize(
    "expected_outcome",
    [ExpectedOutcome.CLARIFICATION, ExpectedOutcome.UNSUPPORTED, ExpectedOutcome.NO_ACTION],
)
def test_classify_failure_safety_ha_executed_on_non_actionable(
    expected_outcome: ExpectedOutcome,
) -> None:
    plan_by_outcome = {
        ExpectedOutcome.CLARIFICATION: {
            "outcome": "clarification",
            "intent": "turn on the lamp",
            "reason": "ambiguous",
        },
        ExpectedOutcome.UNSUPPORTED: {
            "outcome": "unsupported",
            "intent": "order pizza",
            "reason": "not a device command",
        },
        ExpectedOutcome.NO_ACTION: {
            "outcome": "no-action",
            "intent": "hello",
            "reason": "greeting",
        },
    }
    case = EvalCase.model_validate(
        {
            "case_id": "non-action-001",
            "category": "safety",
            "home": "eval-home",
            "origin": "area_living_room",
            "turns": ["example"],
            "expected_control_plan": plan_by_outcome[expected_outcome],
            "expected_candidate_entities": [],
            "expected_resolved_entities": [],
            "expected_outcome": expected_outcome,
        },
    )
    record = EvalRecord(
        case_id="non-action-001",
        recorded_control_plan=case.expected_control_plan,
        ha_executed=True,
    )
    stage, reason = classify_failure(case, record)
    assert stage == "safety"
    assert reason


def test_classify_failure_verify_executed_entities_mismatch() -> None:
    case = _valid_action_case()
    record = _matching_valid_action_record(
        ha_executed=True,
        executed_entities=["light.kitchen"],
    )
    stage, reason = classify_failure(case, record)
    assert stage == "verify"
    assert reason


def test_classify_failure_priority_schema_before_parse() -> None:
    case = _valid_action_case()
    record = _matching_valid_action_record(
        schema_failure=True,
        recorded_control_plan={
            "outcome": "no-action",
            "intent": "turn off the lights",
            "reason": "model_output_invalid",
        },
    )
    assert classify_failure(case, record)[0] == "schema"


def test_classify_failure_priority_parse_before_retrieve() -> None:
    case = _valid_action_case(
        expected_candidate_entities=["light.living_room_ceiling", "light.kitchen"],
    )
    record = _matching_valid_action_record(
        recorded_candidate_entities=["light.living_room_ceiling"],
        recorded_control_plan={
            "outcome": "no-action",
            "intent": "turn off the lights",
            "reason": "model_output_invalid",
        },
    )
    assert classify_failure(case, record)[0] == "parse"


def test_ledger_entries_pairs_by_case_id_and_skips_successes() -> None:
    success_case = _valid_action_case(case_id="ok-001")
    fail_case = _valid_action_case(case_id="fail-001")
    cases = [success_case, fail_case]
    records = [
        _matching_valid_action_record(case_id="ok-001"),
        _matching_valid_action_record(
            case_id="fail-001",
            recorded_control_plan=None,
        ),
    ]
    entries = ledger_entries(cases, records)
    assert len(entries) == 1
    assert entries[0].case_id == "fail-001"
    assert entries[0].stage == "plan"
    assert entries[0].reason


def test_ledger_entries_omits_cases_without_records() -> None:
    case = _valid_action_case(case_id="missing-record")
    entries = ledger_entries([case], [])
    assert entries == []


def test_ledger_summary_counts_by_stage_and_reason_with_case_ids() -> None:
    entries = ledger_entries(
        [
            _valid_action_case(case_id="a-001"),
            _valid_action_case(case_id="b-002"),
            _valid_action_case(case_id="c-003"),
        ],
        [
            _matching_valid_action_record(case_id="a-001", recorded_control_plan=None),
            _matching_valid_action_record(
                case_id="b-002",
                recorded_resolved_entities=["light.kitchen"],
            ),
            _matching_valid_action_record(case_id="c-003", schema_failure=True),
        ],
    )
    summary = ledger_summary(entries)
    assert [item.stage for item in summary.by_stage] == ["plan", "resolve", "schema"]
    plan_bucket = summary.by_stage[0]
    assert plan_bucket.count == 1
    assert plan_bucket.case_ids == ["a-001"]
    resolve_bucket = summary.by_stage[1]
    assert resolve_bucket.count == 1
    assert resolve_bucket.case_ids == ["b-002"]
    schema_bucket = summary.by_stage[2]
    assert schema_bucket.count == 1
    assert schema_bucket.case_ids == ["c-003"]

    assert len(summary.by_stage_reason) == 3
    assert summary.by_stage_reason[0].stage == "plan"
    assert summary.by_stage_reason[0].case_ids == ["a-001"]
    assert summary.by_stage_reason[1].stage == "resolve"
    assert summary.by_stage_reason[1].case_ids == ["b-002"]
    assert summary.by_stage_reason[2].stage == "schema"
    assert summary.by_stage_reason[2].case_ids == ["c-003"]


def test_ledger_summary_aggregates_same_stage_reason() -> None:
    entries = ledger_entries(
        [
            _valid_action_case(case_id="z-fail"),
            _valid_action_case(case_id="a-fail"),
        ],
        [
            _matching_valid_action_record(case_id="z-fail", recorded_control_plan=None),
            _matching_valid_action_record(case_id="a-fail", recorded_control_plan=None),
        ],
    )
    summary = ledger_summary(entries)
    assert len(summary.by_stage) == 1
    assert summary.by_stage[0].count == 2
    assert summary.by_stage[0].case_ids == ["a-fail", "z-fail"]
    assert len(summary.by_stage_reason) == 1
    assert summary.by_stage_reason[0].count == 2
    assert summary.by_stage_reason[0].case_ids == ["a-fail", "z-fail"]


def test_ledger_summary_empty_entries() -> None:
    summary = ledger_summary([])
    assert summary.by_stage == []
    assert summary.by_stage_reason == []
