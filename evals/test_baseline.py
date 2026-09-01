"""Frozen FakeModelRuntime baseline for controller dry-run wiring."""

from __future__ import annotations

import json
from pathlib import Path

from evals.executor import controller_dry_run_executor
from evals.metrics import MetricScore, score_records
from evals.schema import EvalCase

_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "baseline_core_slice.json"

_NUMERATOR_FIELDS = (
    "total_cases",
    "control_plan_exact_match",
    "control_plan_semantic_wrong",
    "control_plan_schema_failure",
    "candidate_retrieval_numerator",
    "candidate_retrieval_denominator",
    "exact_target_numerator",
    "exact_target_denominator",
    "wrong_device_numerator",
    "wrong_device_denominator",
    "unintended_entity_count",
    "false_execution_numerator",
    "false_execution_denominator",
    "clarification_true_positive",
    "clarification_false_positive",
    "clarification_false_negative",
    "query_numerator",
    "query_denominator",
    "follow_up_numerator",
    "follow_up_denominator",
)


def _load_fixture() -> tuple[list[EvalCase], dict[str, float | int]]:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = [EvalCase.model_validate(raw) for raw in payload["cases"]]
    expected_metrics = payload["expected_metrics"]
    return cases, expected_metrics


def _numerator_fields(score: MetricScore) -> dict[str, float | int]:
    return {field: getattr(score, field) for field in _NUMERATOR_FIELDS}


def test_baseline_core_slice_matches_golden_numerators() -> None:
    cases, expected_metrics = _load_fixture()
    records = [controller_dry_run_executor(case).record for case in cases]
    score = score_records(cases, records)

    assert score.missing_records == []
    assert _numerator_fields(score) == {field: expected_metrics[field] for field in _NUMERATOR_FIELDS}
