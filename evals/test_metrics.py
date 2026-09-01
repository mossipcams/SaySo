"""Metric scorer tests against a hand-calculated fixture."""

from __future__ import annotations

import json
from pathlib import Path

from evals.metrics import EvalRecord, MetricScore, canonicalize_control_plan, score_records
from evals.schema import EvalCase

_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "metrics_handcalc.json"


def _load_fixture() -> tuple[list[EvalCase], list[EvalRecord], dict[str, str], dict[str, float | int]]:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = [EvalCase.model_validate(raw) for raw in payload["cases"]]
    records = [EvalRecord.model_validate(raw) for raw in payload["records"]]
    expected_query_answers = payload.get("expected_query_answers", {})
    expected_metrics = payload["expected_metrics"]
    return cases, records, expected_query_answers, expected_metrics


def _metric_fields(score: MetricScore) -> dict[str, float | int]:
    return {
        "total_cases": score.total_cases,
        "control_plan_exact_match": score.control_plan_exact_match,
        "control_plan_semantic_wrong": score.control_plan_semantic_wrong,
        "control_plan_schema_failure": score.control_plan_schema_failure,
        "control_plan_accuracy": score.control_plan_accuracy,
        "candidate_retrieval_recall": score.candidate_retrieval_recall,
        "candidate_retrieval_numerator": score.candidate_retrieval_numerator,
        "candidate_retrieval_denominator": score.candidate_retrieval_denominator,
        "mean_candidate_set_size": score.mean_candidate_set_size,
        "exact_target_resolution": score.exact_target_resolution,
        "exact_target_numerator": score.exact_target_numerator,
        "exact_target_denominator": score.exact_target_denominator,
        "wrong_device_rate": score.wrong_device_rate,
        "wrong_device_numerator": score.wrong_device_numerator,
        "wrong_device_denominator": score.wrong_device_denominator,
        "unintended_entity_count": score.unintended_entity_count,
        "false_execution_rate": score.false_execution_rate,
        "false_execution_numerator": score.false_execution_numerator,
        "false_execution_denominator": score.false_execution_denominator,
        "clarification_precision": score.clarification_precision,
        "clarification_recall": score.clarification_recall,
        "clarification_true_positive": score.clarification_true_positive,
        "clarification_false_positive": score.clarification_false_positive,
        "clarification_false_negative": score.clarification_false_negative,
        "query_accuracy": score.query_accuracy,
        "query_numerator": score.query_numerator,
        "query_denominator": score.query_denominator,
        "follow_up_accuracy": score.follow_up_accuracy,
        "follow_up_numerator": score.follow_up_numerator,
        "follow_up_denominator": score.follow_up_denominator,
    }


def test_canonicalize_control_plan_sorts_unordered_lists_and_defaults() -> None:
    raw = {
        "outcome": "action",
        "intent": "turn on lights",
        "domain": "light",
        "exclude": ["lamp", "ceiling"],
        "include": ["desk", "ceiling"],
        "targets": ["b", "a"],
        "state": "on",
    }
    canonical = canonicalize_control_plan(raw)
    assert canonical["targets"] == ["a", "b"]
    assert canonical["include"] == ["ceiling", "desk"]
    assert canonical["exclude"] == ["ceiling", "lamp"]


def test_hand_calculated_fixture_matches_exactly() -> None:
    cases, records, expected_query_answers, expected_metrics = _load_fixture()
    score = score_records(cases, records, expected_query_answers=expected_query_answers)
    assert score.missing_records == []
    assert _metric_fields(score) == expected_metrics


def test_schema_failure_is_counted_separately_from_semantic_wrong_plan() -> None:
    cases, records, expected_query_answers, _ = _load_fixture()
    score = score_records(cases, records, expected_query_answers=expected_query_answers)
    assert score.control_plan_schema_failure == 1
    assert score.control_plan_semantic_wrong == 3
    assert (
        score.control_plan_exact_match
        + score.control_plan_semantic_wrong
        + score.control_plan_schema_failure
        == score.total_cases
    )
