"""Expansion gate tests."""

from __future__ import annotations

from evals.gate import expansion_allowed
from evals.latency import LatencyReport
from evals.ledger import LedgerEntry, LedgerSummary, ledger_summary
from evals.metrics import MetricScore


def _passing_score(**overrides: object) -> MetricScore:
    payload: dict[str, object] = {
        "total_cases": 1,
        "control_plan_exact_match": 1,
        "control_plan_semantic_wrong": 0,
        "control_plan_schema_failure": 0,
        "control_plan_accuracy": 1.0,
        "candidate_retrieval_recall": 1.0,
        "candidate_retrieval_numerator": 1,
        "candidate_retrieval_denominator": 1,
        "mean_candidate_set_size": 1.0,
        "exact_target_resolution": 1.0,
        "exact_target_numerator": 1,
        "exact_target_denominator": 1,
        "wrong_device_rate": 0.0,
        "wrong_device_numerator": 0,
        "wrong_device_denominator": 0,
        "unintended_entity_count": 0,
        "false_execution_rate": 0.0,
        "false_execution_numerator": 0,
        "false_execution_denominator": 1,
        "clarification_precision": 0.0,
        "clarification_recall": 0.0,
        "clarification_true_positive": 0,
        "clarification_false_positive": 0,
        "clarification_false_negative": 0,
        "query_accuracy": 0.0,
        "query_numerator": 0,
        "query_denominator": 0,
        "follow_up_accuracy": 0.0,
        "follow_up_numerator": 0,
        "follow_up_denominator": 0,
        "missing_records": [],
    }
    payload.update(overrides)
    return MetricScore(**payload)  # type: ignore[arg-type]


def _passing_latency(**overrides: object) -> LatencyReport:
    payload: dict[str, object] = {
        "n": 1,
        "median": 10.0,
        "p95": 10.0,
        "stages": {},
    }
    payload.update(overrides)
    return LatencyReport(**payload)  # type: ignore[arg-type]


def _empty_ledger() -> LedgerSummary:
    return ledger_summary([])


def test_expansion_allowed_passes_when_all_criteria_met() -> None:
    allowed, reasons = expansion_allowed(
        _passing_score(),
        _empty_ledger(),
        _passing_latency(),
    )

    assert allowed is True
    assert reasons == []


def test_expansion_allowed_fails_when_false_execution_rate_nonzero() -> None:
    allowed, reasons = expansion_allowed(
        _passing_score(
            false_execution_rate=0.5,
            false_execution_numerator=1,
            false_execution_denominator=2,
        ),
        _empty_ledger(),
        _passing_latency(),
    )

    assert allowed is False
    assert any("false_execution_rate" in reason for reason in reasons)


def test_expansion_allowed_fails_closed_when_false_execution_denominator_zero() -> None:
    allowed, reasons = expansion_allowed(
        _passing_score(
            false_execution_rate=0.0,
            false_execution_numerator=0,
            false_execution_denominator=0,
        ),
        _empty_ledger(),
        _passing_latency(),
    )

    assert allowed is False
    assert any("fail closed" in reason for reason in reasons)


def test_expansion_allowed_fails_when_wrong_device_rate_nonzero() -> None:
    allowed, reasons = expansion_allowed(
        _passing_score(
            wrong_device_rate=1.0,
            wrong_device_numerator=1,
            wrong_device_denominator=1,
        ),
        _empty_ledger(),
        _passing_latency(),
    )

    assert allowed is False
    assert any("wrong_device_rate" in reason for reason in reasons)


def test_expansion_allowed_fails_when_latency_n_zero() -> None:
    allowed, reasons = expansion_allowed(
        _passing_score(),
        _empty_ledger(),
        _passing_latency(n=0, median=0.0, p95=0.0),
    )

    assert allowed is False
    assert any("latency.n" in reason for reason in reasons)


def test_expansion_allowed_fails_on_schema_executor_crash() -> None:
    summary = ledger_summary(
        [
            LedgerEntry(case_id="crash-001", stage="schema", reason="executor_error"),
        ],
    )

    allowed, reasons = expansion_allowed(
        _passing_score(),
        summary,
        _passing_latency(),
    )

    assert allowed is False
    assert any("schema executor crash" in reason for reason in reasons)


def test_expansion_allowed_fails_on_unclassified_schema_executor_crash() -> None:
    summary = ledger_summary(
        [
            LedgerEntry(case_id="crash-002", stage="schema", reason="unexpected boom"),
        ],
    )

    allowed, reasons = expansion_allowed(
        _passing_score(),
        summary,
        _passing_latency(),
    )

    assert allowed is False
    assert any("unclassified schema executor crash" in reason for reason in reasons)


def test_expansion_allowed_allows_classified_schema_failure() -> None:
    summary = ledger_summary(
        [
            LedgerEntry(case_id="schema-001", stage="schema", reason="schema_failure"),
        ],
    )

    allowed, reasons = expansion_allowed(
        _passing_score(),
        summary,
        _passing_latency(),
    )

    assert allowed is True
    assert reasons == []
