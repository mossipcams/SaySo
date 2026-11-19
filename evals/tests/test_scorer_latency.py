from __future__ import annotations

from typing import Any

from evals.cases import select_cases
from evals.outcomes import CaseResult, Outcome, Turn
from evals.runner import evaluate
from evals.scorer import summarize

LEGACY_SUMMARY_KEYS = {
    "total",
    "passed",
    "overall_pass_rate",
    "category_pass_rate",
    "action_execution_accuracy",
    "false_action_rate",
    "clarification_accuracy",
    "refusal_accuracy",
    "malformed_output_rate",
    "unknown_tool_rate",
    "invalid_argument_rate",
    "outcomes",
    "flags",
}


def _result(latencies: list[float]) -> CaseResult:
    return CaseResult(
        case_id="case",
        category="ordinary",
        outcome=Outcome.VALID_TOOL_CALL,
        passed=True,
        flags=[],
        turn=Turn(raw={}),
        expected={"response_type": "action"},
        request_latencies_ms=latencies,
    )


def test_p50_and_p95_use_nearest_rank_across_cases() -> None:
    summary = summarize(
        [
            _result([10.0, 20.0, 30.0, 40.0, 50.0]),
            _result([60.0, 70.0, 80.0, 90.0, 100.0]),
        ]
    )

    assert summary["latency_p50_ms"] == 50.0
    assert summary["latency_p95_ms"] == 100.0


def test_single_request_is_both_percentiles() -> None:
    summary = summarize([_result([42.0])])

    assert summary["latency_p50_ms"] == 42.0
    assert summary["latency_p95_ms"] == 42.0


def test_latency_is_none_without_requests() -> None:
    summary = summarize([_result([])])

    assert summary["latency_p50_ms"] is None
    assert summary["latency_p95_ms"] is None


def test_existing_summary_fields_are_kept() -> None:
    summary: dict[str, Any] = summarize([_result([5.0])])

    assert LEGACY_SUMMARY_KEYS <= set(summary)


class _Timed:

    name = "timed"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        return {"choices": [{"message": {"role": "assistant", "content": "Sorry."}}]}


def test_evaluate_records_one_latency_per_request() -> None:
    adapter = _Timed()

    results = evaluate(select_cases(case_id="realistic_eval_20260908_ordinary_01"), adapter)

    latencies = results[0].request_latencies_ms
    assert len(latencies) == adapter.calls >= 1
    assert all(ms >= 0.0 for ms in latencies)
    assert summarize(results)["latency_p50_ms"] is not None
