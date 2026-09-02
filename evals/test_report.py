"""Statistical eval report blob tests."""

from __future__ import annotations

import json
from dataclasses import asdict

from evals.config import BenchmarkConfig
from evals.latency import LatencyReport, latency_report
from evals.ledger import LedgerEntry, LedgerSummary, ledger_summary
from evals.metrics import MetricScore
from evals.report import build_eval_report, default_report_path, write_eval_report


def _sample_score() -> MetricScore:
    return MetricScore(
        total_cases=4,
        control_plan_exact_match=2,
        control_plan_semantic_wrong=1,
        control_plan_schema_failure=1,
        control_plan_accuracy=0.5,
        candidate_retrieval_recall=1.0,
        candidate_retrieval_numerator=2,
        candidate_retrieval_denominator=2,
        mean_candidate_set_size=1.0,
        exact_target_resolution=0.5,
        exact_target_numerator=1,
        exact_target_denominator=2,
        wrong_device_rate=0.0,
        wrong_device_numerator=0,
        wrong_device_denominator=0,
        unintended_entity_count=0,
        false_execution_rate=0.0,
        false_execution_numerator=0,
        false_execution_denominator=1,
        clarification_precision=0.0,
        clarification_recall=0.0,
        clarification_true_positive=0,
        clarification_false_positive=0,
        clarification_false_negative=0,
        query_accuracy=0.0,
        query_numerator=0,
        query_denominator=0,
        follow_up_accuracy=0.0,
        follow_up_numerator=0,
        follow_up_denominator=0,
        missing_records=[],
    )


def _sample_ledger_summary() -> LedgerSummary:
    return ledger_summary(
        [
            LedgerEntry(case_id="a-001", stage="plan", reason="control plan mismatch"),
            LedgerEntry(case_id="b-002", stage="schema", reason="schema_failure"),
        ],
    )


def _sample_latency(*, warm_only: bool = True) -> LatencyReport:
    rows = [
        {"total_ms": 1000.0, "cold_start": True},
        {"total_ms": 50.0, "plan_ms": 10.0},
        {"total_ms": 150.0, "plan_ms": 30.0},
    ]
    return latency_report(rows, warm_only=warm_only)


def test_build_eval_report_includes_metric_numerators_and_denominators() -> None:
    report = build_eval_report(
        _sample_score(),
        _sample_ledger_summary(),
        _sample_latency(),
        BenchmarkConfig(),
    )

    metrics = report["metrics"]
    assert metrics["total_cases"] == 4
    assert metrics["control_plan"]["exact_match"] == 2
    assert metrics["candidate_retrieval"]["numerator"] == 2
    assert metrics["candidate_retrieval"]["denominator"] == 2
    assert metrics["exact_target"]["numerator"] == 1
    assert metrics["exact_target"]["denominator"] == 2
    assert metrics["false_execution"]["numerator"] == 0
    assert metrics["false_execution"]["denominator"] == 1


def test_build_eval_report_includes_failures_by_stage() -> None:
    report = build_eval_report(
        _sample_score(),
        _sample_ledger_summary(),
        _sample_latency(),
        BenchmarkConfig(),
    )

    failures = report["failures"]
    assert failures["by_stage"] == [
        {"stage": "plan", "count": 1, "case_ids": ["a-001"]},
        {"stage": "schema", "count": 1, "case_ids": ["b-002"]},
    ]
    assert failures["by_stage_reason"][0]["stage"] == "plan"
    assert failures["by_stage_reason"][0]["reason"] == "control plan mismatch"
    assert failures["by_stage_reason"][0]["case_ids"] == ["a-001"]


def test_build_eval_report_includes_cold_readiness_separate_from_warm_latency() -> None:
    rows = [
        {"total_ms": 1000.0, "cold_start": True, "readiness_ms": 800.0, "plan_ms": 900.0},
        {"total_ms": 50.0, "plan_ms": 10.0, "request_ms": 20.0, "verify_ms": 30.0},
        {"total_ms": 150.0, "plan_ms": 30.0, "request_ms": 40.0, "verify_ms": 60.0},
    ]
    report = build_eval_report(
        _sample_score(),
        _sample_ledger_summary(),
        latency_report(rows, warm_only=True),
        BenchmarkConfig(),
        rows=rows,
    )

    assert report["latency"]["cold_readiness_ms"] == {"n": 1, "median": 800.0, "p95": 800.0}
    assert report["latency"]["n"] == 2
    assert report["latency"]["stages"]["plan_ms"] == {"n": 2, "median": 10.0, "p95": 30.0}


def test_build_eval_report_includes_latency_sample_size_and_percentiles() -> None:
    report = build_eval_report(
        _sample_score(),
        _sample_ledger_summary(),
        _sample_latency(),
        BenchmarkConfig(),
    )

    latency = report["latency"]
    assert latency["warm_only"] is True
    assert latency["n"] == 2
    assert latency["median"] == 50.0
    assert latency["p95"] == 150.0
    assert latency["stages"]["plan_ms"] == {"n": 2, "median": 10.0, "p95": 30.0}


def test_build_eval_report_records_warm_only_false_when_requested() -> None:
    report = build_eval_report(
        _sample_score(),
        _sample_ledger_summary(),
        _sample_latency(warm_only=False),
        BenchmarkConfig(),
        warm_only=False,
    )

    assert report["latency"]["warm_only"] is False
    assert report["latency"]["n"] == 3


def test_build_eval_report_includes_config_metadata() -> None:
    config = BenchmarkConfig(model_id="test-model", runtime="fake", seed=42)
    report = build_eval_report(
        _sample_score(),
        _sample_ledger_summary(),
        _sample_latency(),
        config,
    )

    assert report["config"] == asdict(config)


def test_build_eval_report_is_json_serializable() -> None:
    report = build_eval_report(
        _sample_score(),
        _sample_ledger_summary(),
        _sample_latency(),
        BenchmarkConfig(),
    )
    encoded = json.dumps(report, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["latency"]["n"] == 2


def test_default_report_path_uses_corpus_name() -> None:
    path = default_report_path("core")
    assert path.name == "core.report.json"
    assert path.parent.name == "reports"


def test_write_eval_report_writes_sorted_json(tmp_path) -> None:
    report = build_eval_report(
        _sample_score(),
        LedgerSummary(by_stage=[], by_stage_reason=[]),
        LatencyReport(n=0, median=0.0, p95=0.0, stages={}),
        BenchmarkConfig(),
    )
    output = tmp_path / "out.report.json"
    write_eval_report(output, report)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"]["total_cases"] == 4
    assert output.read_text(encoding="utf-8").endswith("\n")
