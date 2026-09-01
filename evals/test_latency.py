"""Latency percentile report tests."""

from __future__ import annotations

from evals.latency import latency_report


def test_latency_report_empty_input_returns_zero_sample() -> None:
    report = latency_report([])
    assert report.n == 0
    assert report.median == 0.0
    assert report.p95 == 0.0
    assert report.stages == {}


def test_latency_report_single_row_nearest_rank() -> None:
    report = latency_report([{"total_ms": 42.0}])
    assert report.n == 1
    assert report.median == 42.0
    assert report.p95 == 42.0


def test_latency_report_median_and_p95_use_nearest_rank() -> None:
    rows = [{"total_ms": float(value)} for value in (10, 20, 30, 40, 50)]
    report = latency_report(rows, warm_only=False)
    assert report.n == 5
    assert report.median == 30.0
    assert report.p95 == 50.0


def test_latency_report_even_count_median_nearest_rank() -> None:
    rows = [{"total_ms": float(value)} for value in (10, 20, 30, 40)]
    report = latency_report(rows, warm_only=False)
    assert report.n == 4
    assert report.median == 20.0
    assert report.p95 == 40.0


def test_latency_report_warm_only_excludes_cold_and_warmup_rows() -> None:
    rows = [
        {"total_ms": 1000.0, "cold_start": True},
        {"total_ms": 900.0, "warmup": True},
        {"total_ms": 50.0},
        {"total_ms": 150.0},
    ]
    report = latency_report(rows, warm_only=True)
    assert report.n == 2
    assert report.median == 50.0
    assert report.p95 == 150.0


def test_latency_report_warm_only_false_includes_cold_and_warmup_rows() -> None:
    rows = [
        {"total_ms": 1000.0, "cold_start": True},
        {"total_ms": 50.0},
    ]
    report = latency_report(rows, warm_only=False)
    assert report.n == 2
    assert report.median == 50.0
    assert report.p95 == 1000.0


def test_latency_report_includes_present_stage_fields() -> None:
    rows = [
        {"total_ms": 100.0, "plan_ms": 10.0, "retrieve_ms": 5.0},
        {"total_ms": 200.0, "plan_ms": 30.0, "retrieve_ms": 15.0},
        {"total_ms": 300.0, "plan_ms": 50.0},
    ]
    report = latency_report(rows, warm_only=False)
    assert report.n == 3
    assert set(report.stages) == {"plan_ms", "retrieve_ms"}
    assert report.stages["plan_ms"].n == 3
    assert report.stages["plan_ms"].median == 30.0
    assert report.stages["plan_ms"].p95 == 50.0
    assert report.stages["retrieve_ms"].n == 2
    assert report.stages["retrieve_ms"].median == 5.0
    assert report.stages["retrieve_ms"].p95 == 15.0


def test_latency_report_omits_stage_fields_absent_from_all_rows() -> None:
    rows = [{"total_ms": 100.0}, {"total_ms": 200.0}]
    report = latency_report(rows, warm_only=False)
    assert report.stages == {}
