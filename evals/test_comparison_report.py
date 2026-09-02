"""Model comparison report tests (unit 9.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import executor as executor_module
from evals.comparison_report import (
    ComparisonReportError,
    build_comparison_report,
    default_comparison_report_path,
    write_comparison_report,
)
from evals.config import (
    comparison_baseline_benchmark_config,
    parse_benchmark_config,
    sayso_comparison_benchmark_config,
)
from evals.corpus import COMPARISON_SCENARIO_COUNTS, load_comparison_corpus
from evals.executor import comparison_baseline_executor, controller_dry_run_executor
from evals.report import load_benchmark_jsonl_rows
from evals.runner import run_benchmark


def _write_config_header(path: Path, config) -> None:
    path.write_text(
        json.dumps(parse_benchmark_config(config), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _boundary_row(case_id: str, *, ha_executed: bool = False) -> dict[str, object]:
    return {
        "case_id": case_id,
        "ha_executed": ha_executed,
        "recorded_control_plan": {"outcome": "action", "intent": "x", "domain": "light"},
        "recorded_candidate_entities": ["light.living_room_ceiling"],
        "recorded_resolved_entities": ["light.living_room_ceiling"],
        "total_ms": 100.0,
        "plan_ms": 20.0,
        "request_ms": 40.0,
        "verify_ms": 60.0,
    }


@pytest.fixture
def comparison_cases():
    return load_comparison_corpus()


@pytest.fixture
def comparison_fixture_paths(
    tmp_path: Path,
    comparison_cases,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    executor_module._resident_fake_runtime = None
    executor_module._resident_comparison_runtime = None

    sayso_output = tmp_path / "sayso.jsonl"
    baseline_output = tmp_path / "baseline.jsonl"

    run_benchmark(
        comparison_cases,
        sayso_output,
        controller_dry_run_executor,
        config=sayso_comparison_benchmark_config(),
    )
    run_benchmark(
        comparison_cases,
        baseline_output,
        comparison_baseline_executor,
        config=comparison_baseline_benchmark_config(),
    )
    return sayso_output, baseline_output


def test_build_comparison_report_rejects_missing_cases(
    tmp_path: Path,
    comparison_cases,
    comparison_fixture_paths: tuple[Path, Path],
) -> None:
    sayso_output, baseline_output = comparison_fixture_paths
    incomplete = tmp_path / "incomplete.jsonl"
    rows = load_benchmark_jsonl_rows(sayso_output)[:1]
    _write_config_header(incomplete, sayso_comparison_benchmark_config())
    for row in rows:
        _append_row(incomplete, row)

    with pytest.raises(ComparisonReportError, match="missing cases"):
        build_comparison_report(incomplete, baseline_output, cases=comparison_cases)


def test_build_comparison_report_rejects_unequal_run_counts(
    tmp_path: Path,
    comparison_cases,
    comparison_fixture_paths: tuple[Path, Path],
) -> None:
    sayso_output, baseline_output = comparison_fixture_paths
    extra = tmp_path / "extra.jsonl"
    _write_config_header(extra, comparison_baseline_benchmark_config())
    for row in load_benchmark_jsonl_rows(baseline_output):
        _append_row(extra, row)
    _append_row(extra, _boundary_row("comparison-warm-001"))

    with pytest.raises(ComparisonReportError, match="unequal run counts"):
        build_comparison_report(sayso_output, extra, cases=comparison_cases)


def test_build_comparison_report_rejects_live_actuation_without_allowlist(
    tmp_path: Path,
    comparison_cases,
    comparison_fixture_paths: tuple[Path, Path],
) -> None:
    sayso_output, baseline_output = comparison_fixture_paths
    live = tmp_path / "live.jsonl"
    _write_config_header(live, sayso_comparison_benchmark_config())
    for row in load_benchmark_jsonl_rows(sayso_output):
        patched = dict(row)
        if row["case_id"] == "comparison-warm-001":
            patched["ha_executed"] = True
        _append_row(live, patched)

    with pytest.raises(ComparisonReportError, match="live actuation"):
        build_comparison_report(live, baseline_output, cases=comparison_cases)


def test_build_comparison_report_rejects_mixed_timing_definitions(
    tmp_path: Path,
    comparison_cases,
    comparison_fixture_paths: tuple[Path, Path],
) -> None:
    sayso_output, baseline_output = comparison_fixture_paths
    mixed = tmp_path / "mixed.jsonl"
    _write_config_header(mixed, sayso_comparison_benchmark_config())
    for row in load_benchmark_jsonl_rows(sayso_output):
        patched = dict(row)
        if row["case_id"] == "comparison-warm-001":
            patched["plan_ms"] = 50.0
            patched["request_ms"] = 10.0
        _append_row(mixed, patched)

    with pytest.raises(ComparisonReportError, match="mixed timing"):
        build_comparison_report(mixed, baseline_output, cases=comparison_cases)


def test_build_comparison_report_covers_all_six_scenarios(
    comparison_cases,
    comparison_fixture_paths: tuple[Path, Path],
) -> None:
    sayso_output, baseline_output = comparison_fixture_paths
    report = build_comparison_report(sayso_output, baseline_output, cases=comparison_cases)

    assert report["report_kind"] == "model_comparison"
    assert report["scenarios"] == list(COMPARISON_SCENARIO_COUNTS.keys())
    for model_key in ("sayso", "home_llm_270m"):
        by_scenario = report[model_key]["by_scenario"]
        assert set(by_scenario) == set(COMPARISON_SCENARIO_COUNTS)


def test_build_comparison_report_includes_accuracy_safety_and_latency(
    comparison_cases,
    comparison_fixture_paths: tuple[Path, Path],
) -> None:
    sayso_output, baseline_output = comparison_fixture_paths
    report = build_comparison_report(sayso_output, baseline_output, cases=comparison_cases)

    for model_key in ("sayso", "home_llm_270m"):
        payload = report[model_key]
        metrics = payload["metrics"]
        latency = payload["latency"]
        assert "control_plan" in metrics
        assert "false_execution" in metrics
        assert "wrong_device" in metrics
        assert latency["warm_only"] is True
        assert "cold_readiness_ms" in latency
        assert latency["stages"]["verify_ms"]["n"] >= 1


def test_build_comparison_report_records_model_configs(
    comparison_cases,
    comparison_fixture_paths: tuple[Path, Path],
) -> None:
    sayso_output, baseline_output = comparison_fixture_paths
    report = build_comparison_report(sayso_output, baseline_output, cases=comparison_cases)

    assert report["sayso"]["config"]["model_id"] == sayso_comparison_benchmark_config().model_id
    assert (
        report["home_llm_270m"]["config"]["model_id"]
        == comparison_baseline_benchmark_config().model_id
    )


def test_build_comparison_report_is_reproducible(
    comparison_cases,
    comparison_fixture_paths: tuple[Path, Path],
) -> None:
    sayso_output, baseline_output = comparison_fixture_paths
    first = build_comparison_report(sayso_output, baseline_output, cases=comparison_cases)
    second = build_comparison_report(sayso_output, baseline_output, cases=comparison_cases)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_build_comparison_report_summary_matches_raw_jsonl(
    comparison_cases,
    comparison_fixture_paths: tuple[Path, Path],
) -> None:
    sayso_output, baseline_output = comparison_fixture_paths
    report = build_comparison_report(sayso_output, baseline_output, cases=comparison_cases)

    sayso_rows = load_benchmark_jsonl_rows(sayso_output)
    warm_rows = [row for row in sayso_rows if row.get("cold_start") is not True]
    assert report["sayso"]["latency"]["n"] == len(warm_rows)
    assert report["summary"]["sayso"]["accuracy"] == report["sayso"]["metrics"]["control_plan"]["accuracy"]
    assert (
        report["summary"]["home_llm_270m"]["verified_action_latency_ms"]
        == report["home_llm_270m"]["latency"]["stages"]["verify_ms"]["median"]
    )


def test_default_comparison_report_path_uses_comparison_name() -> None:
    path = default_comparison_report_path()
    assert path.name == "comparison.report.json"
    assert path.parent.name == "reports"


def test_write_comparison_report_writes_sorted_json(
    comparison_cases,
    comparison_fixture_paths: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    sayso_output, baseline_output = comparison_fixture_paths
    report = build_comparison_report(sayso_output, baseline_output, cases=comparison_cases)
    output = tmp_path / "comparison.report.json"
    write_comparison_report(output, report)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["report_kind"] == "model_comparison"
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_build_comparison_report_rejects_missing_boundary_fields(
    tmp_path: Path,
    comparison_cases,
    comparison_fixture_paths: tuple[Path, Path],
) -> None:
    sayso_output, baseline_output = comparison_fixture_paths
    incomplete = tmp_path / "no-boundary.jsonl"
    _write_config_header(incomplete, sayso_comparison_benchmark_config())
    for row in load_benchmark_jsonl_rows(sayso_output):
        patched = dict(row)
        patched.pop("verify_ms", None)
        _append_row(incomplete, patched)

    with pytest.raises(ComparisonReportError, match="mixed timing"):
        build_comparison_report(incomplete, baseline_output, cases=comparison_cases)
