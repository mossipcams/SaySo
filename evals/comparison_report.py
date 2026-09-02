"""Reproducible SaySo vs Home-LLM comparison report assembly."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evals.config import BenchmarkConfig, load_benchmark_config
from evals.corpus import COMPARISON_SCENARIO_COUNTS, load_comparison_corpus, validate_comparison_corpus
from evals.latency import LATENCY_BOUNDARY_FIELDS, latency_report
from evals.metrics import score_records
from evals.report import (
    _serialize_ledger_summary,
    _serialize_latency_report,
    _serialize_metric_score,
    build_gate_inputs_from_rows,
    load_benchmark_jsonl_rows,
    record_from_jsonl_row,
)
from evals.schema import EvalCase

TIMING_DEFINITION_ID = "eos_boundary_v1"
COMPARISON_REPORT_KIND = "model_comparison"


class ComparisonReportError(ValueError):
    """Comparison inputs failed validation."""


def default_comparison_report_path() -> Path:
    return Path(__file__).resolve().parent / "reports" / "comparison.report.json"


def _case_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["case_id"]) for row in rows if row.get("case_id")]


def validate_missing_cases(cases: list[EvalCase], rows: list[dict[str, Any]]) -> None:
    expected = {case.case_id for case in cases}
    present = set(_case_ids(rows))
    missing = sorted(expected - present)
    if missing:
        msg = f"missing cases: {', '.join(missing)}"
        raise ComparisonReportError(msg)


def validate_equal_run_counts(
    sayso_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> None:
    sayso_counts = Counter(_case_ids(sayso_rows))
    baseline_counts = Counter(_case_ids(baseline_rows))
    if sayso_counts != baseline_counts:
        msg = (
            "unequal run counts between models: "
            f"sayso={dict(sayso_counts)} baseline={dict(baseline_counts)}"
        )
        raise ComparisonReportError(msg)


def validate_no_live_actuation_without_allowlist(rows: list[dict[str, Any]]) -> None:
    live_case_ids = sorted(
        str(row["case_id"])
        for row in rows
        if row.get("ha_executed") is True and row.get("case_id")
    )
    if not live_case_ids:
        return
    msg = (
        "live actuation without allowlist on "
        + ", ".join(live_case_ids)
    )
    raise ComparisonReportError(msg)


def validate_timing_definitions(rows: list[dict[str, Any]]) -> None:
    definitions: set[str | None] = set()
    for row in rows:
        if row.get("warmup") is True:
            continue
        definitions.add(
            None if row.get("timing_definition") is None else str(row["timing_definition"]),
        )
        values: list[float] = []
        for field in LATENCY_BOUNDARY_FIELDS:
            raw = row.get(field)
            if raw is None:
                msg = f"mixed timing definitions: row {row.get('case_id')} missing {field}"
                raise ComparisonReportError(msg)
            values.append(float(raw))
        if values[0] > values[1] or values[1] > values[2]:
            msg = (
                "mixed timing definitions: non-monotonic boundaries on "
                f"{row.get('case_id')}"
            )
            raise ComparisonReportError(msg)
    if len(definitions) > 1:
        msg = f"mixed timing definitions: {sorted(definitions)}"
        raise ComparisonReportError(msg)


def validate_comparison_inputs(
    cases: list[EvalCase],
    sayso_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> None:
    validate_comparison_corpus(cases)
    validate_missing_cases(cases, sayso_rows)
    validate_missing_cases(cases, baseline_rows)
    validate_equal_run_counts(sayso_rows, baseline_rows)
    validate_no_live_actuation_without_allowlist(sayso_rows)
    validate_no_live_actuation_without_allowlist(baseline_rows)
    validate_timing_definitions(sayso_rows)
    validate_timing_definitions(baseline_rows)


def _cases_for_scenario(cases: list[EvalCase], scenario: str) -> list[EvalCase]:
    return [case for case in cases if case.category == scenario]


def _rows_for_cases(rows: list[dict[str, Any]], case_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("case_id")) in case_ids]


def _build_side_payload(
    cases: list[EvalCase],
    rows: list[dict[str, Any]],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    score, summary, latency = build_gate_inputs_from_rows(cases, rows)
    by_scenario: dict[str, Any] = {}
    for scenario in COMPARISON_SCENARIO_COUNTS:
        scenario_cases = _cases_for_scenario(cases, scenario)
        scenario_ids = {case.case_id for case in scenario_cases}
        scenario_rows = _rows_for_cases(rows, scenario_ids)
        scenario_score = score_records(scenario_cases, [record_from_jsonl_row(row) for row in scenario_rows])
        scenario_latency = latency_report(scenario_rows, warm_only=True)
        by_scenario[scenario] = {
            "case_ids": sorted(scenario_ids),
            "metrics": _serialize_metric_score(scenario_score),
            "latency": _serialize_latency_report(
                scenario_latency,
                warm_only=True,
                rows=scenario_rows,
            ),
        }
    return {
        "config": asdict(config),
        "metrics": _serialize_metric_score(score),
        "failures": _serialize_ledger_summary(summary),
        "latency": _serialize_latency_report(latency, warm_only=True, rows=rows),
        "by_scenario": by_scenario,
    }


def _summary_for_side(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    latency = payload["latency"]
    verify_stats = latency["stages"].get("verify_ms", {"median": 0.0, "p95": 0.0, "n": 0})
    return {
        "accuracy": metrics["control_plan"]["accuracy"],
        "false_execution_rate": metrics["false_execution"]["rate"],
        "wrong_device_rate": metrics["wrong_device"]["rate"],
        "warm_latency_ms": {
            "median": latency["median"],
            "p95": latency["p95"],
            "n": latency["n"],
        },
        "cold_readiness_ms": latency["cold_readiness_ms"],
        "verified_action_latency_ms": verify_stats["median"],
        "verified_action_latency_p95_ms": verify_stats["p95"],
    }


def build_comparison_report(
    sayso_output: str | Path,
    baseline_output: str | Path,
    *,
    cases: list[EvalCase] | None = None,
) -> dict[str, Any]:
    """Build a reproducible comparison report from two benchmark JSONL runs."""
    case_list = cases or load_comparison_corpus()
    sayso_path = Path(sayso_output)
    baseline_path = Path(baseline_output)
    sayso_rows = load_benchmark_jsonl_rows(sayso_path)
    baseline_rows = load_benchmark_jsonl_rows(baseline_path)
    validate_comparison_inputs(case_list, sayso_rows, baseline_rows)

    sayso_config = load_benchmark_config(sayso_path) or BenchmarkConfig()
    baseline_config = load_benchmark_config(baseline_path) or BenchmarkConfig()

    sayso_payload = _build_side_payload(case_list, sayso_rows, sayso_config)
    baseline_payload = _build_side_payload(case_list, baseline_rows, baseline_config)

    sayso_summary = _summary_for_side(sayso_payload)
    baseline_summary = _summary_for_side(baseline_payload)

    return {
        "report_kind": COMPARISON_REPORT_KIND,
        "timing_definition": TIMING_DEFINITION_ID,
        "scenarios": list(COMPARISON_SCENARIO_COUNTS.keys()),
        "sayso": sayso_payload,
        "home_llm_270m": baseline_payload,
        "summary": {
            "sayso": sayso_summary,
            "home_llm_270m": baseline_summary,
            "accuracy_delta": sayso_summary["accuracy"] - baseline_summary["accuracy"],
            "false_execution_delta": (
                sayso_summary["false_execution_rate"] - baseline_summary["false_execution_rate"]
            ),
            "wrong_device_delta": (
                sayso_summary["wrong_device_rate"] - baseline_summary["wrong_device_rate"]
            ),
            "warm_latency_median_delta_ms": (
                sayso_summary["warm_latency_ms"]["median"]
                - baseline_summary["warm_latency_ms"]["median"]
            ),
            "verified_action_latency_median_delta_ms": (
                sayso_summary["verified_action_latency_ms"]
                - baseline_summary["verified_action_latency_ms"]
            ),
            "cold_readiness_median_delta_ms": (
                sayso_summary["cold_readiness_ms"]["median"]
                - baseline_summary["cold_readiness_ms"]["median"]
            ),
        },
    }


def write_comparison_report(output_path: str | Path, report: dict[str, Any]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path
