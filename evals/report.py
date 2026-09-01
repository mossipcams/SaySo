"""Statistical eval report assembly for benchmark scores and latency."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evals.config import BenchmarkConfig, is_benchmark_config_line, load_benchmark_config
from evals.latency import LatencyReport
from evals.ledger import LedgerSummary
from evals.metrics import EvalRecord, MetricScore, score_records
from evals.schema import EvalCase

_EVAL_RECORD_FIELDS = frozenset(EvalRecord.model_fields)


def default_report_path(corpus: str) -> Path:
    return Path(__file__).resolve().parent / "reports" / f"{corpus}.report.json"


def _serialize_metric_score(score: MetricScore) -> dict[str, Any]:
    return {
        "total_cases": score.total_cases,
        "missing_records": list(score.missing_records),
        "control_plan": {
            "exact_match": score.control_plan_exact_match,
            "semantic_wrong": score.control_plan_semantic_wrong,
            "schema_failure": score.control_plan_schema_failure,
            "accuracy": score.control_plan_accuracy,
        },
        "candidate_retrieval": {
            "numerator": score.candidate_retrieval_numerator,
            "denominator": score.candidate_retrieval_denominator,
            "recall": score.candidate_retrieval_recall,
            "mean_candidate_set_size": score.mean_candidate_set_size,
        },
        "exact_target": {
            "numerator": score.exact_target_numerator,
            "denominator": score.exact_target_denominator,
            "accuracy": score.exact_target_resolution,
        },
        "wrong_device": {
            "numerator": score.wrong_device_numerator,
            "denominator": score.wrong_device_denominator,
            "rate": score.wrong_device_rate,
            "unintended_entity_count": score.unintended_entity_count,
        },
        "false_execution": {
            "numerator": score.false_execution_numerator,
            "denominator": score.false_execution_denominator,
            "rate": score.false_execution_rate,
        },
        "clarification": {
            "true_positive": score.clarification_true_positive,
            "false_positive": score.clarification_false_positive,
            "false_negative": score.clarification_false_negative,
            "precision": score.clarification_precision,
            "recall": score.clarification_recall,
        },
        "query": {
            "numerator": score.query_numerator,
            "denominator": score.query_denominator,
            "accuracy": score.query_accuracy,
        },
        "follow_up": {
            "numerator": score.follow_up_numerator,
            "denominator": score.follow_up_denominator,
            "accuracy": score.follow_up_accuracy,
        },
    }


def _serialize_ledger_summary(summary: LedgerSummary) -> dict[str, Any]:
    return {
        "by_stage": [
            {
                "stage": item.stage,
                "count": item.count,
                "case_ids": list(item.case_ids),
            }
            for item in summary.by_stage
        ],
        "by_stage_reason": [
            {
                "stage": item.stage,
                "reason": item.reason,
                "count": item.count,
                "case_ids": list(item.case_ids),
            }
            for item in summary.by_stage_reason
        ],
    }


def _serialize_latency_report(latency: LatencyReport, *, warm_only: bool) -> dict[str, Any]:
    return {
        "warm_only": warm_only,
        "n": latency.n,
        "median": latency.median,
        "p95": latency.p95,
        "stages": {
            field: {
                "n": stats.n,
                "median": stats.median,
                "p95": stats.p95,
            }
            for field, stats in latency.stages.items()
        },
    }


def build_eval_report(
    score: MetricScore,
    ledger_summary: LedgerSummary,
    latency: LatencyReport,
    config: BenchmarkConfig,
    *,
    warm_only: bool = True,
) -> dict[str, Any]:
    """Build a JSON-serializable statistical report for one benchmark run."""
    return {
        "metrics": _serialize_metric_score(score),
        "failures": _serialize_ledger_summary(ledger_summary),
        "latency": _serialize_latency_report(latency, warm_only=warm_only),
        "config": asdict(config),
    }


def write_eval_report(output_path: str | Path, report: dict[str, Any]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def load_benchmark_jsonl_rows(output_path: str | Path) -> list[dict[str, Any]]:
    path = Path(output_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if is_benchmark_config_line(payload):
            continue
        rows.append(payload)
    return rows


def record_from_jsonl_row(row: dict[str, Any]) -> EvalRecord:
    payload = {key: value for key, value in row.items() if key in _EVAL_RECORD_FIELDS}
    return EvalRecord.model_validate(payload)


def ledger_summary_from_benchmark_output(
    cases: list[EvalCase],
    output_path: str | Path,
) -> LedgerSummary:
    """Build a ledger summary, classifying JSONL executor errors as schema crashes."""
    from evals.ledger import LedgerEntry, ledger_entries, ledger_summary as summarize_ledger

    rows = load_benchmark_jsonl_rows(output_path)
    records = [record_from_jsonl_row(row) for row in rows]
    entries = ledger_entries(cases, records)
    entries_by_case = {entry.case_id: entry for entry in entries}
    for row in rows:
        case_id = row.get("case_id")
        executor_error = row.get("executor_error")
        if not isinstance(case_id, str) or not case_id:
            continue
        if not isinstance(executor_error, str) or not executor_error:
            continue
        entries_by_case[case_id] = LedgerEntry(
            case_id=case_id,
            stage="schema",
            reason="executor_error",
        )
    return summarize_ledger(list(entries_by_case.values()))


def build_gate_inputs_from_benchmark_output(
    cases: list[EvalCase],
    output_path: str | Path,
    *,
    warm_only: bool = True,
    expected_query_answers: dict[str, str] | None = None,
) -> tuple[MetricScore, LedgerSummary, LatencyReport]:
    """Score a benchmark JSONL run and return gate inputs."""
    from evals.latency import latency_report

    rows = load_benchmark_jsonl_rows(output_path)
    records = [record_from_jsonl_row(row) for row in rows]
    score = score_records(cases, records, expected_query_answers=expected_query_answers)
    summary = ledger_summary_from_benchmark_output(cases, output_path)
    latency = latency_report(rows, warm_only=warm_only)
    return score, summary, latency


def build_report_from_benchmark_output(
    cases: list[EvalCase],
    output_path: str | Path,
    *,
    warm_only: bool = True,
    expected_query_answers: dict[str, str] | None = None,
) -> dict[str, Any]:
    score, summary, latency = build_gate_inputs_from_benchmark_output(
        cases,
        output_path,
        warm_only=warm_only,
        expected_query_answers=expected_query_answers,
    )
    config = load_benchmark_config(output_path) or BenchmarkConfig()

    return build_eval_report(
        score,
        summary,
        latency,
        config,
        warm_only=warm_only,
    )
