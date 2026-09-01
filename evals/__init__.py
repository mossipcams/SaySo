"""SaySo evaluation datasets, schema, and benchmark tooling."""

from evals.metrics import EvalRecord, MetricScore, canonicalize_control_plan, score_records
from evals.runner import (
    BenchmarkRunResult,
    CaseExecutionResult,
    CaseExecutor,
    CaseTiming,
    dry_run_executor,
    load_output_case_ids,
    run_benchmark,
)
from evals.schema import EvalCase, EvalSchemaError, ExpectedOutcome, load_eval_cases_jsonl, parse_eval_case

__all__ = [
    "BenchmarkRunResult",
    "CaseExecutionResult",
    "CaseExecutor",
    "CaseTiming",
    "EvalCase",
    "EvalRecord",
    "EvalSchemaError",
    "ExpectedOutcome",
    "MetricScore",
    "canonicalize_control_plan",
    "dry_run_executor",
    "load_eval_cases_jsonl",
    "load_output_case_ids",
    "parse_eval_case",
    "run_benchmark",
    "score_records",
]
