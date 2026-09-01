"""SaySo evaluation datasets, schema, and benchmark tooling."""

from evals.config import BenchmarkConfig, DEFAULT_MODEL_ID
from evals.executor import controller_dry_run_executor, execute_controller_dry_run
from evals.metrics import EvalRecord, MetricScore, canonicalize_control_plan, score_records
from evals.mlx_executor import (
    MLX_EVAL_ENV_VAR,
    build_mlx_model_runtime,
    controller_mlx_executor,
    is_mlx_eval_enabled,
    is_mlx_lm_available,
    resolve_eval_executor,
)
from evals.gate import expansion_allowed
from evals.report import build_eval_report, build_report_from_benchmark_output, write_eval_report
from evals.runner import (
    BenchmarkRunResult,
    CaseExecutionResult,
    CaseExecutor,
    CaseTiming,
    dry_run_executor,
    gate_executor_for_live_safety,
    load_output_case_ids,
    run_benchmark,
)
from evals.schema import EvalCase, EvalSchemaError, ExpectedOutcome, load_eval_cases_jsonl, parse_eval_case

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRunResult",
    "build_eval_report",
    "build_report_from_benchmark_output",
    "DEFAULT_MODEL_ID",
    "MLX_EVAL_ENV_VAR",
    "CaseExecutionResult",
    "CaseExecutor",
    "CaseTiming",
    "EvalCase",
    "EvalRecord",
    "EvalSchemaError",
    "ExpectedOutcome",
    "MetricScore",
    "build_mlx_model_runtime",
    "canonicalize_control_plan",
    "controller_dry_run_executor",
    "controller_mlx_executor",
    "dry_run_executor",
    "execute_controller_dry_run",
    "expansion_allowed",
    "gate_executor_for_live_safety",
    "is_mlx_eval_enabled",
    "is_mlx_lm_available",
    "load_eval_cases_jsonl",
    "load_output_case_ids",
    "parse_eval_case",
    "resolve_eval_executor",
    "run_benchmark",
    "score_records",
    "write_eval_report",
]
