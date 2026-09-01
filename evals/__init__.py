"""SaySo evaluation datasets, schema, and benchmark tooling."""

from evals.metrics import EvalRecord, MetricScore, canonicalize_control_plan, score_records
from evals.schema import EvalCase, EvalSchemaError, ExpectedOutcome, load_eval_cases_jsonl, parse_eval_case

__all__ = [
    "EvalCase",
    "EvalRecord",
    "EvalSchemaError",
    "ExpectedOutcome",
    "MetricScore",
    "canonicalize_control_plan",
    "load_eval_cases_jsonl",
    "parse_eval_case",
    "score_records",
]
