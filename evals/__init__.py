"""SaySo evaluation datasets, schema, and benchmark tooling."""

from evals.schema import EvalCase, EvalSchemaError, ExpectedOutcome, load_eval_cases_jsonl, parse_eval_case

__all__ = [
    "EvalCase",
    "EvalSchemaError",
    "ExpectedOutcome",
    "load_eval_cases_jsonl",
    "parse_eval_case",
]
