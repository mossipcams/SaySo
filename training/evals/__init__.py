"""SaySo training evaluation package."""

from .harness import evaluate_checkpoint, load_eval_jsonl, write_results
from .llamacpp import parse_chat_completion
from .metrics import MetricSummary, normalize_json_value, score_tool_call_protocol
from .v3_quality import excluded_train_prompts, gold_user_prompts

__all__ = [
    "MetricSummary",
    "evaluate_checkpoint",
    "excluded_train_prompts",
    "gold_user_prompts",
    "load_eval_jsonl",
    "normalize_json_value",
    "parse_chat_completion",
    "score_tool_call_protocol",
    "write_results",
]
