"""Tests for the evaluation harness."""

from __future__ import annotations

from evals.harness import evaluate_checkpoint
from evals.recipe_lock import build_quality_eval_examples, expected_tool_calls


def _example_with_tool_calls() -> dict:
    return next(row for row in build_quality_eval_examples() if expected_tool_calls(row))


def _chat_completion_for_tool_calls(calls: list[dict]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.get("id", f"call_{index + 1}"),
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                            },
                        }
                        for index, call in enumerate(calls)
                    ],
                }
            }
        ]
    }


def test_evaluate_checkpoint_passes_identical_tool_call_prediction() -> None:
    """Regression: score model message only, not [*gold_messages, actual]."""
    example = _example_with_tool_calls()
    calls = expected_tool_calls(example)

    def infer_fn(_example: dict) -> dict:
        return _chat_completion_for_tool_calls(calls)

    summary = evaluate_checkpoint([example], infer_fn)
    assert summary.total == 1
    assert summary.per_example[0].failure_category is None
