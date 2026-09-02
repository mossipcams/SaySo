"""Evaluation harness for SaySo checkpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .llamacpp import parse_chat_completion
from .metrics import ExampleScore, MetricSummary, score_expected_vs_actual, score_tool_call_protocol, summarize_scores


def load_eval_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load evaluation examples from JSONL."""
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def evaluate_checkpoint(
    examples: list[dict[str, Any]],
    infer_fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    checkpoint_id: str = "base",
) -> MetricSummary:
    """Run inference over examples and score results."""
    scores: list[ExampleScore] = []

    for example in examples:
        started = time.perf_counter()
        try:
            response_body = infer_fn(example)
            parsed = parse_chat_completion(response_body)
            actual_message = parsed["message"]
        except Exception as exc:  # noqa: BLE001 - harness records all failures
            scores.append(
                ExampleScore(
                    expected=example,
                    actual={"error": str(exc)},
                    failure_category="inference_error",
                    checkpoint_id=checkpoint_id,
                )
            )
            continue

        latency_ms = (time.perf_counter() - started) * 1000.0
        expected_messages = example.get("messages") or []
        actual_messages = [*expected_messages, actual_message]

        protocol_ok = score_tool_call_protocol([actual_message])
        _, _, _, category = score_expected_vs_actual(expected_messages, actual_messages)

        if not protocol_ok:
            category = category or "protocol_invalid"

        scores.append(
            ExampleScore(
                expected=example,
                actual=actual_message,
                failure_category=category,
                latency_ms=latency_ms,
                checkpoint_id=checkpoint_id,
            )
        )

    return summarize_scores(scores)


def write_results(path: Path, summary: MetricSummary) -> None:
    """Write per-example results and aggregate rates."""
    payload = {
        "rates": summary.rates(),
        "total": summary.total,
        "examples": [
            {
                "failure_category": item.failure_category,
                "latency_ms": item.latency_ms,
                "checkpoint_id": item.checkpoint_id,
                "expected": item.expected,
                "actual": item.actual,
            }
            for item in summary.per_example
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
