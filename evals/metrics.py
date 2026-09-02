"""Aggregate offline evaluation metrics for request size and tool quality."""

from __future__ import annotations

from typing import Any

REQUIRED_METRIC_FIELDS: frozenset[str] = frozenset(
    {
        "serialized_request_bytes",
        "prompt_tokens",
        "tool_accuracy",
        "invalid_call_rate",
        "latency_ms",
    }
)

REQUIRED_LATENCY_FIELDS: frozenset[str] = frozenset({"p50", "p95"})

REQUIRED_LIVE_LATENCY_FIELDS: frozenset[str] = frozenset(
    {"warmups", "repetitions", "ttft_ms", "end_to_end_ms"}
)

LATENCY_TOLERANCE_P50_FRACTION: float = 0.10
LATENCY_TOLERANCE_P95_FRACTION: float = 0.15
LATENCY_TOLERANCE_P50_FLOOR_MS: float = 5.0
LATENCY_TOLERANCE_P95_FLOOR_MS: float = 10.0


def validate_metrics_report(metrics: dict[str, Any]) -> None:
    """Reject reports that omit required metric fields."""
    missing = sorted(REQUIRED_METRIC_FIELDS - metrics.keys())
    if missing:
        raise ValueError(f"missing metric fields: {missing}")

    latency = metrics.get("latency_ms")
    if not isinstance(latency, dict):
        raise ValueError("latency_ms must be a mapping")
    missing_latency = sorted(REQUIRED_LATENCY_FIELDS - latency.keys())
    if missing_latency:
        raise ValueError(f"missing latency_ms fields: {missing_latency}")


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def compute_latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """Return deterministic p50 and p95 latency values in milliseconds."""
    ordered = sorted(latencies_ms)
    return {
        "p50": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
    }


def validate_live_latency_report(report: dict[str, Any]) -> None:
    """Reject live latency reports that omit required fields."""
    missing = sorted(REQUIRED_LIVE_LATENCY_FIELDS - report.keys())
    if missing:
        raise ValueError(f"missing live latency fields: {missing}")

    for field in ("ttft_ms", "end_to_end_ms"):
        latency = report.get(field)
        if not isinstance(latency, dict):
            raise ValueError(f"{field} must be a mapping")
        missing_latency = sorted(REQUIRED_LATENCY_FIELDS - latency.keys())
        if missing_latency:
            raise ValueError(f"missing {field} fields: {missing_latency}")


def build_live_latency_report(
    *,
    ttft_samples_ms: list[float],
    end_to_end_samples_ms: list[float],
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    """Build a validated live latency report with median and p95 values."""
    report = {
        "warmups": warmups,
        "repetitions": repetitions,
        "ttft_ms": compute_latency_percentiles(ttft_samples_ms),
        "end_to_end_ms": compute_latency_percentiles(end_to_end_samples_ms),
    }
    validate_live_latency_report(report)
    return report


def derive_latency_tolerance_ms(live_latency: dict[str, Any]) -> dict[str, float]:
    """Return fixed latency slack derived only from a baseline live latency report."""
    validate_live_latency_report(live_latency)
    ttft = live_latency["ttft_ms"]
    end_to_end = live_latency["end_to_end_ms"]
    return {
        "ttft_p50": max(
            LATENCY_TOLERANCE_P50_FLOOR_MS,
            float(ttft["p50"]) * LATENCY_TOLERANCE_P50_FRACTION,
        ),
        "ttft_p95": max(
            LATENCY_TOLERANCE_P95_FLOOR_MS,
            float(ttft["p95"]) * LATENCY_TOLERANCE_P95_FRACTION,
        ),
        "end_to_end_p50": max(
            LATENCY_TOLERANCE_P50_FLOOR_MS,
            float(end_to_end["p50"]) * LATENCY_TOLERANCE_P50_FRACTION,
        ),
        "end_to_end_p95": max(
            LATENCY_TOLERANCE_P95_FLOOR_MS,
            float(end_to_end["p95"]) * LATENCY_TOLERANCE_P95_FRACTION,
        ),
    }


def build_confident_routing_metrics(
    *,
    prompt_tokens_by_case: dict[str, int | None],
    confidently_routed_by_case: dict[str, bool],
) -> dict[str, Any]:
    """Aggregate prompt-token usage for confidently routed eval cases."""
    by_case: dict[str, int] = {}
    for case_id, routed in confidently_routed_by_case.items():
        if not routed:
            continue
        tokens = prompt_tokens_by_case.get(case_id)
        if tokens is None:
            continue
        by_case[case_id] = tokens
    return {
        "case_count": len(by_case),
        "prompt_tokens_total": sum(by_case.values()),
        "prompt_tokens_by_case": dict(sorted(by_case.items())),
    }


def compute_tool_accuracy(
    *,
    tool_case_count: int,
    tool_case_passed: int,
) -> float:
    if tool_case_count == 0:
        return 1.0
    return tool_case_passed / tool_case_count


def compute_invalid_call_rate(
    *,
    tool_call_case_count: int,
    invalid_call_case_count: int,
) -> float:
    if tool_call_case_count == 0:
        return 0.0
    return invalid_call_case_count / tool_call_case_count


def build_metrics_report(
    *,
    serialized_request_bytes: list[int],
    prompt_tokens: list[int | None],
    tool_case_count: int,
    tool_case_passed: int,
    tool_call_case_count: int,
    invalid_call_case_count: int,
    latencies_ms: list[float],
) -> dict[str, Any]:
    """Build a validated metrics report from per-case measurements."""
    usage_prompt_tokens = [value for value in prompt_tokens if value is not None]
    report = {
        "serialized_request_bytes": {
            "total": sum(serialized_request_bytes),
            "mean": (
                sum(serialized_request_bytes) / len(serialized_request_bytes)
                if serialized_request_bytes
                else 0.0
            ),
        },
        "prompt_tokens": {
            "total": sum(usage_prompt_tokens),
            "mean": (
                sum(usage_prompt_tokens) / len(usage_prompt_tokens)
                if usage_prompt_tokens
                else 0.0
            ),
            "missing_usage_count": len(prompt_tokens) - len(usage_prompt_tokens),
        },
        "tool_accuracy": compute_tool_accuracy(
            tool_case_count=tool_case_count,
            tool_case_passed=tool_case_passed,
        ),
        "invalid_call_rate": compute_invalid_call_rate(
            tool_call_case_count=tool_call_case_count,
            invalid_call_case_count=invalid_call_case_count,
        ),
        "latency_ms": compute_latency_percentiles(latencies_ms),
    }
    validate_metrics_report(report)
    return report


def compare_metric_reports(
    baseline: dict[str, Any],
    improved: dict[str, Any],
) -> dict[str, Any]:
    """Compare two metric reports after validating required fields."""
    validate_metrics_report(baseline)
    validate_metrics_report(improved)

    baseline_bytes = int(baseline["serialized_request_bytes"]["total"])
    improved_bytes = int(improved["serialized_request_bytes"]["total"])
    baseline_prompt = int(baseline["prompt_tokens"]["total"])
    improved_prompt = int(improved["prompt_tokens"]["total"])
    baseline_latency = baseline["latency_ms"]
    improved_latency = improved["latency_ms"]

    return {
        "baseline": baseline,
        "improved": improved,
        "delta": {
            "serialized_request_bytes_total": improved_bytes - baseline_bytes,
            "prompt_tokens_total": improved_prompt - baseline_prompt,
            "tool_accuracy": float(improved["tool_accuracy"])
            - float(baseline["tool_accuracy"]),
            "invalid_call_rate": float(improved["invalid_call_rate"])
            - float(baseline["invalid_call_rate"]),
            "latency_ms_p50": float(improved_latency["p50"])
            - float(baseline_latency["p50"]),
            "latency_ms_p95": float(improved_latency["p95"])
            - float(baseline_latency["p95"]),
        },
    }
