"""Compare baseline and candidate evaluation reports with release gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.metrics import validate_live_latency_report, validate_metrics_report

REQUIRED_METADATA_FIELDS: frozenset[str] = frozenset(
    {
        "homeassistant",
        "llama_cpp",
        "model",
        "chat_template",
        "hardware",
        "warmups",
        "repetitions",
    }
)

REQUIRED_RELEASE_FIELDS: frozenset[str] = frozenset(
    {
        "matrix_id",
        "metadata",
        "fingerprints",
        "metrics",
        "confident_routing",
        "live_latency",
        "latency_tolerance_ms",
    }
)

REQUIRED_FINGERPRINT_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "gguf_sha256",
        "llama_server_args",
        "cases_file",
        "cases_version",
    }
)

_LATENCY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("ttft_latency", "ttft_ms", "ttft"),
    ("end_to_end_latency", "end_to_end_ms", "end_to_end"),
)


def validate_release_report(report: dict[str, Any]) -> None:
    """Reject release reports that omit required metadata, metrics, or tolerances."""
    missing = sorted(REQUIRED_RELEASE_FIELDS - report.keys())
    if missing:
        raise ValueError(f"missing release report fields: {missing}")

    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    missing_metadata = sorted(REQUIRED_METADATA_FIELDS - metadata.keys())
    if missing_metadata:
        raise ValueError(f"missing metadata fields: {missing_metadata}")

    fingerprints = report.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("fingerprints must be a mapping")
    missing_fingerprints = sorted(REQUIRED_FINGERPRINT_FIELDS - fingerprints.keys())
    if missing_fingerprints:
        raise ValueError(f"missing fingerprint fields: {missing_fingerprints}")

    validate_metrics_report(report["metrics"])
    validate_live_latency_report(report["live_latency"])

    tolerance = report.get("latency_tolerance_ms")
    if not isinstance(tolerance, dict):
        raise ValueError("latency_tolerance_ms must be a mapping")
    required_tolerance = {
        "ttft_p50",
        "ttft_p95",
        "end_to_end_p50",
        "end_to_end_p95",
    }
    missing_tolerance = sorted(required_tolerance - tolerance.keys())
    if missing_tolerance:
        raise ValueError(f"missing latency_tolerance_ms fields: {missing_tolerance}")

    confident = report.get("confident_routing")
    if not isinstance(confident, dict):
        raise ValueError("confident_routing must be a mapping")
    if "prompt_tokens_total" not in confident:
        raise ValueError("confident_routing missing prompt_tokens_total")


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Load and validate an archived baseline report."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("baseline report must be a JSON object")
    validate_release_report(report)
    return report


def _metadata_mismatches(
    baseline_metadata: dict[str, Any],
    candidate_metadata: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    for field in sorted(REQUIRED_METADATA_FIELDS):
        if baseline_metadata.get(field) != candidate_metadata.get(field):
            mismatches.append(field)
    return mismatches


def _metric_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    baseline_latency = baseline_metrics["latency_ms"]
    candidate_latency = candidate_metrics["latency_ms"]
    return {
        "serialized_request_bytes_total": float(
            candidate_metrics["serialized_request_bytes"]["total"]
            - baseline_metrics["serialized_request_bytes"]["total"]
        ),
        "prompt_tokens_total": float(
            candidate_metrics["prompt_tokens"]["total"]
            - baseline_metrics["prompt_tokens"]["total"]
        ),
        "tool_accuracy": float(candidate_metrics["tool_accuracy"])
        - float(baseline_metrics["tool_accuracy"]),
        "invalid_call_rate": float(candidate_metrics["invalid_call_rate"])
        - float(baseline_metrics["invalid_call_rate"]),
        "latency_ms_p50": float(candidate_latency["p50"]) - float(baseline_latency["p50"]),
        "latency_ms_p95": float(candidate_latency["p95"]) - float(baseline_latency["p95"]),
        "confident_routing_prompt_tokens_total": float(
            candidate["confident_routing"]["prompt_tokens_total"]
            - baseline["confident_routing"]["prompt_tokens_total"]
        ),
    }


def _check_latency_gate(
    *,
    gate_name: str,
    live_key: str,
    tolerance_prefix: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    explanations: dict[str, str],
) -> dict[str, Any]:
    baseline_live = baseline["live_latency"][live_key]
    candidate_live = candidate["live_latency"][live_key]
    tolerance = baseline["latency_tolerance_ms"]
    regressions: list[str] = []
    explained: list[str] = []

    for percentile in ("p50", "p95"):
        tolerance_key = f"{tolerance_prefix}_{percentile}"
        delta = float(candidate_live[percentile]) - float(baseline_live[percentile])
        allowed = float(tolerance[tolerance_key])
        if delta > allowed:
            if tolerance_key in explanations:
                explained.append(tolerance_key)
            else:
                regressions.append(
                    f"{tolerance_key}: delta={delta:.3f} allowed={allowed:.3f}"
                )

    return {
        "passed": not regressions,
        "explained": bool(explained),
        "regressions": regressions,
        "explained_keys": explained,
        "baseline": dict(baseline_live),
        "candidate": dict(candidate_live),
        "tolerance_ms": {
            f"{tolerance_prefix}_p50": float(tolerance[f"{tolerance_prefix}_p50"]),
            f"{tolerance_prefix}_p95": float(tolerance[f"{tolerance_prefix}_p95"]),
        },
        "gate": gate_name,
    }


def compare_eval_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare baseline and candidate reports and evaluate release gates."""
    validate_release_report(baseline)
    validate_release_report(candidate)

    metadata_mismatches = _metadata_mismatches(
        baseline["metadata"],
        candidate["metadata"],
    )
    metadata_gate = {
        "passed": not metadata_mismatches,
        "mismatches": metadata_mismatches,
    }

    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    tool_accuracy_gate = {
        "passed": float(candidate_metrics["tool_accuracy"])
        >= float(baseline_metrics["tool_accuracy"]),
        "baseline": float(baseline_metrics["tool_accuracy"]),
        "candidate": float(candidate_metrics["tool_accuracy"]),
    }
    invalid_call_gate = {
        "passed": float(candidate_metrics["invalid_call_rate"])
        <= float(baseline_metrics["invalid_call_rate"]),
        "baseline": float(baseline_metrics["invalid_call_rate"]),
        "candidate": float(candidate_metrics["invalid_call_rate"]),
    }
    confident_gate = {
        "passed": int(candidate["confident_routing"]["prompt_tokens_total"])
        <= int(baseline["confident_routing"]["prompt_tokens_total"]),
        "baseline": int(baseline["confident_routing"]["prompt_tokens_total"]),
        "candidate": int(candidate["confident_routing"]["prompt_tokens_total"]),
    }

    explanations = dict(candidate.get("latency_explanations", {}))
    latency_gates = {
        gate_name: _check_latency_gate(
            gate_name=gate_name,
            live_key=live_key,
            tolerance_prefix=tolerance_prefix,
            baseline=baseline,
            candidate=candidate,
            explanations=explanations,
        )
        for gate_name, live_key, tolerance_prefix in _LATENCY_METRICS
    }

    gates = {
        "metadata_match": metadata_gate,
        "tool_accuracy": tool_accuracy_gate,
        "invalid_call_rate": invalid_call_gate,
        "confident_routing_prompt_tokens": confident_gate,
        **latency_gates,
    }
    passed = all(gate["passed"] for gate in gates.values())

    return {
        "passed": passed,
        "baseline_matrix_id": baseline["matrix_id"],
        "candidate_matrix_id": candidate["matrix_id"],
        "gates": gates,
        "delta": _metric_delta(baseline, candidate),
    }


def format_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render a compact Markdown summary of a comparison result."""
    lines = [
        "# Eval release comparison",
        "",
        f"- passed: `{comparison['passed']}`",
        f"- baseline: `{comparison['baseline_matrix_id']}`",
        f"- candidate: `{comparison['candidate_matrix_id']}`",
        "",
        "| gate | passed | detail |",
        "| --- | --- | --- |",
    ]
    for gate_name, gate in comparison["gates"].items():
        if gate_name == "metadata_match":
            detail = ", ".join(gate["mismatches"]) or "match"
        elif gate_name in {"tool_accuracy", "invalid_call_rate", "confident_routing_prompt_tokens"}:
            detail = f"baseline={gate['baseline']} candidate={gate['candidate']}"
        elif "regressions" in gate:
            detail = "; ".join(gate["regressions"]) or "within tolerance"
            if gate.get("explained"):
                detail = f"explained ({', '.join(gate['explained_keys'])})"
        else:
            detail = ""
        lines.append(f"| {gate_name} | {gate['passed']} | {detail} |")

    lines.extend(
        [
            "",
            "## deltas",
            "",
        ]
    )
    for key, value in comparison["delta"].items():
        lines.append(f"- {key}: {value:+.3f}")
    return "\n".join(lines)


def format_comparison_json(comparison: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON-serializable comparison payload."""
    return {
        "passed": comparison["passed"],
        "baseline_matrix_id": comparison["baseline_matrix_id"],
        "candidate_matrix_id": comparison["candidate_matrix_id"],
        "gates": {
            name: {
                key: value
                for key, value in gate.items()
                if key not in {"baseline", "candidate", "tolerance_ms"}
            }
            for name, gate in comparison["gates"].items()
        },
        "delta": comparison["delta"],
    }
