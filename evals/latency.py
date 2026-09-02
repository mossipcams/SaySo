"""Warm latency percentile reporting for eval benchmark rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

_STAGE_MS_FIELDS = (
    "stt_ms",
    "retrieve_ms",
    "plan_ms",
    "resolve_ms",
    "validate_ms",
    "request_ms",
    "verify_ms",
)

# Shared EOS boundary fields written to JSONL for model comparison (unit 9.2).
# START: turn processing begins (eval text path; no STT).
# STOP plan_ms: ControlPlan ready.
# STOP request_ms: action_request sent.
# STOP verify_ms: state verification finished.
LATENCY_BOUNDARY_FIELDS = ("plan_ms", "request_ms", "verify_ms")


@dataclass(frozen=True)
class LatencyFieldStats:
    n: int
    median: float
    p95: float


@dataclass(frozen=True)
class LatencyReport:
    n: int
    median: float
    p95: float
    stages: dict[str, LatencyFieldStats]


def _is_warm_row(row: Mapping[str, Any]) -> bool:
    if row.get("cold_start") is True:
        return False
    if row.get("warmup") is True:
        return False
    return True


def _filter_rows(rows: list[Mapping[str, Any]], *, warm_only: bool) -> list[Mapping[str, Any]]:
    if not warm_only:
        return rows
    return [row for row in rows if _is_warm_row(row)]


def _nearest_rank_percentile(sorted_values: list[float], percentile: float) -> float:
    count = len(sorted_values)
    if count == 0:
        return 0.0
    rank = math.ceil(percentile / 100.0 * count)
    return sorted_values[rank - 1]


def _field_stats(values: list[float]) -> LatencyFieldStats:
    count = len(values)
    if count == 0:
        return LatencyFieldStats(n=0, median=0.0, p95=0.0)
    ordered = sorted(values)
    return LatencyFieldStats(
        n=count,
        median=_nearest_rank_percentile(ordered, 50.0),
        p95=_nearest_rank_percentile(ordered, 95.0),
    )


def timing_boundaries_from_stages(
    *,
    plan_stage_ms: float,
    resolve_stage_ms: float,
    validate_stage_ms: float,
    request_stage_ms: float,
    verify_stage_ms: float,
    stt_stage_ms: float = 0.0,
) -> dict[str, float]:
    """Map orchestrator stage timings to shared EOS comparison boundaries."""
    eos_to_plan = stt_stage_ms + plan_stage_ms
    eos_to_action_request = (
        eos_to_plan + resolve_stage_ms + validate_stage_ms + request_stage_ms
    )
    verified_eos_to_action = eos_to_action_request + verify_stage_ms
    return {
        "plan_ms": eos_to_plan,
        "request_ms": eos_to_action_request,
        "verify_ms": verified_eos_to_action,
    }


def boundary_ms(row: Mapping[str, Any], boundary: str) -> float | None:
    """Return one comparison boundary from a JSONL row when present."""
    if boundary not in LATENCY_BOUNDARY_FIELDS:
        msg = f"unknown latency boundary: {boundary}"
        raise ValueError(msg)
    raw = row.get(boundary)
    if raw is None:
        return None
    return float(raw)


def cold_readiness_report(rows: list[Mapping[str, Any]]) -> LatencyFieldStats:
    """Summarize model readiness on cold-start rows, separate from warm turn latency."""
    values: list[float] = []
    for row in rows:
        if row.get("cold_start") is not True:
            continue
        raw = row.get("readiness_ms")
        if raw is None:
            continue
        values.append(float(raw))
    return _field_stats(values)


def _numeric_values(rows: list[Mapping[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.get(field)
        if raw is None:
            continue
        values.append(float(raw))
    return values


def latency_report(rows: list[Mapping[str, Any]], *, warm_only: bool = True) -> LatencyReport:
    """Summarize ``total_ms`` and present stage timings with nearest-rank percentiles."""
    filtered = _filter_rows(rows, warm_only=warm_only)
    total_stats = _field_stats(_numeric_values(filtered, "total_ms"))
    stages: dict[str, LatencyFieldStats] = {}
    for field in _STAGE_MS_FIELDS:
        stats = _field_stats(_numeric_values(filtered, field))
        if stats.n > 0:
            stages[field] = stats
    return LatencyReport(
        n=total_stats.n,
        median=total_stats.median,
        p95=total_stats.p95,
        stages=stages,
    )
