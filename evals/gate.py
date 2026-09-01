"""Expansion gate for corpus growth after reliability baseline checks."""

from __future__ import annotations

from evals.latency import LatencyReport
from evals.ledger import LedgerSummary
from evals.metrics import MetricScore

_CLASSIFIED_SCHEMA_REASONS = frozenset({"schema_failure", "executor_error"})


def expansion_allowed(
    score: MetricScore,
    ledger_summary: LedgerSummary,
    latency: LatencyReport,
) -> tuple[bool, list[str]]:
    """Return whether corpus expansion is allowed and human-readable block reasons."""
    reasons: list[str] = []

    if score.false_execution_denominator == 0:
        reasons.append("false_execution_denominator is 0 (fail closed)")
    elif score.false_execution_rate != 0.0:
        reasons.append(
            "false_execution_rate must be 0 "
            f"(got {score.false_execution_rate} "
            f"with {score.false_execution_numerator}/"
            f"{score.false_execution_denominator})",
        )

    if score.wrong_device_rate != 0.0:
        reasons.append(
            "wrong_device_rate must be 0 "
            f"(got {score.wrong_device_rate} "
            f"with {score.wrong_device_numerator}/"
            f"{score.wrong_device_denominator})",
        )

    if latency.n < 1:
        reasons.append(f"latency.n must be >= 1 (got {latency.n})")

    for item in ledger_summary.by_stage_reason:
        if item.stage != "schema":
            continue
        if item.reason == "executor_error":
            reasons.append(
                "schema executor crash on "
                + ", ".join(item.case_ids),
            )
        elif item.reason not in _CLASSIFIED_SCHEMA_REASONS:
            reasons.append(
                "unclassified schema executor crash "
                f"({item.reason}) on "
                + ", ".join(item.case_ids),
            )

    return (not reasons, reasons)
