"""Load and run offline evaluation cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.metrics import build_confident_routing_metrics, build_metrics_report, derive_latency_tolerance_ms
from evals.scorer import (
    CheckName,
    EvalActual,
    EvalCase,
    case_score_to_dict,
    score_case,
)

_TOOL_QUALITY_CHECKS = frozenset(
    {
        CheckName.TOOL_NAME,
        CheckName.TOOL_ARGS,
        CheckName.TOOL_ORDER,
        CheckName.WRONG_TOOL,
    }
)


@dataclass(frozen=True, slots=True)
class EvalCaseSet:
    """Versioned collection of offline evaluation cases."""

    version: int
    cases: tuple[EvalCase, ...]


@dataclass(frozen=True, slots=True)
class EvalRecord:
    """Recorded llama.cpp outcome plus request and latency measurements."""

    actual: EvalActual
    request_payload: dict[str, Any]
    request_bytes: int
    prompt_tokens: int | None = None
    latency_ms: float = 0.0
    confidently_routed: bool = False


def _parse_checks(raw_checks: Any) -> tuple[CheckName, ...]:
    if not isinstance(raw_checks, list):
        return ()
    checks: list[CheckName] = []
    for item in raw_checks:
        checks.append(CheckName(str(item)))
    return tuple(checks)


def _parse_case(raw: dict[str, Any]) -> EvalCase:
    return EvalCase(
        id=str(raw["id"]),
        category=str(raw["category"]),
        scenario=str(raw["scenario"]),
        description=str(raw.get("description", "")),
        expect=dict(raw.get("expect", {})),
        checks=_parse_checks(raw.get("checks")),
    )


def load_cases(path: str | Path) -> EvalCaseSet:
    """Load a versioned JSON case set from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(payload["version"])
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("cases must be a list")
    cases = tuple(_parse_case(entry) for entry in raw_cases if isinstance(entry, dict))
    return EvalCaseSet(version=version, cases=cases)


def run_eval(
    case_set: EvalCaseSet,
    actuals: dict[str, EvalActual],
) -> dict[str, Any]:
    """Score all cases and return a deterministic JSON report."""
    results: list[dict[str, Any]] = []
    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for case in sorted(case_set.cases, key=lambda item: item.id):
        actual = actuals.get(case.id)
        if actual is None:
            skipped_count += 1
            results.append(
                {
                    "case_id": case.id,
                    "category": case.category,
                    "scenario": case.scenario,
                    "status": "skipped",
                    "passed": None,
                    "checks": {},
                }
            )
            continue

        score = score_case(case, actual)
        if score.passed:
            passed_count += 1
            status = "passed"
        else:
            failed_count += 1
            status = "failed"

        entry = case_score_to_dict(score)
        entry["status"] = status
        results.append(entry)

    return {
        "version": case_set.version,
        "summary": {
            "total": len(case_set.cases),
            "passed": passed_count,
            "failed": failed_count,
            "skipped": skipped_count,
        },
        "results": results,
    }


def _tool_quality_checks(case: EvalCase) -> tuple[CheckName, ...]:
    return tuple(check for check in case.checks if check in _TOOL_QUALITY_CHECKS)


def _aggregate_metrics(
    case_set: EvalCaseSet,
    records: dict[str, EvalRecord],
    report: dict[str, Any],
) -> dict[str, Any]:
    serialized_request_bytes: list[int] = []
    prompt_tokens: list[int | None] = []
    latencies_ms: list[float] = []
    tool_case_count = 0
    tool_case_passed = 0
    tool_call_case_count = 0
    invalid_call_case_count = 0

    results_by_id = {
        str(entry["case_id"]): entry for entry in report.get("results", [])
    }

    for case in sorted(case_set.cases, key=lambda item: item.id):
        record = records.get(case.id)
        if record is None:
            continue

        serialized_request_bytes.append(record.request_bytes)
        prompt_tokens.append(record.prompt_tokens)
        latencies_ms.append(record.latency_ms)

        if record.actual.tool_calls:
            tool_call_case_count += 1
            if record.actual.validation_errors:
                invalid_call_case_count += 1

        quality_checks = _tool_quality_checks(case)
        if quality_checks:
            tool_case_count += 1
            result = results_by_id.get(case.id, {})
            checks = result.get("checks", {})
            if all(
                isinstance(checks.get(check.value), dict)
                and checks[check.value].get("passed") is True
                for check in quality_checks
            ):
                tool_case_passed += 1

    return build_metrics_report(
        serialized_request_bytes=serialized_request_bytes,
        prompt_tokens=prompt_tokens,
        tool_case_count=tool_case_count,
        tool_case_passed=tool_case_passed,
        tool_call_case_count=tool_call_case_count,
        invalid_call_case_count=invalid_call_case_count,
        latencies_ms=latencies_ms,
    )


def run_eval_with_metrics(
    case_set: EvalCaseSet,
    records: dict[str, EvalRecord],
) -> dict[str, Any]:
    """Score cases and attach aggregate request/tool/latency metrics."""
    actuals = {case_id: record.actual for case_id, record in records.items()}
    report = run_eval(case_set, actuals)
    report["metrics"] = _aggregate_metrics(case_set, records, report)
    return report


def _aggregate_confident_routing(
    records: dict[str, EvalRecord],
) -> dict[str, Any]:
    prompt_tokens_by_case = {
        case_id: record.prompt_tokens for case_id, record in records.items()
    }
    confidently_routed_by_case = {
        case_id: record.confidently_routed for case_id, record in records.items()
    }
    return build_confident_routing_metrics(
        prompt_tokens_by_case=prompt_tokens_by_case,
        confidently_routed_by_case=confidently_routed_by_case,
    )


def build_release_report(
    case_set: EvalCaseSet,
    records: dict[str, EvalRecord],
    *,
    matrix_id: str,
    metadata: dict[str, Any],
    fingerprints: dict[str, Any],
    live_latency: dict[str, Any],
    latency_explanations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a release-ready eval report with metadata, metrics, and tolerances."""
    report = run_eval_with_metrics(case_set, records)
    report["matrix_id"] = matrix_id
    report["metadata"] = dict(metadata)
    report["fingerprints"] = dict(fingerprints)
    report["confident_routing"] = _aggregate_confident_routing(records)
    report["live_latency"] = live_latency
    report["latency_tolerance_ms"] = derive_latency_tolerance_ms(live_latency)
    if latency_explanations:
        report["latency_explanations"] = dict(latency_explanations)
    return report
