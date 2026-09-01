"""Seeded, resumable JSONL benchmark runner for SaySo evaluation cases."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from evals.config import BenchmarkConfig, is_benchmark_config_line, write_benchmark_config_header
from evals.metrics import EvalRecord
from evals.schema import EvalCase, load_eval_cases_jsonl


@dataclass(frozen=True)
class CaseTiming:
    total_ms: float
    stt_ms: float | None = None
    retrieve_ms: float | None = None
    plan_ms: float | None = None
    resolve_ms: float | None = None
    validate_ms: float | None = None
    request_ms: float | None = None
    verify_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model_id: str | None = None


@dataclass(frozen=True)
class CaseExecutionResult:
    record: EvalRecord
    timing: CaseTiming


class CaseExecutor(Protocol):
    def __call__(self, case: EvalCase) -> CaseExecutionResult: ...


@dataclass(frozen=True)
class BenchmarkRunResult:
    scored: int
    skipped: int
    warmup_runs: int
    errors: int


def dry_run_executor(case: EvalCase) -> CaseExecutionResult:
    """Default executor: never actuates Home Assistant."""
    start = time.perf_counter()
    record = EvalRecord(case_id=case.case_id, ha_executed=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return CaseExecutionResult(record=record, timing=CaseTiming(total_ms=elapsed_ms))


def mark_non_live_executor(executor: CaseExecutor) -> CaseExecutor:
    """Mark ``executor`` as safe to run without live Home Assistant preflight."""
    executor.live_actuation = False  # type: ignore[attr-defined]
    return executor


def _requires_live_actuation_gate(executor: CaseExecutor) -> bool:
    live_actuation = getattr(executor, "live_actuation", None)
    if live_actuation is False:
        return False
    if live_actuation is True:
        return True
    return executor is not dry_run_executor


mark_non_live_executor(dry_run_executor)


def _live_actuation_preflight_permitted(
    *,
    execute: bool,
    entity_allowlist: frozenset[str],
    case: EvalCase,
) -> bool:
    if not execute or not entity_allowlist:
        return False
    targets = frozenset(case.expected_resolved_entities)
    if not targets:
        return False
    return targets <= entity_allowlist


def gate_executor_for_live_safety(
    executor: CaseExecutor,
    *,
    execute: bool = False,
    entity_allowlist: frozenset[str] | set[str] | list[str] | tuple[str, ...] = (),
) -> CaseExecutor:
    """Wrap ``executor`` so live Home Assistant actuation runs only when safeguards pass."""
    allowlist = frozenset(entity_allowlist)

    def gated(case: EvalCase) -> CaseExecutionResult:
        if _live_actuation_preflight_permitted(
            execute=execute,
            entity_allowlist=allowlist,
            case=case,
        ):
            return executor(case)
        if not _requires_live_actuation_gate(executor):
            return executor(case)
        return dry_run_executor(case)

    return gated


def load_output_case_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    case_ids: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if is_benchmark_config_line(payload):
            continue
        case_ids.add(str(payload["case_id"]))
    return case_ids


def _resolve_cases(cases: list[EvalCase] | str | Path) -> list[EvalCase]:
    if isinstance(cases, list):
        return cases
    path = Path(cases)
    return load_eval_cases_jsonl(path.read_text(encoding="utf-8"))


def _record_to_jsonl(
    record: EvalRecord,
    timing: CaseTiming,
    *,
    executor_error: str | None = None,
    cold_start: bool | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = record.model_dump(mode="json")
    payload["total_ms"] = timing.total_ms
    for field in (
        "stt_ms",
        "retrieve_ms",
        "plan_ms",
        "resolve_ms",
        "validate_ms",
        "request_ms",
        "verify_ms",
        "prompt_tokens",
        "completion_tokens",
        "model_id",
    ):
        value = getattr(timing, field)
        if value is not None:
            payload[field] = value
    if executor_error is not None:
        payload["executor_error"] = executor_error
    if cold_start is True:
        payload["cold_start"] = True
    return payload


def _run_warmup(
    executor: CaseExecutor,
    *,
    warmup_count: int,
    warmup_case: EvalCase | None,
    fallback_case: EvalCase | None,
) -> int:
    if warmup_count <= 0:
        return 0
    target = warmup_case or fallback_case
    if target is None:
        return 0
    for _ in range(warmup_count):
        try:
            executor(target)
        except Exception:
            continue
    return warmup_count


def run_benchmark(
    cases: list[EvalCase] | str | Path,
    output_path: str | Path,
    executor: CaseExecutor | None = None,
    *,
    config: BenchmarkConfig | None = None,
    seed: int = 0,
    warmup_count: int = 0,
    warmup_case: EvalCase | None = None,
    execute: bool = False,
    entity_allowlist: frozenset[str] | set[str] | list[str] | tuple[str, ...] = (),
) -> BenchmarkRunResult:
    """Run eval cases append-only to ``output_path``, skipping completed case IDs."""
    run_config = config or BenchmarkConfig(seed=seed, warmup_count=warmup_count)
    effective_seed = run_config.seed
    effective_warmup = run_config.warmup_count
    random.seed(effective_seed)
    inner_executor = executor or dry_run_executor
    run_executor = gate_executor_for_live_safety(
        inner_executor,
        execute=execute,
        entity_allowlist=entity_allowlist,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    write_benchmark_config_header(output, run_config)

    case_list = _resolve_cases(cases)
    completed = load_output_case_ids(output)
    warmup_runs = _run_warmup(
        run_executor,
        warmup_count=effective_warmup,
        warmup_case=warmup_case,
        fallback_case=case_list[0] if case_list else None,
    )

    cold_start_pending = run_config.cold_start and effective_warmup == 0
    scored = 0
    skipped = 0
    errors = 0
    with output.open("a", encoding="utf-8") as out:
        for case in case_list:
            if case.case_id in completed:
                skipped += 1
                continue

            tag_cold_start = cold_start_pending
            start = time.perf_counter()
            try:
                result = run_executor(case)
                line = _record_to_jsonl(
                    result.record,
                    result.timing,
                    cold_start=tag_cold_start or None,
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                failure = EvalRecord(
                    case_id=case.case_id,
                    schema_failure=True,
                    ha_executed=False,
                )
                line = _record_to_jsonl(
                    failure,
                    CaseTiming(total_ms=elapsed_ms),
                    executor_error=str(exc),
                    cold_start=tag_cold_start or None,
                )
                errors += 1

            if tag_cold_start:
                cold_start_pending = False
            out.write(json.dumps(line, sort_keys=True) + "\n")
            out.flush()
            completed.add(case.case_id)
            scored += 1

    return BenchmarkRunResult(
        scored=scored,
        skipped=skipped,
        warmup_runs=warmup_runs,
        errors=errors,
    )
