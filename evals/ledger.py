"""Failure ledger classification for SaySo evaluation records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from evals.metrics import EvalRecord, FailureStage, control_plans_match
from evals.schema import EvalCase, ExpectedOutcome

_STAGE_ORDER: tuple[FailureStage, ...] = (
    "stt",
    "retrieve",
    "plan",
    "parse",
    "resolve",
    "safety",
    "request",
    "verify",
    "schema",
)
_STAGE_RANK = {stage: index for index, stage in enumerate(_STAGE_ORDER)}

_FALSE_EXECUTION_OUTCOMES = frozenset(
    {
        ExpectedOutcome.CLARIFICATION,
        ExpectedOutcome.UNSUPPORTED,
        ExpectedOutcome.NO_ACTION,
    },
)


@dataclass(frozen=True)
class LedgerEntry:
    case_id: str
    stage: FailureStage
    reason: str


@dataclass(frozen=True)
class StageSummary:
    stage: FailureStage
    count: int
    case_ids: list[str]


@dataclass(frozen=True)
class StageReasonSummary:
    stage: FailureStage
    reason: str
    count: int
    case_ids: list[str]


@dataclass(frozen=True)
class LedgerSummary:
    by_stage: list[StageSummary]
    by_stage_reason: list[StageReasonSummary]


def _stage_sort_key(stage: FailureStage) -> tuple[int, str]:
    return (_STAGE_RANK.get(stage, len(_STAGE_ORDER)), stage)


def ledger_entries(
    cases: list[EvalCase],
    records: list[EvalRecord],
) -> list[LedgerEntry]:
    """Join cases to records by case_id and return classified failures only."""
    records_by_id = {record.case_id: record for record in records}
    entries: list[LedgerEntry] = []
    for case in cases:
        record = records_by_id.get(case.case_id)
        if record is None:
            continue
        failure = classify_failure(case, record)
        if failure is None:
            continue
        stage, reason = failure
        entries.append(LedgerEntry(case_id=case.case_id, stage=stage, reason=reason))
    return entries


def ledger_summary(entries: list[LedgerEntry]) -> LedgerSummary:
    """Aggregate ledger entries into deterministic stage and stage/reason counts."""
    stage_case_ids: dict[FailureStage, list[str]] = defaultdict(list)
    stage_reason_case_ids: dict[tuple[FailureStage, str], list[str]] = defaultdict(list)

    for entry in entries:
        stage_case_ids[entry.stage].append(entry.case_id)
        stage_reason_case_ids[(entry.stage, entry.reason)].append(entry.case_id)

    by_stage = [
        StageSummary(stage=stage, count=len(case_ids), case_ids=sorted(case_ids))
        for stage, case_ids in sorted(stage_case_ids.items(), key=lambda item: _stage_sort_key(item[0]))
    ]
    by_stage_reason = [
        StageReasonSummary(
            stage=stage,
            reason=reason,
            count=len(case_ids),
            case_ids=sorted(case_ids),
        )
        for (stage, reason), case_ids in sorted(
            stage_reason_case_ids.items(),
            key=lambda item: (_stage_sort_key(item[0][0]), item[0][1]),
        )
    ]
    return LedgerSummary(by_stage=by_stage, by_stage_reason=by_stage_reason)


def classify_failure(case: EvalCase, record: EvalRecord) -> tuple[FailureStage, str] | None:
    """Return the first matching failure stage and reason, or None on success."""
    if record.schema_failure:
        return ("schema", "schema_failure")

    plan = record.recorded_control_plan
    if (
        plan is not None
        and plan.get("outcome") == "no-action"
        and plan.get("reason") == "model_output_invalid"
    ):
        return ("parse", "model_output_invalid")

    expected_candidates = case.expected_candidate_entities
    if expected_candidates:
        recorded_candidates = set(record.recorded_candidate_entities)
        missing = sorted(entity for entity in expected_candidates if entity not in recorded_candidates)
        if missing:
            return ("retrieve", f"missing expected candidates: {', '.join(missing)}")

    if not control_plans_match(case.expected_control_plan, record.recorded_control_plan):
        if record.recorded_control_plan is None:
            return ("plan", "missing recorded control plan")
        return ("plan", "control plan mismatch")

    if set(record.recorded_resolved_entities) != set(case.expected_resolved_entities):
        return ("resolve", "resolved entity mismatch")

    if record.ha_executed and case.expected_outcome in _FALSE_EXECUTION_OUTCOMES:
        return ("safety", f"ha_executed on {case.expected_outcome}")

    if (
        case.expected_outcome == ExpectedOutcome.VALID_ACTION
        and record.ha_executed
        and set(record.executed_entities) != set(case.expected_resolved_entities)
    ):
        return ("verify", "executed entity mismatch")

    return None
