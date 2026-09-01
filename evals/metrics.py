"""Metric scoring for recorded SaySo evaluation results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from evals.schema import EvalCase, ExpectedOutcome
from sayso_server.control_plan import ControlPlan

_ACTIONABLE_OUTCOMES = frozenset({ExpectedOutcome.VALID_ACTION, ExpectedOutcome.VALID_QUERY})
_FALSE_EXECUTION_OUTCOMES = frozenset(
    {
        ExpectedOutcome.CLARIFICATION,
        ExpectedOutcome.UNSUPPORTED,
        ExpectedOutcome.NO_ACTION,
    },
)
_UNORDERED_PLAN_LIST_FIELDS = frozenset({"targets", "include", "exclude"})
_FOLLOW_UP_CATEGORIES = frozenset({"active_followup", "followup", "follow_up"})
FailureStage = Literal[
    "stt",
    "retrieve",
    "plan",
    "parse",
    "resolve",
    "safety",
    "request",
    "verify",
    "schema",
]


class EvalRecord(BaseModel):
    case_id: str = Field(min_length=1)
    recorded_control_plan: dict[str, Any] | None = None
    schema_failure: bool = False
    recorded_candidate_entities: list[str] = Field(default_factory=list)
    recorded_resolved_entities: list[str] = Field(default_factory=list)
    executed_entities: list[str] = Field(default_factory=list)
    ha_executed: bool = False
    recorded_query_answer: str | None = None
    recorded_follow_up_plan: dict[str, Any] | None = None
    recorded_follow_up_resolved_entities: list[str] | None = None
    failure_stage: FailureStage | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class MetricScore:
    total_cases: int
    control_plan_exact_match: int
    control_plan_semantic_wrong: int
    control_plan_schema_failure: int
    control_plan_accuracy: float
    candidate_retrieval_recall: float
    candidate_retrieval_numerator: int
    candidate_retrieval_denominator: int
    mean_candidate_set_size: float
    exact_target_resolution: float
    exact_target_numerator: int
    exact_target_denominator: int
    wrong_device_rate: float
    wrong_device_numerator: int
    wrong_device_denominator: int
    unintended_entity_count: int
    false_execution_rate: float
    false_execution_numerator: int
    false_execution_denominator: int
    clarification_precision: float
    clarification_recall: float
    clarification_true_positive: int
    clarification_false_positive: int
    clarification_false_negative: int
    query_accuracy: float
    query_numerator: int
    query_denominator: int
    follow_up_accuracy: float
    follow_up_numerator: int
    follow_up_denominator: int
    missing_records: list[str] = field(default_factory=list)


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def _entity_set(values: list[str]) -> frozenset[str]:
    return frozenset(values)


def canonicalize_control_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize optional defaults and unordered semantic-name lists."""
    validated = ControlPlan.model_validate(plan)
    canonical = validated.model_dump(mode="json", exclude_none=True)
    for key in _UNORDERED_PLAN_LIST_FIELDS:
        if key in canonical:
            canonical[key] = sorted(canonical[key])
        else:
            canonical[key] = []
    return canonical


def try_canonicalize_control_plan(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    try:
        return canonicalize_control_plan(plan)
    except ValidationError:
        return None


def normalize_query_answer(answer: str) -> str:
    collapsed = re.sub(r"\s+", " ", answer.strip().lower())
    return collapsed.rstrip(".!?")


def control_plans_match(expected: dict[str, Any], recorded: dict[str, Any] | None) -> bool:
    if recorded is None:
        return False
    expected_canonical = try_canonicalize_control_plan(expected)
    recorded_canonical = try_canonicalize_control_plan(recorded)
    if expected_canonical is None or recorded_canonical is None:
        return False
    return expected_canonical == recorded_canonical


def _is_actionable(case: EvalCase) -> bool:
    return case.expected_outcome in _ACTIONABLE_OUTCOMES


def _is_follow_up_case(case: EvalCase) -> bool:
    return len(case.turns) > 1 or case.category in _FOLLOW_UP_CATEGORIES


def _recorded_requested_clarification(record: EvalRecord) -> bool:
    if record.schema_failure or record.recorded_control_plan is None:
        return False
    return record.recorded_control_plan.get("outcome") == "clarification"


def _expected_requested_clarification(case: EvalCase) -> bool:
    return case.expected_outcome == ExpectedOutcome.CLARIFICATION


def score_records(
    cases: list[EvalCase],
    records: list[EvalRecord],
    *,
    expected_query_answers: dict[str, str] | None = None,
) -> MetricScore:
    """Score recorded eval results against authored EvalCase expectations."""
    records_by_id = {record.case_id: record for record in records}
    answers_by_id = expected_query_answers or {}

    control_plan_exact_match = 0
    control_plan_semantic_wrong = 0
    control_plan_schema_failure = 0

    candidate_hits = 0
    candidate_total = 0
    candidate_sizes: list[int] = []

    target_hits = 0
    target_total = 0

    wrong_device_cases = 0
    executed_action_cases = 0
    unintended_entity_count = 0

    false_execution_cases = 0
    false_execution_total = 0

    clarification_tp = 0
    clarification_fp = 0
    clarification_fn = 0

    query_hits = 0
    query_total = 0

    follow_up_hits = 0
    follow_up_total = 0

    missing_records: list[str] = []

    for case in cases:
        record = records_by_id.get(case.case_id)
        if record is None:
            missing_records.append(case.case_id)
            continue

        if record.schema_failure:
            control_plan_schema_failure += 1
        elif control_plans_match(case.expected_control_plan, record.recorded_control_plan):
            control_plan_exact_match += 1
        else:
            control_plan_semantic_wrong += 1

        if _is_actionable(case):
            candidate_total += 1
            candidate_sizes.append(len(record.recorded_candidate_entities))
            required = _entity_set(case.expected_resolved_entities)
            retrieved = _entity_set(record.recorded_candidate_entities)
            if required.issubset(retrieved):
                candidate_hits += 1

            target_total += 1
            if _entity_set(record.recorded_resolved_entities) == _entity_set(case.expected_resolved_entities):
                target_hits += 1

        if record.ha_executed:
            executed_action_cases += 1
            expected_targets = _entity_set(case.expected_resolved_entities)
            unintended = _entity_set(record.executed_entities) - expected_targets
            if unintended:
                wrong_device_cases += 1
                unintended_entity_count += len(unintended)

        if case.expected_outcome in _FALSE_EXECUTION_OUTCOMES:
            false_execution_total += 1
            if record.ha_executed:
                false_execution_cases += 1

        expected_clarification = _expected_requested_clarification(case)
        recorded_clarification = _recorded_requested_clarification(record)
        if expected_clarification and recorded_clarification:
            clarification_tp += 1
        elif expected_clarification and not recorded_clarification:
            clarification_fn += 1
        elif not expected_clarification and recorded_clarification:
            clarification_fp += 1

        if case.expected_outcome == ExpectedOutcome.VALID_QUERY:
            query_total += 1
            expected_answer = answers_by_id.get(case.case_id)
            recorded_answer = record.recorded_query_answer
            if (
                expected_answer is not None
                and recorded_answer is not None
                and normalize_query_answer(recorded_answer) == normalize_query_answer(expected_answer)
            ):
                query_hits += 1

        if _is_follow_up_case(case):
            follow_up_total += 1
            plan_ok = control_plans_match(
                case.expected_control_plan,
                record.recorded_follow_up_plan or record.recorded_control_plan,
            )
            expected_targets = record.recorded_follow_up_resolved_entities
            if expected_targets is None:
                targets_ok = _entity_set(record.recorded_resolved_entities) == _entity_set(
                    case.expected_resolved_entities,
                )
            else:
                targets_ok = _entity_set(expected_targets) == _entity_set(case.expected_resolved_entities)
            if plan_ok and targets_ok:
                follow_up_hits += 1

    total_cases = len(cases)
    control_plan_accuracy = control_plan_exact_match / total_cases if total_cases else 0.0
    candidate_retrieval_recall = candidate_hits / candidate_total if candidate_total else 0.0
    mean_candidate_set_size = sum(candidate_sizes) / len(candidate_sizes) if candidate_sizes else 0.0
    exact_target_resolution = target_hits / target_total if target_total else 0.0
    wrong_device_rate = wrong_device_cases / executed_action_cases if executed_action_cases else 0.0
    false_execution_rate = false_execution_cases / false_execution_total if false_execution_total else 0.0

    clarification_predicted = clarification_tp + clarification_fp
    clarification_expected = clarification_tp + clarification_fn
    clarification_precision = clarification_tp / clarification_predicted if clarification_predicted else 0.0
    clarification_recall = clarification_tp / clarification_expected if clarification_expected else 0.0
    query_accuracy = query_hits / query_total if query_total else 0.0
    follow_up_accuracy = follow_up_hits / follow_up_total if follow_up_total else 0.0

    return MetricScore(
        total_cases=total_cases,
        control_plan_exact_match=control_plan_exact_match,
        control_plan_semantic_wrong=control_plan_semantic_wrong,
        control_plan_schema_failure=control_plan_schema_failure,
        control_plan_accuracy=control_plan_accuracy,
        candidate_retrieval_recall=candidate_retrieval_recall,
        candidate_retrieval_numerator=candidate_hits,
        candidate_retrieval_denominator=candidate_total,
        mean_candidate_set_size=mean_candidate_set_size,
        exact_target_resolution=exact_target_resolution,
        exact_target_numerator=target_hits,
        exact_target_denominator=target_total,
        wrong_device_rate=wrong_device_rate,
        wrong_device_numerator=wrong_device_cases,
        wrong_device_denominator=executed_action_cases,
        unintended_entity_count=unintended_entity_count,
        false_execution_rate=false_execution_rate,
        false_execution_numerator=false_execution_cases,
        false_execution_denominator=false_execution_total,
        clarification_precision=clarification_precision,
        clarification_recall=clarification_recall,
        clarification_true_positive=clarification_tp,
        clarification_false_positive=clarification_fp,
        clarification_false_negative=clarification_fn,
        query_accuracy=query_accuracy,
        query_numerator=query_hits,
        query_denominator=query_total,
        follow_up_accuracy=follow_up_accuracy,
        follow_up_numerator=follow_up_hits,
        follow_up_denominator=follow_up_total,
        missing_records=missing_records,
    )
