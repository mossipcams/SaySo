"""JSONL evaluation case schema for SaySo benchmark corpora."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from sayso_server.control_plan import ControlPlan


class ExpectedOutcome(StrEnum):
    VALID_ACTION = "valid_action"
    VALID_QUERY = "valid_query"
    CLARIFICATION = "clarification"
    UNSUPPORTED = "unsupported"
    NO_ACTION = "no_action"


_PLAN_OUTCOME_BY_EXPECTED: dict[ExpectedOutcome, str] = {
    ExpectedOutcome.VALID_ACTION: "action",
    ExpectedOutcome.VALID_QUERY: "query",
    ExpectedOutcome.CLARIFICATION: "clarification",
    ExpectedOutcome.UNSUPPORTED: "unsupported",
    ExpectedOutcome.NO_ACTION: "no-action",
}


class EvalSchemaError(ValueError):
    """Evaluation case JSONL failed validation."""


class EvalCase(BaseModel):
    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    home: str = Field(min_length=1)
    origin: str = Field(min_length=1)
    turns: list[str] = Field(min_length=1)
    expected_control_plan: dict[str, Any]
    expected_candidate_entities: list[str]
    expected_resolved_entities: list[str]
    expected_outcome: ExpectedOutcome
    execution_allowed: bool = False

    @field_validator("turns")
    @classmethod
    def validate_turns_non_empty(cls, turns: list[str]) -> list[str]:
        if any(not turn.strip() for turn in turns):
            msg = "turns must contain non-empty utterances"
            raise ValueError(msg)
        return turns

    @model_validator(mode="after")
    def validate_expected_outcome_complete(self) -> EvalCase:
        if not self.expected_control_plan:
            msg = "expected_control_plan must be populated for expected_outcome"
            raise ValueError(msg)

        try:
            plan = ControlPlan.model_validate(self.expected_control_plan)
        except ValidationError as exc:
            msg = f"expected_control_plan is invalid: {exc.errors()[0]['msg']}"
            raise ValueError(msg) from exc

        expected_plan_outcome = _PLAN_OUTCOME_BY_EXPECTED[self.expected_outcome]
        if plan.outcome != expected_plan_outcome:
            msg = (
                "expected_outcome "
                f"'{self.expected_outcome}' requires control plan outcome "
                f"'{expected_plan_outcome}', got '{plan.outcome}'"
            )
            raise ValueError(msg)

        if self.expected_outcome == ExpectedOutcome.VALID_ACTION:
            if not self.expected_resolved_entities:
                msg = "valid_action requires non-empty expected_resolved_entities"
                raise ValueError(msg)
            missing = set(self.expected_resolved_entities) - set(self.expected_candidate_entities)
            if missing:
                msg = (
                    "expected_candidate_entities must include all "
                    f"expected_resolved_entities; missing {sorted(missing)}"
                )
                raise ValueError(msg)

        return self


def parse_eval_case(raw: str | bytes | bytearray, *, line_number: int = 1) -> EvalCase:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"line {line_number}: invalid JSON: {exc.msg}"
        raise EvalSchemaError(msg) from exc

    try:
        return EvalCase.model_validate(data)
    except ValidationError as exc:
        msg = f"line {line_number}: {exc.errors()[0]['msg']}"
        raise EvalSchemaError(msg) from exc


def load_eval_cases_jsonl(text: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        cases.append(parse_eval_case(stripped, line_number=line_number))
    return cases
