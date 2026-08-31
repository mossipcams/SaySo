"""Typed ControlPlan outcomes emitted by the language model."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from sayso_server.models import (
    ActionState,
    ClimateMode,
    IntentField,
    Scope,
    SemanticName,
    validate_state_value_mode,
)


class ActionPlan(IntentField):
    outcome: Literal["action"] = "action"
    domain: str
    scope: Scope | None = None
    targets: list[SemanticName] = Field(default_factory=list)
    include: list[SemanticName] = Field(default_factory=list)
    exclude: list[SemanticName] = Field(default_factory=list)
    state: ActionState | None = None
    value: float | int | None = None
    mode: ClimateMode | None = None

    @model_validator(mode="after")
    def validate_action_fields(self) -> ActionPlan:
        validate_state_value_mode(state=self.state, value=self.value, mode=self.mode)
        return self


class QueryPlan(IntentField):
    outcome: Literal["query"] = "query"
    domain: str
    scope: Scope | None = None
    targets: list[SemanticName] = Field(default_factory=list)
    include: list[SemanticName] = Field(default_factory=list)
    exclude: list[SemanticName] = Field(default_factory=list)
    attribute: str | None = None


class ClarificationPlan(IntentField):
    outcome: Literal["clarification"] = "clarification"
    reason: str = Field(min_length=1)


class UnsupportedPlan(IntentField):
    outcome: Literal["unsupported"] = "unsupported"
    reason: str = Field(min_length=1)


class NoActionPlan(IntentField):
    outcome: Literal["no-action"] = "no-action"
    reason: str = Field(min_length=1)


PlanUnion = Annotated[
    ActionPlan | QueryPlan | ClarificationPlan | UnsupportedPlan | NoActionPlan,
    Field(discriminator="outcome"),
]

_adapter = TypeAdapter(PlanUnion)


class ControlPlan:
    """Validate and parse model-emitted control plans."""

    Action = ActionPlan
    Query = QueryPlan
    Clarification = ClarificationPlan
    Unsupported = UnsupportedPlan
    NoAction = NoActionPlan

    @classmethod
    def model_validate(cls, obj: object, /, **kwargs: object) -> BaseModel:
        return _adapter.validate_python(obj, **kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, /, **kwargs: object) -> BaseModel:
        return _adapter.validate_json(json_data, **kwargs)

    @classmethod
    def json_schema(cls) -> dict[str, object]:
        schema = _adapter.json_schema()
        schema["title"] = "ControlPlan"
        return schema
