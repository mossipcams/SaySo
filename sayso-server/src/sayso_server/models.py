"""Shared ControlPlan building blocks."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, model_validator

ENTITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z0-9_]+$")


def is_entity_id(value: str) -> bool:
    return bool(ENTITY_ID_PATTERN.match(value))


def validate_semantic_name(value: str) -> str:
    if is_entity_id(value):
        msg = "semantic targets must be names or aliases, not Home Assistant entity IDs"
        raise ValueError(msg)
    return value


SemanticName = Annotated[str, AfterValidator(validate_semantic_name)]


class ScopeKind(StrEnum):
    CURRENT_AREA = "current_area"
    NAMED_AREA = "named_area"
    FLOOR = "floor"
    ALL = "all"


class Scope(BaseModel):
    kind: ScopeKind
    name: str | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> Scope:
        if self.kind in {ScopeKind.NAMED_AREA, ScopeKind.FLOOR} and not self.name:
            msg = "named_area and floor scopes require a name"
            raise ValueError(msg)
        if self.name is not None:
            validate_semantic_name(self.name)
        return self


class ActionState(StrEnum):
    ON = "on"
    OFF = "off"
    TOGGLE = "toggle"
    OPEN = "open"
    CLOSE = "close"
    LOCK = "lock"
    UNLOCK = "unlock"
    ACTIVATE = "activate"


class ClimateMode(StrEnum):
    HEAT = "heat"
    COOL = "cool"
    AUTO = "auto"
    OFF = "off"


STATES_INCOMPATIBLE_WITH_VALUE = frozenset(
    {
        ActionState.OFF,
        ActionState.TOGGLE,
        ActionState.CLOSE,
        ActionState.UNLOCK,
        ActionState.ACTIVATE,
    }
)


def validate_state_value_mode(
    *,
    state: ActionState | None,
    value: float | int | None,
    mode: ClimateMode | None,
) -> None:
    if state is not None and mode is not None:
        msg = "state and mode cannot both be set"
        raise ValueError(msg)
    if state is not None and state in STATES_INCOMPATIBLE_WITH_VALUE and value is not None:
        msg = "state and value are incompatible"
        raise ValueError(msg)


class IntentField(BaseModel):
    intent: str = Field(min_length=1)
