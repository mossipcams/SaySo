"""Per-satellite conversation state with TTL-bound referent resolution."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, Field


class ReferentKind(StrEnum):
    LAST_TARGET = "last_target"
    LAST_INTENT = "last_intent"


class LastTarget(BaseModel):
    entity_ids: list[str] = Field(min_length=1)


class LastIntent(BaseModel):
    intent: str = Field(min_length=1)
    outcome: str = Field(min_length=1)


class ConversationReferent(BaseModel):
    satellite_id: str = Field(min_length=1)
    kind: ReferentKind
    recorded_at: float
    generation: int = Field(ge=1)


class SatelliteConversationState(BaseModel):
    last_target: LastTarget | None = None
    last_target_at: float | None = None
    last_target_generation: int = 0
    last_intent: LastIntent | None = None
    last_intent_at: float | None = None
    last_intent_generation: int = 0


class ConversationStore:
    """In-memory per-satellite conversation state with configurable TTL."""

    def __init__(
        self,
        ttl_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            msg = "ttl_seconds must be positive"
            raise ValueError(msg)
        self._ttl_seconds = ttl_seconds
        self._clock = clock or time.monotonic
        self._states: dict[str, SatelliteConversationState] = {}

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def get_state(self, satellite_id: str) -> SatelliteConversationState:
        return self._states.setdefault(satellite_id, SatelliteConversationState())

    def record_last_target(self, satellite_id: str, target: LastTarget) -> ConversationReferent:
        recorded_at = self._clock()
        state = self.get_state(satellite_id)
        state.last_target_generation += 1
        state.last_target = target
        state.last_target_at = recorded_at
        return ConversationReferent(
            satellite_id=satellite_id,
            kind=ReferentKind.LAST_TARGET,
            recorded_at=recorded_at,
            generation=state.last_target_generation,
        )

    def record_last_intent(self, satellite_id: str, intent: LastIntent) -> ConversationReferent:
        recorded_at = self._clock()
        state = self.get_state(satellite_id)
        state.last_intent_generation += 1
        state.last_intent = intent
        state.last_intent_at = recorded_at
        return ConversationReferent(
            satellite_id=satellite_id,
            kind=ReferentKind.LAST_INTENT,
            recorded_at=recorded_at,
            generation=state.last_intent_generation,
        )

    def resolve_last_target(
        self,
        referent: ConversationReferent,
        *,
        satellite_id: str,
    ) -> LastTarget | None:
        return self._resolve(referent, satellite_id=satellite_id, kind=ReferentKind.LAST_TARGET)

    def resolve_last_intent(
        self,
        referent: ConversationReferent,
        *,
        satellite_id: str,
    ) -> LastIntent | None:
        return self._resolve(referent, satellite_id=satellite_id, kind=ReferentKind.LAST_INTENT)

    def active_last_target(self, satellite_id: str) -> LastTarget | None:
        """Return the current last target when its TTL has not expired."""
        return self._active_referent(
            satellite_id,
            value_attr="last_target",
            recorded_at_attr="last_target_at",
        )

    def active_last_intent(self, satellite_id: str) -> LastIntent | None:
        """Return the current last intent when its TTL has not expired."""
        return self._active_referent(
            satellite_id,
            value_attr="last_intent",
            recorded_at_attr="last_intent_at",
        )

    def _active_referent(
        self,
        satellite_id: str,
        *,
        value_attr: str,
        recorded_at_attr: str,
    ) -> LastTarget | LastIntent | None:
        state = self._states.get(satellite_id)
        if state is None:
            return None
        value = getattr(state, value_attr)
        recorded_at = getattr(state, recorded_at_attr)
        if value is None or recorded_at is None:
            return None
        if self._clock() - recorded_at > self._ttl_seconds:
            return None
        return value

    def _resolve(
        self,
        referent: ConversationReferent,
        *,
        satellite_id: str,
        kind: ReferentKind,
    ) -> LastTarget | LastIntent | None:
        if referent.satellite_id != satellite_id:
            return None
        if referent.kind != kind:
            return None
        if self._clock() - referent.recorded_at > self._ttl_seconds:
            return None

        state = self._states.get(satellite_id)
        if state is None:
            return None

        if kind == ReferentKind.LAST_TARGET:
            if (
                state.last_target is None
                or state.last_target_generation != referent.generation
            ):
                return None
            return state.last_target

        if state.last_intent is None or state.last_intent_generation != referent.generation:
            return None
        return state.last_intent
