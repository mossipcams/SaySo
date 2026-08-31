"""Narrow model runtime contract for ControlPlan generation."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import BaseModel, Field

from sayso_server.control_plan import (
    ActionPlan,
    ClarificationPlan,
    ControlPlan,
    NoActionPlan,
    QueryPlan,
    UnsupportedPlan,
)

ValidatedPlan = ActionPlan | QueryPlan | ClarificationPlan | UnsupportedPlan | NoActionPlan


class ModelMetadata(BaseModel):
    model_id: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    revision: str | None = None
    warm: bool = False
    resident: bool = False


class PlanGenerationResult(BaseModel):
    plan: ValidatedPlan
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    metadata: ModelMetadata


class ModelRuntime(ABC):
    """Load-once runtime that emits validated ControlPlans."""

    @abstractmethod
    def load(self) -> None:
        """Prepare the runtime for generation."""

    @abstractmethod
    def generate_plan(self, text: str) -> PlanGenerationResult:
        """Generate a validated ControlPlan for user text."""


class FakeModelRuntime(ModelRuntime):
    """In-process runtime for tests and offline development."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        model_id: str = "fake",
        revision: str | None = "test",
    ) -> None:
        self._clock = clock or time.monotonic
        self._model_id = model_id
        self._revision = revision
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    def generate_plan(self, text: str) -> PlanGenerationResult:
        if not self._loaded:
            msg = "model runtime must be loaded before generate_plan"
            raise RuntimeError(msg)

        started = self._clock()
        plan = ControlPlan.model_validate(
            {
                "outcome": "query",
                "intent": text,
                "domain": "light",
            }
        )
        elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
        prompt_tokens = len(text.split())

        return PlanGenerationResult(
            plan=plan,
            prompt_tokens=prompt_tokens,
            completion_tokens=1,
            latency_ms=elapsed_ms,
            metadata=ModelMetadata(
                model_id=self._model_id,
                runtime="fake",
                revision=self._revision,
            ),
        )
