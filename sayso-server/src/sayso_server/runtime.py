"""Narrow model runtime contract for ControlPlan generation."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import BaseModel, Field

from sayso_server.candidates import retrieve_candidates
from sayso_server.control_plan import (
    ActionPlan,
    ClarificationPlan,
    NoActionPlan,
    QueryPlan,
    UnsupportedPlan,
)
from sayso_server.conversation import SatelliteConversationState
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.parser import parse_model_output
from sayso_server.prompt import PromptOrigin, build_lfm_prompt
from sayso_server.scoring import lookup_origin

_logger = logging.getLogger(__name__)

ValidatedPlan = ActionPlan | QueryPlan | ClarificationPlan | UnsupportedPlan | NoActionPlan


def parse_lfm_prompt_payload(prompt: str) -> dict[str, object]:
    """Parse the JSON payload from a built LFM prompt."""
    json_start = prompt.index("{")
    loaded = json.loads(prompt[json_start:])
    if not isinstance(loaded, dict):
        msg = "LFM prompt payload must be a JSON object"
        raise TypeError(msg)
    return loaded


class ModelMetadata(BaseModel):
    model_id: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    revision: str | None = None
    warm: bool = False
    resident: bool = False


class RawGenerationResult(BaseModel):
    text: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    metadata: ModelMetadata


class PlanGenerationResult(BaseModel):
    plan: ValidatedPlan
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    metadata: ModelMetadata


class ModelRuntime(ABC):
    """Load-once runtime that emits raw model text from an LFM prompt."""

    @abstractmethod
    def load(self) -> None:
        """Prepare the runtime for generation."""

    @abstractmethod
    def generate(self, prompt: str) -> RawGenerationResult:
        """Generate raw model text for a built LFM prompt."""


def compose_plan_generation(
    *,
    runtime: ModelRuntime,
    snapshot: HomeGraphSnapshot,
    satellite_id: str,
    area_id: str,
    text: str,
    conversation: SatelliteConversationState | None = None,
) -> PlanGenerationResult:
    """Retrieve candidates, build the LFM prompt, generate, and parse model output."""
    conv = conversation or SatelliteConversationState()
    scored = retrieve_candidates(
        snapshot,
        origin_area_id=area_id,
        request=text,
        conversation=conv,
        limit=1,
    )
    area, _ = lookup_origin(snapshot, area_id)
    if area is None:
        msg = "origin area is required"
        raise RuntimeError(msg)

    prompt = build_lfm_prompt(
        user_text=text,
        origin=PromptOrigin(
            satellite_id=satellite_id,
            area_name=area.name,
            area_aliases=list(area.aliases),
        ),
        conversation=conv,
        candidates=[candidate.item for candidate in scored],
        areas=snapshot.areas,
    )
    raw = runtime.generate(prompt)
    _logger.warning("raw model sample: %s", raw.text)
    plan = parse_model_output(raw.text, intent=text)
    return PlanGenerationResult(
        plan=plan,
        prompt_tokens=raw.prompt_tokens,
        completion_tokens=raw.completion_tokens,
        latency_ms=raw.latency_ms,
        metadata=raw.metadata,
    )


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

    def generate(self, prompt: str) -> RawGenerationResult:
        if not self._loaded:
            msg = "model runtime must be loaded before generate"
            raise RuntimeError(msg)

        started = self._clock()
        payload = parse_lfm_prompt_payload(prompt)
        user_text = payload["user_text"]
        raw_text = json.dumps(
            {
                "outcome": "query",
                "intent": user_text,
                "domain": "light",
            }
        )
        elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
        prompt_tokens = len(prompt.split())

        return RawGenerationResult(
            text=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=1,
            latency_ms=elapsed_ms,
            metadata=ModelMetadata(
                model_id=self._model_id,
                runtime="fake",
                revision=self._revision,
            ),
        )
