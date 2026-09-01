"""JSONL interaction telemetry with monotonic stage timings."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal, Protocol, TextIO

from pydantic import BaseModel, Field

from sayso_server.runtime import PlanGenerationResult

STAGE_NAMES: tuple[str, ...] = ("plan", "resolve", "validate", "request", "verify")


class StageTimings(BaseModel):
    plan_ms: float = Field(ge=0)
    resolve_ms: float = Field(ge=0)
    validate_ms: float = Field(ge=0)
    request_ms: float = Field(ge=0)
    verify_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class ModelTelemetry(BaseModel):
    model_id: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    revision: str | None = None
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)


class InteractionTelemetryRecord(BaseModel):
    schema_version: Literal[1] = 1
    correlation_id: str = Field(min_length=1)
    satellite_id: str = Field(min_length=1)
    area_id: str = Field(min_length=1)
    input_type: Literal["text"] = "text"
    category: str = Field(min_length=1)
    reason: str | None = None
    request_id: str | None = None
    monotonic_started_at: float = Field(ge=0)
    stages: StageTimings
    model: ModelTelemetry | None = None


def mandatory_field_names() -> frozenset[str]:
    """Return top-level fields required on every telemetry record."""

    return frozenset(
        {
            "schema_version",
            "correlation_id",
            "satellite_id",
            "area_id",
            "input_type",
            "category",
            "monotonic_started_at",
            "stages",
        },
    )


class TelemetrySink(Protocol):
    def write(self, record: InteractionTelemetryRecord) -> None: ...


class JsonlTelemetrySink:
    """Append one JSON object per line to a text stream."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, record: InteractionTelemetryRecord) -> None:
        line = json.dumps(record.model_dump(mode="json"), separators=(",", ":"))
        self._stream.write(f"{line}\n")


class InteractionTelemetry:
    """Collect monotonic stage timings for one interaction."""

    def __init__(
        self,
        *,
        correlation_id: str,
        satellite_id: str,
        area_id: str,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._clock = clock or time.monotonic
        self.correlation_id = correlation_id
        self.satellite_id = satellite_id
        self.area_id = area_id
        self.monotonic_started_at = self._clock()
        self._stage_ms: dict[str, float] = dict.fromkeys(STAGE_NAMES, 0.0)
        self._model: ModelTelemetry | None = None

    @contextmanager
    def time_stage(self, stage: str) -> Iterator[None]:
        if stage not in STAGE_NAMES:
            msg = f"unknown telemetry stage: {stage}"
            raise ValueError(msg)
        started = self._clock()
        try:
            yield
        finally:
            elapsed_ms = max(0.0, (self._clock() - started) * 1000.0)
            self._stage_ms[stage] += elapsed_ms

    def set_model_from_generation(self, generation: PlanGenerationResult) -> None:
        self._model = ModelTelemetry(
            model_id=generation.metadata.model_id,
            runtime=generation.metadata.runtime,
            revision=generation.metadata.revision,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            latency_ms=generation.latency_ms,
        )

    def finish(
        self,
        *,
        category: str,
        reason: str | None = None,
        request_id: str | None = None,
    ) -> InteractionTelemetryRecord:
        total_ms = sum(self._stage_ms.values())
        record = InteractionTelemetryRecord(
            correlation_id=self.correlation_id,
            satellite_id=self.satellite_id,
            area_id=self.area_id,
            category=category,
            reason=reason,
            request_id=request_id,
            monotonic_started_at=self.monotonic_started_at,
            stages=StageTimings(
                plan_ms=self._stage_ms["plan"],
                resolve_ms=self._stage_ms["resolve"],
                validate_ms=self._stage_ms["validate"],
                request_ms=self._stage_ms["request"],
                verify_ms=self._stage_ms["verify"],
                total_ms=total_ms,
            ),
            model=self._model,
        )
        missing = mandatory_field_names() - record.model_dump(mode="json").keys()
        if missing:
            msg = f"telemetry record missing mandatory fields: {sorted(missing)}"
            raise RuntimeError(msg)
        return record


__all__ = [
    "InteractionTelemetry",
    "InteractionTelemetryRecord",
    "JsonlTelemetrySink",
    "ModelTelemetry",
    "STAGE_NAMES",
    "StageTimings",
    "TelemetrySink",
    "mandatory_field_names",
]
