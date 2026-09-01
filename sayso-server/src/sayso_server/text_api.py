"""Validated POST /api/v1/text endpoint and response envelope."""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal, Protocol

from aiohttp import web
from pydantic import BaseModel, Field, ValidationError

from sayso_server.api import API_VERSION
from sayso_server.auth import bearer_token_valid
from sayso_server.const import TEXT_PATH
from sayso_server.conversation import ConversationStore
from sayso_server.graph_store import HomeGraphStore
from sayso_server.ha_client import ActionRequestClient
from sayso_server.orchestrator import execute_control_plan
from sayso_server.runtime import FakeModelRuntime, ModelRuntime
from sayso_server.satellites import SatelliteRegistry
from sayso_server.telemetry import InteractionTelemetry, TelemetrySink


class TextRequestPayload(BaseModel):
    satellite_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class TextRequestEnvelope(BaseModel):
    version: Literal[API_VERSION]
    type: Literal["text"]
    correlation_id: str = Field(min_length=1)
    payload: TextRequestPayload


class TextResponsePayload(BaseModel):
    category: str
    reason: str | None = None
    plan: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = None


class TextResponseEnvelope(BaseModel):
    version: Literal[API_VERSION]
    type: Literal["text_response"]
    correlation_id: str = Field(min_length=1)
    payload: TextResponsePayload


class ErrorResponseEnvelope(BaseModel):
    version: Literal[API_VERSION]
    type: Literal["error"]
    correlation_id: str = Field(min_length=1)
    payload: dict[str, str]


class TextController(Protocol):
    def handle(
        self,
        *,
        satellite_id: str,
        area_id: str,
        text: str,
        correlation_id: str,
    ) -> dict[str, Any]: ...


class OrchestratorTextController:
    """Run model generation and the deterministic execution orchestrator."""

    def __init__(
        self,
        *,
        runtime: ModelRuntime,
        ha_client: ActionRequestClient,
        graph_store: HomeGraphStore,
        conversation_store: ConversationStore | None = None,
        telemetry_sink: TelemetrySink | None = None,
    ) -> None:
        self._runtime = runtime
        self._ha_client = ha_client
        self._graph_store = graph_store
        self._conversation_store = conversation_store
        self._telemetry_sink = telemetry_sink

    def handle(
        self,
        *,
        satellite_id: str,
        area_id: str,
        text: str,
        correlation_id: str,
    ) -> dict[str, Any]:
        snapshot = self._graph_store.snapshot
        if snapshot is None:
            msg = "home graph snapshot is required"
            raise RuntimeError(msg)

        request_id = correlation_id or str(uuid.uuid4())
        telemetry = InteractionTelemetry(
            correlation_id=correlation_id,
            satellite_id=satellite_id,
            area_id=area_id,
        )
        with telemetry.time_stage("plan"):
            generation = self._runtime.generate_plan(text)
        telemetry.set_model_from_generation(generation)
        outcome = execute_control_plan(
            generation.plan,
            snapshot,
            origin_area_id=area_id,
            ha_client=self._ha_client,
            request_id=request_id,
            conversation_store=self._conversation_store,
            satellite_id=satellite_id,
            telemetry=telemetry,
        )
        if self._telemetry_sink is not None:
            self._telemetry_sink.write(
                telemetry.finish(
                    category=outcome.category.value,
                    reason=outcome.reason,
                    request_id=outcome.request_id,
                ),
            )
        plan_payload = (
            outcome.plan.model_dump(mode="json")
            if hasattr(outcome.plan, "model_dump")
            else dict(outcome.plan) if isinstance(outcome.plan, dict) else {}
        )
        return {
            "category": outcome.category.value,
            "reason": outcome.reason,
            "plan": plan_payload,
            "request_id": outcome.request_id,
        }


def create_text_handler(
    *,
    token: str,
    satellite_registry: SatelliteRegistry,
    graph_store: HomeGraphStore,
    text_controller: TextController | None = None,
) -> web.RequestHandler:
    """Create the aiohttp handler for POST /api/v1/text."""

    async def text(request: web.Request) -> web.Response:
        if not bearer_token_valid(
            authorization=request.headers.get("Authorization"),
            expected_token=token,
        ):
            return web.Response(status=401)

        raw = await request.text()
        try:
            data = json.loads(raw)
            envelope = TextRequestEnvelope.model_validate(data)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
            correlation_id = _extract_correlation_id(raw)
            error = ErrorResponseEnvelope(
                version=API_VERSION,
                type="error",
                correlation_id=correlation_id,
                payload={"code": "invalid_request", "message": "invalid text request envelope"},
            )
            return web.json_response(error.model_dump(mode="json"), status=400)

        area_id, error_code = satellite_registry.resolve_area_id(
            envelope.payload.satellite_id,
            snapshot=graph_store.snapshot,
        )
        if error_code is not None or area_id is None:
            error = ErrorResponseEnvelope(
                version=API_VERSION,
                type="error",
                correlation_id=envelope.correlation_id,
                payload={
                    "code": error_code or "invalid_satellite",
                    "message": "satellite context is invalid",
                },
            )
            return web.json_response(error.model_dump(mode="json"), status=400)

        if text_controller is None:
            error = ErrorResponseEnvelope(
                version=API_VERSION,
                type="error",
                correlation_id=envelope.correlation_id,
                payload={"code": "not_configured", "message": "text controller unavailable"},
            )
            return web.json_response(error.model_dump(mode="json"), status=503)

        payload = text_controller.handle(
            satellite_id=envelope.payload.satellite_id,
            area_id=area_id,
            text=envelope.payload.text,
            correlation_id=envelope.correlation_id,
        )
        response = TextResponseEnvelope(
            version=API_VERSION,
            type="text_response",
            correlation_id=envelope.correlation_id,
            payload=TextResponsePayload.model_validate(payload),
        )
        return web.json_response(response.model_dump(mode="json"))

    return text


def _extract_correlation_id(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "invalid"
    correlation_id = parsed.get("correlation_id")
    if isinstance(correlation_id, str) and correlation_id:
        return correlation_id
    return "invalid"


def default_text_dependencies() -> tuple[SatelliteRegistry, HomeGraphStore, TextController]:
    """Build default in-process text dependencies for the aiohttp app."""

    runtime = FakeModelRuntime()
    runtime.load()
    registry = SatelliteRegistry()
    graph_store = HomeGraphStore()
    from sayso_server.ha_client import FakeHaClient

    controller = OrchestratorTextController(
        runtime=runtime,
        ha_client=FakeHaClient(),
        graph_store=graph_store,
    )
    return registry, graph_store, controller


__all__ = [
    "ErrorResponseEnvelope",
    "OrchestratorTextController",
    "TextController",
    "TextRequestEnvelope",
    "TextResponseEnvelope",
    "create_text_handler",
    "default_text_dependencies",
]
