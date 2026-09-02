"""Validated POST /api/v1/text endpoint and response envelope."""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal, Protocol

from aiohttp import web
from pydantic import BaseModel, Field, ValidationError

from sayso_server.api import API_VERSION
from sayso_server.auth import bearer_token_valid
from sayso_server.candidates import retrieve_candidates
from sayso_server.const import TEXT_PATH
from sayso_server.control_plan import ActionPlan, ClarificationPlan, NoActionPlan, QueryPlan
from sayso_server.conversation import ConversationStore, SatelliteConversationState
from sayso_server.followups import resolve_follow_up
from sayso_server.graph_store import HomeGraphStore
from sayso_server.ha_client import ActionRequestClient
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import Scope, ScopeKind
from sayso_server.orchestrator import execute_control_plan, execute_control_plan_async
from sayso_server.parser import parse_model_output
from sayso_server.prompt import PromptOrigin, build_lfm_prompt
from sayso_server.readiness import text_execution_refusal
from sayso_server.resolver import resolve_action_entities
from sayso_server.response_policy import resolve_response_policy
from sayso_server.results import ExecutionCategory, ExecutionOutcome
from sayso_server.runtime import (
    FakeModelRuntime,
    ModelRuntime,
    PlanGenerationResult,
    compose_plan_generation,
)
from sayso_server.satellites import SatelliteRegistry
from sayso_server.session import HaGatewayBinding
from sayso_server.telemetry import InteractionTelemetry, TelemetrySink

MISSING_ORIGIN_AREA_SENTINEL = ""
MISSING_ORIGIN_PROMPT_AREA_NAME = "unknown"
MISSING_ORIGIN_CLARIFICATION_REASON = "which area?"


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
    response_mode: str | None = None
    response_content: str | None = None


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


class ConversationRequestPayload(BaseModel):
    transcript: str = Field(min_length=1)
    source_id: str | None = None
    area_id: str | None = None
    stt_ms: float = Field(default=0.0, ge=0.0)


def resolve_conversation_area(
    payload: ConversationRequestPayload,
    *,
    graph_store: HomeGraphStore,
) -> tuple[str | None, str | None]:
    """Return ``(area_id, error_code)`` for a HA-supplied conversation origin."""

    if payload.area_id is None:
        if graph_store.snapshot is None:
            return None, "no_graph"
        return None, None

    snapshot = graph_store.snapshot
    if snapshot is None:
        return None, "no_graph"

    for area in snapshot.areas:
        if area.id == payload.area_id:
            return area.id, None

    return None, "unknown_area"


def _effective_action_scope(plan: ActionPlan) -> Scope | None:
    scope = plan.scope
    if scope is None and (plan.targets or plan.include or plan.exclude):
        return Scope(kind=ScopeKind.CURRENT_AREA)
    return scope


def _effective_query_scope(plan: QueryPlan) -> Scope | None:
    scope = plan.scope
    if scope is None and (plan.targets or plan.include or plan.exclude):
        return Scope(kind=ScopeKind.CURRENT_AREA)
    return scope


def _explicit_target_resolves_without_origin(
    plan: ActionPlan,
    snapshot: HomeGraphSnapshot,
    *,
    scope: Scope,
) -> bool:
    if not (plan.targets or plan.include):
        return False
    resolution = resolve_action_entities(
        snapshot,
        origin_area_id=MISSING_ORIGIN_AREA_SENTINEL,
        intent=plan.intent,
        scope=scope,
        domain=plan.domain,
        targets=plan.targets,
        include=plan.include,
        exclude=plan.exclude,
    )
    return len(resolution.entity_ids) == 1


def missing_origin_clarification(
    plan: BaseModel,
    snapshot: HomeGraphSnapshot,
    *,
    satellite_id: str | None,
    conversation_store: ConversationStore | None,
) -> ClarificationPlan | None:
    """Return area clarification when origin is required but absent."""

    if isinstance(plan, ActionPlan):
        if conversation_store is not None and satellite_id is not None:
            follow_up = resolve_follow_up(plan, conversation_store, satellite_id=satellite_id)
            if follow_up.outcome == "resolved":
                return None
            if follow_up.outcome == "clarification" and follow_up.clarification is not None:
                return follow_up.clarification

        effective_scope = _effective_action_scope(plan)
        if effective_scope is None or effective_scope.kind != ScopeKind.CURRENT_AREA:
            return None
        if _explicit_target_resolves_without_origin(
            plan,
            snapshot,
            scope=effective_scope,
        ):
            return None
        return ClarificationPlan(
            intent=plan.intent,
            reason=MISSING_ORIGIN_CLARIFICATION_REASON,
        )

    if isinstance(plan, QueryPlan):
        effective_scope = _effective_query_scope(plan)
        if effective_scope is not None and effective_scope.kind == ScopeKind.CURRENT_AREA:
            return ClarificationPlan(
                intent=plan.intent,
                reason=MISSING_ORIGIN_CLARIFICATION_REASON,
            )
    return None


def compose_plan_generation_without_origin(
    *,
    runtime: ModelRuntime,
    snapshot: HomeGraphSnapshot,
    satellite_id: str,
    text: str,
    conversation: SatelliteConversationState | None = None,
) -> PlanGenerationResult:
    """Build and parse a ControlPlan when no HA source area is available."""
    conv = conversation or SatelliteConversationState()
    scored = retrieve_candidates(
        snapshot,
        origin_area_id=MISSING_ORIGIN_AREA_SENTINEL,
        request=text,
        conversation=conv,
        limit=1,
    )
    prompt = build_lfm_prompt(
        user_text=text,
        origin=PromptOrigin(
            satellite_id=satellite_id,
            area_name=MISSING_ORIGIN_PROMPT_AREA_NAME,
            area_aliases=[],
        ),
        conversation=conv,
        candidates=[candidate.item for candidate in scored],
        areas=snapshot.areas,
    )
    raw = runtime.generate(prompt)
    plan = parse_model_output(raw.text, intent=text)
    return PlanGenerationResult(
        plan=plan,
        prompt_tokens=raw.prompt_tokens,
        completion_tokens=raw.completion_tokens,
        latency_ms=raw.latency_ms,
        metadata=raw.metadata,
    )


def conversation_response_payload(controller_payload: dict[str, Any]) -> dict[str, Any]:
    """Map a text-controller payload to a conversation_response payload."""

    category = controller_payload.get("category", "")
    reason = controller_payload.get("reason")
    plan = controller_payload.get("plan")
    plan_outcome = plan.get("outcome") if isinstance(plan, dict) else None
    if plan_outcome == "clarification" or (
        isinstance(reason, str) and reason.startswith("clarification required: ")
    ):
        response_type = "clarification"
    elif category == ExecutionCategory.COMPLETED.value:
        response_type = "action_done"
    elif category == ExecutionCategory.NO_ACTION.value:
        response_type = "no_action"
    elif category == ExecutionCategory.REJECTED.value:
        response_type = "rejected"
    elif category == ExecutionCategory.FAILED.value:
        response_type = "failed"
    else:
        response_type = "error"

    content = controller_payload.get("response_content")
    speech = "Done." if content == "\a" else (content if isinstance(content, str) else "")
    return {
        "speech": speech,
        "response_type": response_type,
    }


class TextController(Protocol):
    def handle(
        self,
        *,
        satellite_id: str,
        area_id: str | None,
        text: str,
        correlation_id: str,
        input_type: Literal["text", "audio"] = "text",
        stt_ms: float = 0.0,
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
        ha_gateway_binding: HaGatewayBinding | None = None,
    ) -> None:
        self._runtime = runtime
        self._ha_client = ha_client
        self._graph_store = graph_store
        self._conversation_store = conversation_store
        self._telemetry_sink = telemetry_sink
        self._ha_gateway_binding = ha_gateway_binding

    @staticmethod
    def _execution_payload(outcome: ExecutionOutcome) -> dict[str, Any]:
        plan_payload = (
            outcome.plan.model_dump(mode="json")
            if hasattr(outcome.plan, "model_dump")
            else dict(outcome.plan) if isinstance(outcome.plan, dict) else {}
        )
        policy = resolve_response_policy(outcome)
        return {
            "category": outcome.category.value,
            "reason": outcome.reason,
            "plan": plan_payload,
            "request_id": outcome.request_id,
            "response_mode": policy.mode.value,
            "response_content": policy.content,
        }

    def _refusal_payload(self, *, text: str, message: str) -> dict[str, Any]:
        plan = NoActionPlan(intent=text, reason=message)
        return self._execution_payload(
            ExecutionOutcome(
                category=ExecutionCategory.NO_ACTION,
                plan=plan,
                reason=message,
            ),
        )

    def _execution_refusal(self, *, text: str) -> dict[str, Any] | None:
        refusal = text_execution_refusal(
            graph_snapshot=self._graph_store.snapshot,
            ha_gateway_binding=self._ha_gateway_binding,
        )
        if refusal is None:
            return None
        _code, message = refusal
        return self._refusal_payload(text=text, message=message)

    def _telemetry_area_id(self, area_id: str | None) -> str:
        return area_id or MISSING_ORIGIN_PROMPT_AREA_NAME

    def _origin_area_id(self, area_id: str | None) -> str:
        return area_id or MISSING_ORIGIN_AREA_SENTINEL

    def _generate_plan(
        self,
        *,
        snapshot: HomeGraphSnapshot,
        satellite_id: str,
        area_id: str | None,
        text: str,
        conversation: SatelliteConversationState | None,
    ) -> PlanGenerationResult:
        if area_id is None:
            return compose_plan_generation_without_origin(
                runtime=self._runtime,
                snapshot=snapshot,
                satellite_id=satellite_id,
                text=text,
                conversation=conversation,
            )
        return compose_plan_generation(
            runtime=self._runtime,
            snapshot=snapshot,
            satellite_id=satellite_id,
            area_id=area_id,
            text=text,
            conversation=conversation,
        )

    def _plan_for_execution(
        self,
        generation: PlanGenerationResult,
        snapshot: HomeGraphSnapshot,
        *,
        satellite_id: str,
        area_id: str | None,
    ) -> BaseModel:
        if area_id is not None:
            return generation.plan
        clarification = missing_origin_clarification(
            generation.plan,
            snapshot,
            satellite_id=satellite_id,
            conversation_store=self._conversation_store,
        )
        return clarification or generation.plan

    def handle(
        self,
        *,
        satellite_id: str,
        area_id: str | None,
        text: str,
        correlation_id: str,
        input_type: Literal["text", "audio"] = "text",
        stt_ms: float = 0.0,
    ) -> dict[str, Any]:
        refusal = self._execution_refusal(text=text)
        if refusal is not None:
            return refusal

        snapshot = self._graph_store.snapshot
        assert snapshot is not None

        request_id = correlation_id or str(uuid.uuid4())
        telemetry = InteractionTelemetry(
            correlation_id=correlation_id,
            satellite_id=satellite_id,
            area_id=self._telemetry_area_id(area_id),
            input_type=input_type,
        )
        if stt_ms > 0.0:
            telemetry.record_stage_ms("stt", stt_ms)
        conversation = (
            self._conversation_store.get_state(satellite_id)
            if self._conversation_store is not None
            else None
        )
        with telemetry.time_stage("plan"):
            generation = self._generate_plan(
                snapshot=snapshot,
                satellite_id=satellite_id,
                area_id=area_id,
                text=text,
                conversation=conversation,
            )
        telemetry.set_model_from_generation(generation)
        plan = self._plan_for_execution(
            generation,
            snapshot,
            satellite_id=satellite_id,
            area_id=area_id,
        )
        outcome = execute_control_plan(
            plan,
            snapshot,
            origin_area_id=self._origin_area_id(area_id),
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
        return self._execution_payload(outcome)

    async def handle_async(
        self,
        *,
        satellite_id: str,
        area_id: str | None,
        text: str,
        correlation_id: str,
        input_type: Literal["text", "audio"] = "text",
        stt_ms: float = 0.0,
    ) -> dict[str, Any]:
        refusal = self._execution_refusal(text=text)
        if refusal is not None:
            return refusal

        snapshot = self._graph_store.snapshot
        assert snapshot is not None

        request_id = correlation_id or str(uuid.uuid4())
        telemetry = InteractionTelemetry(
            correlation_id=correlation_id,
            satellite_id=satellite_id,
            area_id=self._telemetry_area_id(area_id),
            input_type=input_type,
        )
        if stt_ms > 0.0:
            telemetry.record_stage_ms("stt", stt_ms)
        conversation = (
            self._conversation_store.get_state(satellite_id)
            if self._conversation_store is not None
            else None
        )
        with telemetry.time_stage("plan"):
            generation = self._generate_plan(
                snapshot=snapshot,
                satellite_id=satellite_id,
                area_id=area_id,
                text=text,
                conversation=conversation,
            )
        telemetry.set_model_from_generation(generation)
        plan = self._plan_for_execution(
            generation,
            snapshot,
            satellite_id=satellite_id,
            area_id=area_id,
        )
        execute = (
            execute_control_plan_async
            if hasattr(self._ha_client, "collect_action_results")
            else execute_control_plan
        )
        if execute is execute_control_plan_async:
            outcome = await execute(
                plan,
                snapshot,
                origin_area_id=self._origin_area_id(area_id),
                ha_client=self._ha_client,
                request_id=request_id,
                conversation_store=self._conversation_store,
                satellite_id=satellite_id,
                telemetry=telemetry,
            )
        else:
            outcome = execute(
                plan,
                snapshot,
                origin_area_id=self._origin_area_id(area_id),
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
        return self._execution_payload(outcome)


def create_text_handler(
    *,
    token: str,
    satellite_registry: SatelliteRegistry,
    graph_store: HomeGraphStore,
    text_controller: TextController | None = None,
    ha_gateway_binding: HaGatewayBinding | None = None,
    require_live_ha: bool = False,
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

        registration = satellite_registry.get(envelope.payload.satellite_id)
        if registration is None:
            error = ErrorResponseEnvelope(
                version=API_VERSION,
                type="error",
                correlation_id=envelope.correlation_id,
                payload={
                    "code": "unknown_satellite",
                    "message": "satellite context is invalid",
                },
            )
            return web.json_response(error.model_dump(mode="json"), status=400)

        if graph_store.snapshot is None:
            error = ErrorResponseEnvelope(
                version=API_VERSION,
                type="error",
                correlation_id=envelope.correlation_id,
                payload={
                    "code": "no_graph",
                    "message": "satellite context is invalid",
                },
            )
            return web.json_response(error.model_dump(mode="json"), status=400)

        area_id = None

        if require_live_ha and ha_gateway_binding is not None:
            refusal = text_execution_refusal(
                graph_snapshot=graph_store.snapshot,
                ha_gateway_binding=ha_gateway_binding,
            )
            if refusal is not None:
                code, message = refusal
                status = 503 if code == "ha_disconnected" else 400
                error = ErrorResponseEnvelope(
                    version=API_VERSION,
                    type="error",
                    correlation_id=envelope.correlation_id,
                    payload={"code": code, "message": message},
                )
                return web.json_response(error.model_dump(mode="json"), status=status)

        if text_controller is None:
            error = ErrorResponseEnvelope(
                version=API_VERSION,
                type="error",
                correlation_id=envelope.correlation_id,
                payload={"code": "not_configured", "message": "text controller unavailable"},
            )
            return web.json_response(error.model_dump(mode="json"), status=503)

        handle_async = getattr(text_controller, "handle_async", None)
        if handle_async is not None:
            payload = await handle_async(
                satellite_id=envelope.payload.satellite_id,
                area_id=area_id,
                text=envelope.payload.text,
                correlation_id=envelope.correlation_id,
            )
        else:
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


def create_live_text_controller(
    ha_gateway_binding: object,
    *,
    runtime: ModelRuntime | None = None,
    graph_store: HomeGraphStore | None = None,
    conversation_store: ConversationStore | None = None,
    telemetry_sink: TelemetrySink | None = None,
) -> OrchestratorTextController:
    """Build a text controller that sends action_request on the live HA WebSocket."""

    from sayso_server.ha_ws_client import BoundHaWsActionClient

    model_runtime = runtime
    if model_runtime is None:
        model_runtime = FakeModelRuntime()
        model_runtime.load()
    return OrchestratorTextController(
        runtime=model_runtime,
        ha_client=BoundHaWsActionClient(ha_gateway_binding),  # type: ignore[arg-type]
        graph_store=graph_store or HomeGraphStore(),
        conversation_store=conversation_store,
        telemetry_sink=telemetry_sink,
        ha_gateway_binding=ha_gateway_binding,  # type: ignore[arg-type]
    )


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
    "ConversationRequestPayload",
    "ErrorResponseEnvelope",
    "MISSING_ORIGIN_CLARIFICATION_REASON",
    "OrchestratorTextController",
    "TextController",
    "TextRequestEnvelope",
    "TextResponseEnvelope",
    "compose_plan_generation_without_origin",
    "conversation_response_payload",
    "create_live_text_controller",
    "create_text_handler",
    "default_text_dependencies",
    "missing_origin_clarification",
    "resolve_conversation_area",
]
