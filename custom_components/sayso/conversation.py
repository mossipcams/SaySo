from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, override

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers import intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import SaySoConfigEntry, SaySoRuntimeData
from .core.area_context import AreaContext, apply_area_context, build_area_context
from .boundary import (
    action_metadata,
    apply_action_summary,
    boundary_schema,
    first_target,
    inference_error_type,
    record_boundary,
    tool_result_data,
    tool_result_failed,
    validate_arguments,
)
from .client import ChatCompletionResult, ToolCall
from .const import (
    DOMAIN,
    DOMAIN_PLURALS,
    ERROR_ACTION_FAILED,
    ERROR_AREA_HAS_NO_DOMAIN,
    ERROR_EMPTY_RESPONSE,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_MEDIA_NOT_PAUSED,
    ERROR_MEDIA_NOT_PLAYING,
    ERROR_NAME_NOT_FOUND,
    ERROR_NO_TIMERS,
    ERROR_NOT_UNDERSTOOD,
    ERROR_REQUEST_TIMEOUT,
    ERROR_TOOL_ITERATION_LIMIT,
    HA_MATCH_FAILED_ERROR,
    HELD_CALL_TTL,
)
from .diagnostics import BoundaryFailureCode, BoundaryPhase
from .core.exceptions import (
    SaySoError,
    SaySoInvalidResponseError,
    SaySoInvalidToolCallsError,
    SaySoInvalidToolEnvelopeError,
    SaySoTimeoutError,
)
from .core.history import model_history
from .core.grounding import GroundingEntity, GroundingHome
from .core.held import HeldState, resolve_held_option
from .core.post_model import (
    ExecutionResult,
    FailureKind,
    Phase,
    TurnInput,
    TurnOutcome,
    ValidationResult,
    run_turn,
)
from .routing import (
    build_routing_catalog,
    build_routing_preferences,
    build_routing_registries,
    identify_command_domain,
)
from .schema import (
    CompiledToolSchema,
    _is_query_tool,
    _unwrap_source_tool,
    build_tool_availability_names,
    build_tool_map,
    compile_llm_tools,
)
from .transcript import chat_log_to_messages as _chat_log_to_messages
from .tracing import ErrorType, Stage, TraceContext

_LOGGER = logging.getLogger(__name__)

_Held = HeldState
_resolve_held_option = resolve_held_option

_BOUNDARY_CODES = {
    FailureKind.SCHEMA_MISMATCH: BoundaryFailureCode.SCHEMA_MISMATCH,
    FailureKind.INVALID_ARGUMENTS: BoundaryFailureCode.INVALID_ARGUMENTS,
    FailureKind.UNAVAILABLE_TOOL: BoundaryFailureCode.UNAVAILABLE_TOOL,
}
_MODEL_LOG_LABELS = {
    Phase.INITIAL: "llama.cpp",
    Phase.CORRECTION: "llama.cpp correction",
    Phase.FOLLOW_UP: "llama.cpp follow-up",
}


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: SaySoConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    async_add_entities([SaySoConversationEntity(config_entry)])


@dataclass(slots=True)
class _Turn:
    entry_id: str
    agent_id: str
    runtime: SaySoRuntimeData
    user_input: conversation.ConversationInput
    chat_log: conversation.ChatLog
    trace: TraceContext
    complete_schema: CompiledToolSchema | None = None
    active_schema: CompiledToolSchema | None = None
    correction_used: bool = False
    area_context: AreaContext | None = None
    exposed_domains: frozenset[str] = frozenset()

    @property
    def messages(self) -> list[dict[str, Any]]:
        messages = model_history(_chat_log_to_messages(self.chat_log.content))
        if self.area_context is None:
            return messages
        return apply_area_context(messages, self.area_context)

    def error(
        self,
        speech: str,
        *,
        stage: Stage | str = Stage.SAYSO_REQUEST,
        error_type: ErrorType = ErrorType.UNKNOWN,
    ) -> conversation.ConversationResult:
        return _error_result(
            self.user_input,
            self.chat_log,
            speech,
            trace=self.trace,
            stage=stage,
            error_type=error_type,
        )

    def text(self, content: str | None) -> conversation.ConversationResult:
        with self.trace.stage(Stage.RESPONSE) as span:
            self.chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(agent_id=self.agent_id, content=content)
            )
            span.metadata["text_length"] = len(content or "")
            return conversation.async_get_result_from_chat_log(
                self.user_input, self.chat_log
            )

    def boundary_failure(
        self,
        code: BoundaryFailureCode,
        phase: BoundaryPhase,
        *,
        ha_error: str | None = None,
        speech: str = ERROR_NOT_UNDERSTOOD,
    ) -> conversation.ConversationResult:
        record_boundary(
            self.entry_id,
            code,
            phase,
            boundary_schema(
                phase,
                active_schema=self.active_schema,
                complete_schema=self.complete_schema,
                correction_used=self.correction_used,
            ),
            ha_error=ha_error,
            trace=self.trace,
        )
        return _error_result(self.user_input, self.chat_log, speech)

    def model_failure(
        self,
        err: BaseException,
        phase: BoundaryPhase,
        *,
        log_label: str = "llama.cpp",
    ) -> conversation.ConversationResult:
        if isinstance(err, SaySoTimeoutError):
            return self.boundary_failure(
                BoundaryFailureCode.REQUEST_TIMEOUT,
                phase,
                speech=ERROR_REQUEST_TIMEOUT,
            )
        if not isinstance(err, SaySoError):
            raise err
        _LOGGER.debug("trace_id=%s %s error: %s", self.trace.trace_id, log_label, err)
        return self.error(
            ERROR_NOT_UNDERSTOOD
            if isinstance(err, SaySoInvalidToolCallsError)
            else ERROR_MODEL_UNAVAILABLE,
            stage=Stage.INFERENCE,
            error_type=(
                ErrorType.INVALID_MODEL_OUTPUT
                if isinstance(err, SaySoInvalidResponseError)
                else ErrorType.MODEL_UNAVAILABLE
            ),
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None,
        correction: bool = False,
    ) -> ChatCompletionResult:
        trace, runtime = self.trace, self.runtime
        span = trace.open(
            Stage.INFERENCE,
            model=runtime.model,
            **({"correction": True} if correction else {}),
        )
        try:
            result = await runtime.engine.async_chat_completion(
                messages,
                tools=tools,
                temperature=runtime.temperature,
                max_tokens=runtime.max_output_tokens,
            )
        except BaseException as err:
            trace.close(span, success=False)
            trace.fail(Stage.INFERENCE, inference_error_type(err), str(err))
            raise
        span.metadata["tool_calls"] = len(result.tool_calls)
        if result.prompt_tokens is not None:
            span.metadata["prompt_tokens"] = result.prompt_tokens
        if result.prefix_hash is not None:
            span.metadata["prefix_hash"] = result.prefix_hash
        if result.reused_tokens is not None:
            span.metadata["reused_tokens"] = result.reused_tokens
        trace.close(span)
        return result

    async def execute(
        self,
        validated: list[tuple[ToolCall, dict[str, Any]]],
        tool_map: dict[str, llm.Tool],
        span: Any,
    ) -> tuple[bool, str | None, str | None]:
        from .ducking import async_wait_for_restores

        await async_wait_for_restores(self.chat_log.hass)
        content = conversation.AssistantContent(
            agent_id=self.agent_id,
            content=None,
            tool_calls=[
                llm.ToolInput(
                    id=tool_call.id,
                    tool_name=tool_map[tool_call.name].name,
                    tool_args=normalized_args,
                )
                for tool_call, normalized_args in validated
            ],
        )
        batch_failed = False
        ha_error: str | None = None
        ha_error_text: str | None = None
        resolved_target: str | None = None
        async for result in self.chat_log.async_add_assistant_content(content):
            data = tool_result_data(result)
            if tool_result_failed(result):
                batch_failed = True
                if ha_error is None:
                    error = data.get("error")
                    if isinstance(error, str) and error:
                        ha_error = error
                        error_text = data.get("error_text")
                        if isinstance(error_text, str):
                            ha_error_text = error_text
            elif resolved_target is None:
                resolved_target = first_target(data)
        if resolved_target is not None:
            span.metadata["target"] = resolved_target
        return batch_failed, ha_error, ha_error_text


def _is_voice_turn(user_input: conversation.ConversationInput) -> bool:
    return bool(getattr(user_input, "satellite_id", None) or user_input.device_id)


def _grounding_home(
    hass: HomeAssistant,
    user_input: conversation.ConversationInput,
    assistant: str,
) -> GroundingHome:
    registry = er.async_get(hass)
    areas = ar.async_get(hass)
    entities = []
    for state in hass.states.async_all():
        if not async_should_expose(hass, assistant, state.entity_id):
            continue
        entry = registry.async_get(state.entity_id)
        area_id = area_id_for(hass, state.entity_id)
        area_entry = areas.async_get_area(area_id) if area_id else None
        area_name = area_entry.name if area_entry is not None else None
        entities.append(
            GroundingEntity(
                state.entity_id,
                state.name,
                tuple(a for a in entry.aliases if isinstance(a, str))
                if entry is not None
                else (),
                state.domain,
                area_name,
                entry.device_id if entry is not None else None,
            )
        )
    return GroundingHome(
        entities=tuple(entities),
        areas={
            area.name: tuple(a for a in area.aliases if isinstance(a, str))
            for area in areas.async_list_areas()
        },
        satellite_area=_satellite_area_name(hass, user_input),
        satellite_device_id=user_input.device_id,
    )


class SaySoConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: SaySoConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = entry.entry_id

        self._held: dict[str, HeldState] = {}
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="SaySo",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def _runtime(self) -> SaySoRuntimeData:
        return self._entry.runtime_data

    @property
    @override
    def supported_languages(self) -> list[str]:
        return ["en"]

    @override
    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self._entry, self)

    @override
    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self._entry)
        await super().async_will_remove_from_hass()

    @override
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        trace = self._runtime.tracer.async_start(user_input.context, user_input.text)
        turn = _Turn(
            entry_id=self._entry.entry_id,
            agent_id=self.entity_id,
            runtime=self._runtime,
            user_input=user_input,
            chat_log=chat_log,
            trace=trace,
        )
        span = trace.open(Stage.SAYSO_REQUEST)
        try:
            return await self._async_answer(turn)
        except Exception as err:
            trace.fail(Stage.SAYSO_REQUEST, ErrorType.UNKNOWN, type(err).__name__)
            raise
        finally:
            trace.close(span, success=trace.success is not False)
            self._runtime.tracer.async_finish(trace)

    async def _async_answer(self, turn: _Turn) -> conversation.ConversationResult:
        chat_log, trace = turn.chat_log, turn.trace
        llm_context = turn.user_input.as_llm_context(DOMAIN)

        with trace.stage(Stage.CONTEXT) as span:
            try:
                await chat_log.async_provide_llm_data(
                    llm_context,
                    turn.runtime.llm_api,
                    turn.runtime.system_prompt,
                    turn.user_input.extra_system_prompt,
                )
            except conversation.ConverseError as err:
                trace.fail(Stage.CONTEXT, ErrorType.UNKNOWN, type(err).__name__)
                return err.as_conversation_result()

            try:
                turn.complete_schema = compile_llm_tools(chat_log.llm_api)
            except SaySoInvalidToolEnvelopeError:
                return turn.error(
                    ERROR_NOT_UNDERSTOOD,
                    stage=Stage.CONTEXT,
                    error_type=ErrorType.SCHEMA_MISMATCH,
                )

            domain_hint = identify_command_domain(
                turn.user_input.text,
                build_routing_catalog(self.hass, assistant=llm_context.assistant),
                registries=build_routing_registries(self.hass),
                preferences=build_routing_preferences(
                    self.hass,
                    llm_context,
                    satellite_id=getattr(turn.user_input, "satellite_id", None),
                ),
            )
            turn.active_schema = turn.complete_schema
            turn.area_context = build_area_context(
                turn.user_input.text,
                {
                    area.name: area.aliases
                    for area in ar.async_get(self.hass).async_list_areas()
                },
                _satellite_area_name(self.hass, turn.user_input),
            )
            turn.exposed_domains = frozenset(
                state.domain
                for state in self.hass.states.async_all()
                if async_should_expose(
                    self.hass, llm_context.assistant, state.entity_id
                )
            )
            span.metadata["domain_hint"] = domain_hint
            span.metadata["tools"] = (
                len(turn.active_schema.tools) if turn.active_schema is not None else 0
            )

        held = self._held.pop(chat_log.conversation_id, None)
        return await self._post_model(turn, held=held)

    async def _async_run_tool_calls(
        self,
        turn: _Turn,
        tool_calls: list[ToolCall],
        *,
        confirmed: bool = False,
    ) -> conversation.ConversationResult:
        return await self._post_model(turn, calls=tool_calls, confirmed=confirmed)

    async def _post_model(
        self,
        turn: _Turn,
        *,
        held: HeldState | None = None,
        calls: list[ToolCall] | None = None,
        confirmed: bool = False,
    ) -> conversation.ConversationResult:
        chat_log, trace = turn.chat_log, turn.trace
        llm_api = chat_log.llm_api
        tools = llm_api.tools if llm_api is not None else []
        tool_map = build_tool_map(tools)

        async def complete(messages: list[dict[str, Any]], *, correction: bool):
            return await turn.complete(
                messages, tools=_tools_of(turn.active_schema), correction=correction
            )

        def validate(tool_calls: list[ToolCall]) -> ValidationResult:
            validated, failures = validate_arguments(tool_calls, tool_map, trace)
            return ValidationResult(validated, failures)

        async def execute(
            validated: list[tuple[ToolCall, dict[str, Any]]],
        ) -> ExecutionResult:
            action_span = trace.open(
                Stage.HA_ACTION, **action_metadata(validated, tool_map)
            )
            batch_failed, ha_error, ha_error_text = await turn.execute(
                validated, tool_map, action_span
            )
            apply_action_summary(trace, action_span.metadata)
            trace.close(action_span, success=not batch_failed)
            return ExecutionResult(batch_failed, ha_error, ha_error_text)

        outcome = await run_turn(
            TurnInput(
                user_text=turn.user_input.text,
                transcript=lambda: turn.messages,
                tools=_tools_of(turn.complete_schema) or (),
                schema_fingerprint=(
                    turn.complete_schema.fingerprint
                    if turn.complete_schema is not None
                    else None
                ),
                available_names=build_tool_availability_names(tools),
                action_tools=frozenset(
                    name for name, tool in tool_map.items() if _is_action_tool(tool)
                ),
                exposed_domains=turn.exposed_domains,
                home=(
                    (
                        lambda: _grounding_home(
                            self.hass,
                            turn.user_input,
                            turn.user_input.as_llm_context(DOMAIN).assistant,
                        )
                    )
                    if _is_voice_turn(turn.user_input)
                    else None
                ),
                tool_api=llm_api is not None,
                iteration_limit=lambda: turn.runtime.max_tool_iterations,
                now=time.monotonic(),
                held_ttl=HELD_CALL_TTL,
                held=held,
                calls=tuple(calls) if calls is not None else None,
                confirmed=confirmed,
            ),
            complete=complete,
            validate=validate,
            execute=execute,
            recorder=trace,
        )
        return self._respond(turn, outcome)

    def _respond(
        self, turn: _Turn, outcome: TurnOutcome
    ) -> conversation.ConversationResult:
        turn.correction_used = outcome.correction_used
        if outcome.held is not None:
            self._held[turn.chat_log.conversation_id] = outcome.held
        failure = outcome.failure
        if failure is None:
            return turn.text(outcome.speech)
        kind = failure.kind
        phase = BoundaryPhase(failure.phase.value)
        if kind in (FailureKind.TIMEOUT, FailureKind.MODEL_FAILED):
            return turn.model_failure(
                failure.error, phase, log_label=_MODEL_LOG_LABELS[failure.phase]
            )
        if kind is FailureKind.EMPTY_RESPONSE:
            return turn.error(
                ERROR_EMPTY_RESPONSE,
                stage=(
                    Stage.INFERENCE
                    if failure.phase is Phase.INITIAL
                    else Stage.RESPONSE
                ),
                error_type=ErrorType.EMPTY_RESPONSE,
            )
        if kind is FailureKind.EMPTY_CORRECTION:
            return turn.error(
                ERROR_NOT_UNDERSTOOD,
                stage=Stage.TOOL_PARSE,
                error_type=ErrorType.INVALID_MODEL_OUTPUT,
            )
        if kind is FailureKind.NO_TOOL_API:
            return turn.error(
                ERROR_NOT_UNDERSTOOD,
                stage=Stage.TOOL_PARSE,
                error_type=ErrorType.UNAVAILABLE_TOOL,
            )
        if kind is FailureKind.TOOL_EXECUTION_FAILED:
            return turn.boundary_failure(
                BoundaryFailureCode.TOOL_EXECUTION_FAILED,
                phase,
                ha_error=failure.ha_error,
                speech=execution_failure_speech(
                    self.hass,
                    turn.user_input.as_llm_context(DOMAIN).assistant,
                    list(failure.validated),
                    failure.ha_error,
                    failure.ha_error_text,
                    turn.user_input.text,
                ),
            )
        if kind is FailureKind.ITERATION_LIMIT:
            return turn.boundary_failure(
                BoundaryFailureCode.ITERATION_LIMIT,
                phase,
                speech=ERROR_TOOL_ITERATION_LIMIT,
            )
        return turn.boundary_failure(_BOUNDARY_CODES[kind], phase)


def _tools_of(
    schema: CompiledToolSchema | None,
) -> tuple[dict[str, Any], ...] | None:
    return schema.tools if schema is not None else None


def area_id_for(
    hass: HomeAssistant,
    entity_id: str | None,
    device_id: str | None = None,
) -> str | None:
    entry = er.async_get(hass).async_get(entity_id) if entity_id else None
    if entry is not None and getattr(entry, "area_id", None):
        return entry.area_id
    device_reg = dr.async_get(hass)
    for candidate in (device_id, entry.device_id if entry is not None else None):
        if not candidate:
            continue
        device = device_reg.async_get(candidate)
        if device is not None and device.area_id:
            return device.area_id
    return None


def execution_failure_speech(
    hass: HomeAssistant,
    assistant_id: str | None,
    validated: list[tuple[ToolCall, dict[str, Any]]],
    ha_error: str | None,
    ha_error_text: str | None = None,
    user_text: str = "",
) -> str:
    if len(validated) != 1:
        return ERROR_ACTION_FAILED
    call, args = validated[0]
    tool_name = call.name.rsplit("__", 1)[-1]
    name = args.get("name")
    if (
        ha_error == "TimerNotFoundError"
        and tool_name
        in {
            intent.INTENT_START_TIMER,
            intent.INTENT_CANCEL_TIMER,
            intent.INTENT_CANCEL_ALL_TIMERS,
            intent.INTENT_INCREASE_TIMER,
            intent.INTENT_DECREASE_TIMER,
            intent.INTENT_PAUSE_TIMER,
            intent.INTENT_UNPAUSE_TIMER,
            intent.INTENT_TIMER_STATUS,
        }
        and not name
    ):
        return ERROR_NO_TIMERS
    if ha_error != HA_MATCH_FAILED_ERROR:
        return ERROR_ACTION_FAILED
    error_text = ha_error_text or ""
    if "MatchFailedReason.NAME" in error_text or error_text.startswith(
        "No entities matched the name"
    ):
        if isinstance(name, str) and name and name.casefold() in user_text.casefold():
            return ERROR_NAME_NOT_FOUND.format(name=name)
        return ERROR_NOT_UNDERSTOOD
    domain = args.get("domain")
    if isinstance(domain, list):
        domain = domain[0] if len(domain) == 1 else None
    if "MatchFailedReason.STATE" in error_text or error_text.startswith(
        "No entities were in the required state"
    ):
        if isinstance(name, str) and name:
            if tool_name == "HassMediaPause":
                return ERROR_MEDIA_NOT_PLAYING.format(name=name)
            if tool_name == "HassMediaUnpause":
                return ERROR_MEDIA_NOT_PAUSED.format(name=name)
        return ERROR_ACTION_FAILED
    area_name = args.get("area")
    if (
        args.get("name")
        or not isinstance(domain, str)
        or not domain
        or not isinstance(area_name, str)
        or not area_name
        or assistant_id is None
    ):
        return ERROR_ACTION_FAILED
    area = ar.async_get(hass).async_get_area_by_name(area_name)
    if area is None:
        return ERROR_ACTION_FAILED
    if any(
        area_id_for(hass, state.entity_id) == area.id
        and async_should_expose(hass, assistant_id, state.entity_id)
        for state in hass.states.async_all(domain)
    ):
        return ERROR_ACTION_FAILED
    return ERROR_AREA_HAS_NO_DOMAIN.format(
        area=area.name.lower(),
        domains=DOMAIN_PLURALS.get(domain, f"{domain.replace('_', ' ')}s"),
    )


def _satellite_area_name(
    hass: HomeAssistant,
    user_input: conversation.ConversationInput,
) -> str | None:
    satellite_id = getattr(user_input, "satellite_id", None)
    area_id = area_id_for(hass, satellite_id, user_input.device_id)
    if area_id is None:
        return None
    area = ar.async_get(hass).async_get_area(area_id)
    return area.name if area is not None else None


def _is_action_tool(tool: llm.Tool) -> bool:
    source = _unwrap_source_tool(tool)
    return isinstance(source, (llm.ActionTool, llm.IntentTool)) and not _is_query_tool(
        source
    )


def _error_result(
    user_input: conversation.ConversationInput,
    chat_log: conversation.ChatLog,
    speech: str,
    *,
    trace: TraceContext | None = None,
    stage: Stage | str = Stage.SAYSO_REQUEST,
    error_type: ErrorType = ErrorType.UNKNOWN,
) -> conversation.ConversationResult:
    if trace is not None:
        trace.fail(stage, error_type, speech)
    intent_response = intent.IntentResponse(language=user_input.language)
    intent_response.async_set_error(
        intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
        speech,
    )
    return conversation.ConversationResult(
        response=intent_response,
        conversation_id=chat_log.conversation_id,
    )
