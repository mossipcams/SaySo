from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, override

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers import intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import SaySoConfigEntry, SaySoRuntimeData
from .area_context import AreaContext, apply_area_context, build_area_context
from .boundary import (
    action_metadata,
    apply_action_summary,
    boundary_schema,
    first_target,
    inference_error_type,
    is_tool_execution_failure,
    is_well_formed_batch,
    record_boundary,
    tool_result_data,
    validate_arguments,
    validation_failure_code,
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
    ERROR_CANCELLED,
    HA_MATCH_FAILED_ERROR,
    HELD_CALL_TTL,
)
from .diagnostics import BoundaryFailureCode, BoundaryPhase
from .exceptions import (
    SaySoError,
    SaySoInvalidResponseError,
    SaySoInvalidToolCallsError,
    SaySoInvalidToolEnvelopeError,
    SaySoTimeoutError,
)
from .history import model_history
from .grounding import (
    GroundingEntity,
    GroundingHome,
    clarify,
    is_grounded,
    drop_unspoken_area,
    support,
    tokens,
)
from .routing import (
    build_routing_catalog,
    build_routing_preferences,
    build_routing_registries,
    identify_command_domain,
)
from .schema import (
    CompiledToolSchema,
    ToolArgumentFailureCode,
    ToolArgumentValidationError,
    _is_query_tool,
    _unwrap_source_tool,
    build_tool_availability_names,
    build_tool_map,
    compile_llm_tools,
)
from .tool_contract import check_tool_call, tools_by_name
from .transcript import (
    build_correction_messages,
    chat_log_to_messages as _chat_log_to_messages,
)
from .tracing import ErrorType, Stage, TraceContext

_LOGGER = logging.getLogger(__name__)

YES_ANSWER = {"yes", "yeah", "yep", "yup", "sure", "correct", "ok", "okay"}
NO_ANSWER = {
    "no",
    "nope",
    "nah",
    "cancel",
    "neither",
    "never",
    "mind",
    "stop",
}
FILLER = {"the", "it", "that", "one", "please", "do", "thanks", "thank", "you"}


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
            if is_tool_execution_failure(data):
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


def _resolve_held_option(held: _Held, text: str) -> GroundingEntity | None:
    words = tokens(text)
    if (
        len(held.options) == 1
        and any(word in YES_ANSWER for word in words)
        and all(word in YES_ANSWER | FILLER for word in words)
    ):
        return held.options[0]
    scored = [
        (
            max(
                support(label.casefold(), text)
                for label in (option.name, *option.aliases)
            ),
            option,
        )
        for option in held.options
    ]
    best_score, best_option = max(scored, key=lambda pair: pair[0])
    if best_score >= 0.5 and all(
        best_score > score for score, option in scored if option is not best_option
    ):
        return best_option
    return None


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


@dataclass(frozen=True)
class _Held:
    calls: list[ToolCall]
    index: int
    options: tuple[GroundingEntity, ...]
    expires: float


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

        self._held: dict[str, _Held] = {}
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
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

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
            messages = turn.messages
            span.metadata["domain_hint"] = domain_hint
            span.metadata["tools"] = (
                len(turn.active_schema.tools) if turn.active_schema is not None else 0
            )

        held = self._held.pop(chat_log.conversation_id, None)
        if (
            held is not None
            and time.monotonic() < held.expires
            and _is_voice_turn(turn.user_input)
        ):
            words = tokens(turn.user_input.text)
            if (
                words
                and all(word in YES_ANSWER | NO_ANSWER | FILLER for word in words)
                and any(word in NO_ANSWER for word in words)
                and not any(word in YES_ANSWER for word in words)
            ):
                return turn.text(ERROR_CANCELLED)

            option = _resolve_held_option(held, turn.user_input.text)
            if option is not None:
                base_args = held.calls[held.index].arguments
                new_args = {
                    key: value
                    for key, value in base_args.items()
                    if key not in {"area", "floor", "domain", "device_class"}
                }
                new_args["name"] = option.name
                calls = list(held.calls)
                held_call = held.calls[held.index]
                calls[held.index] = ToolCall(
                    id=held_call.id,
                    name=held_call.name,
                    arguments=new_args,
                )
                return await self._async_run_tool_calls(turn, calls, confirmed=True)

        try:
            result = await turn.complete(messages, tools=_tools_of(turn.active_schema))
        except SaySoError as err:
            return turn.model_failure(err, BoundaryPhase.INITIAL)

        if result.tool_calls:
            return await self._async_run_tool_calls(turn, result.tool_calls)
        if not (result.content or "").strip():
            return turn.error(
                ERROR_EMPTY_RESPONSE,
                stage=Stage.INFERENCE,
                error_type=ErrorType.EMPTY_RESPONSE,
            )
        return turn.text(result.content)

    async def _async_run_tool_calls(
        self,
        turn: _Turn,
        tool_calls: list[ToolCall],
        *,
        confirmed: bool = False,
    ) -> conversation.ConversationResult:
        chat_log, trace = turn.chat_log, turn.trace
        if chat_log.llm_api is None:
            return turn.error(
                ERROR_NOT_UNDERSTOOD,
                stage=Stage.TOOL_PARSE,
                error_type=ErrorType.UNAVAILABLE_TOOL,
            )

        available_tools = build_tool_availability_names(chat_log.llm_api.tools)
        compiled_tools = tools_by_name(
            turn.complete_schema.tools if turn.complete_schema is not None else ()
        )
        tool_map = build_tool_map(chat_log.llm_api.tools)
        iteration = 1
        tools_executed = False
        current = tool_calls
        phase = BoundaryPhase.INITIAL

        suppress_grounding = confirmed

        while True:
            parse_span = trace.open(Stage.TOOL_PARSE, calls=len(current))

            if not is_well_formed_batch(current):
                return turn.boundary_failure(
                    BoundaryFailureCode.INVALID_ARGUMENTS, phase
                )
            if any(call.name not in available_tools for call in current):
                return turn.boundary_failure(
                    BoundaryFailureCode.UNAVAILABLE_TOOL, phase
                )

            contract_failures = _contract_failures(
                current, compiled_tools, turn.exposed_domains
            )
            if contract_failures:
                validated, failures = [], contract_failures
            else:
                validated, failures = validate_arguments(current, tool_map, trace)

            if failures:
                repairable = not (
                    validated
                    or turn.correction_used
                    or tools_executed
                    or turn.complete_schema is None
                )
                if not repairable:
                    return turn.boundary_failure(
                        validation_failure_code(failures[0][1]), phase
                    )

                phase = BoundaryPhase.CORRECTION
                trace.close(parse_span, success=False)
                try:
                    corrected = await turn.complete(
                        build_correction_messages(
                            turn.messages,
                            failures,
                            available_tools,
                            turn.complete_schema.fingerprint,
                        ),
                        tools=turn.complete_schema.tools,
                        correction=True,
                    )
                except SaySoError as err:
                    return turn.model_failure(
                        err, phase, log_label="llama.cpp correction"
                    )

                turn.correction_used = True
                if not corrected.tool_calls:
                    return turn.error(
                        ERROR_NOT_UNDERSTOOD,
                        stage=Stage.TOOL_PARSE,
                        error_type=ErrorType.INVALID_MODEL_OUTPUT,
                    )
                current = corrected.tool_calls
                continue

            if _is_voice_turn(turn.user_input) and not suppress_grounding:
                home = _grounding_home(
                    self.hass,
                    turn.user_input,
                    turn.user_input.as_llm_context(DOMAIN).assistant,
                )
                normalized: list[tuple[ToolCall, dict]] = []
                for call, args in validated:
                    new_args = drop_unspoken_area(
                        call.name, args, turn.user_input.text, home
                    )
                    if new_args is not args:
                        parse_span.metadata["area_dropped"] = {
                            "tool": call.name,
                            "name": args.get("name"),
                            "area": args.get("area"),
                        }
                    normalized.append((call, new_args))
                validated = normalized
                for i, (call, args) in enumerate(validated):
                    if not is_grounded(call.name, args, turn.user_input.text, home):
                        clar = clarify(call.name, args, turn.user_input.text, home)
                        parse_span.metadata["clarify"] = clar.speech
                        trace.close(parse_span, success=False)
                        if clar.options:
                            self._held[chat_log.conversation_id] = _Held(
                                [c for c, _ in validated],
                                i,
                                clar.options,
                                time.monotonic() + HELD_CALL_TTL,
                            )
                        return turn.text(clar.speech)
            trace.close(parse_span)
            action_span = trace.open(
                Stage.HA_ACTION, **action_metadata(validated, tool_map)
            )
            batch_failed, ha_error, ha_error_text = await turn.execute(
                validated, tool_map, action_span
            )
            apply_action_summary(trace, action_span.metadata)
            trace.close(action_span, success=not batch_failed)

            if batch_failed:
                return turn.boundary_failure(
                    BoundaryFailureCode.TOOL_EXECUTION_FAILED,
                    BoundaryPhase.EXECUTION,
                    ha_error=ha_error,
                    speech=execution_failure_speech(
                        self.hass,
                        turn.user_input.as_llm_context(DOMAIN).assistant,
                        validated,
                        ha_error,
                        ha_error_text,
                        turn.user_input.text,
                    ),
                )

            tools_executed = True

            suppress_grounding = False
            if len(validated) == 1 and _is_action_tool(tool_map[validated[0][0].name]):
                return turn.text("Done.")

            phase = BoundaryPhase.FOLLOW_UP
            try:
                follow_up = await turn.complete(
                    turn.messages, tools=_tools_of(turn.active_schema)
                )
            except SaySoError as err:
                return turn.model_failure(err, phase, log_label="llama.cpp follow-up")

            if follow_up.tool_calls:
                if iteration >= turn.runtime.max_tool_iterations:
                    return turn.boundary_failure(
                        BoundaryFailureCode.ITERATION_LIMIT,
                        phase,
                        speech=ERROR_TOOL_ITERATION_LIMIT,
                    )
                iteration += 1
                current = follow_up.tool_calls
                continue

            if not (follow_up.content or "").strip():
                return turn.error(
                    ERROR_EMPTY_RESPONSE,
                    stage=Stage.RESPONSE,
                    error_type=ErrorType.EMPTY_RESPONSE,
                )
            return turn.text(follow_up.content)


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


def _contract_failures(
    tool_calls: list[ToolCall],
    compiled_tools: dict[str, Any],
    exposed_domains: frozenset[str],
) -> list[tuple[ToolCall, ToolArgumentValidationError]]:
    if not compiled_tools:
        return []
    failures = []
    for call in tool_calls:
        violation = check_tool_call(
            call.name, call.arguments, compiled_tools, exposed_domains or None
        )
        if violation is not None:
            failures.append(
                (
                    call,
                    ToolArgumentValidationError(
                        code=ToolArgumentFailureCode(violation.code),
                        message=violation.message,
                        tool_name=call.name,
                    ),
                )
            )
    return failures


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
