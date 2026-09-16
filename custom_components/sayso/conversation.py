"""Conversation platform for SaySo.

One user turn runs here: build context, pick a tool schema, ask llama.cpp, and
loop over the tool calls it proposes until there is text to speak. Everything
the loop *decides* about untrusted model output lives in :mod:`.boundary`;
everything it *sends* to the model lives in :mod:`.transcript`.

A turn carries the same six things everywhere — who asked, the chat log, the
trace, the two compiled schemas and whether the one correction has been spent —
so it is one ``_Turn`` object rather than six parameters threaded through every
call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, override

from homeassistant.components import conversation
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers import intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import SaySoConfigEntry, SaySoRuntimeData
from .boundary import (
    action_metadata,
    apply_action_summary,
    boundary_schema,
    first_target,
    inference_error_type,
    is_tool_execution_failure,
    is_well_formed_batch,
    record_boundary,
    validate_arguments,
    validation_failure_code,
)
from .client import ChatCompletionResult, ToolCall
from .const import (
    DOMAIN,
    ERROR_ACTION_FAILED,
    ERROR_EMPTY_RESPONSE,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_REQUEST_TIMEOUT,
    ERROR_TOOL_ITERATION_LIMIT,
)
from .diagnostics import BoundaryFailureCode, BoundaryPhase
from .exceptions import (
    SaySoError,
    SaySoInvalidResponseError,
    SaySoInvalidToolEnvelopeError,
    SaySoTimeoutError,
)
from .routing import (
    build_routing_catalog,
    build_routing_preferences,
    build_routing_registries,
    identify_command_domain,
    select_schema_for_domain,
)
from .schema import (
    CompiledToolSchema,
    _is_query_tool,
    _unwrap_source_tool,
    build_tool_availability_names,
    build_tool_map,
    compile_llm_tools,
    expand_compiled_tool_name_aliases,
)
from .transcript import (
    build_correction_messages,
    chat_log_to_messages as _chat_log_to_messages,
    filtered_miss_failures,
)
from .tracing import ErrorType, Stage, TraceContext

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: SaySoConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the SaySo conversation entity."""
    async_add_entities([SaySoConversationEntity(config_entry)])


@dataclass(slots=True)
class _Turn:
    """One user turn, and the four ways it can end."""

    entry_id: str
    agent_id: str
    runtime: SaySoRuntimeData
    user_input: conversation.ConversationInput
    chat_log: conversation.ChatLog
    trace: TraceContext
    complete_schema: CompiledToolSchema | None = None
    active_schema: CompiledToolSchema | None = None
    correction_used: bool = False

    @property
    def messages(self) -> list[dict[str, Any]]:
        """The conversation so far, in llama.cpp's message shape."""
        return _chat_log_to_messages(self.chat_log.content)

    def error(
        self,
        speech: str,
        *,
        stage: Stage | str = Stage.SAYSO_REQUEST,
        error_type: ErrorType = ErrorType.UNKNOWN,
    ) -> conversation.ConversationResult:
        """End the turn with a spoken error, failing the trace at ``stage``."""
        return _error_result(
            self.user_input,
            self.chat_log,
            speech,
            trace=self.trace,
            stage=stage,
            error_type=error_type,
        )

    def text(self, content: str | None) -> conversation.ConversationResult:
        """End the turn with a spoken answer."""
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
        speech: str = ERROR_ACTION_FAILED,
    ) -> conversation.ConversationResult:
        """End the turn at the model boundary, counting the failure."""
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
        # ``record_boundary`` already failed the trace at the right stage.
        return _error_result(self.user_input, self.chat_log, speech)

    def model_failure(
        self,
        err: BaseException,
        phase: BoundaryPhase,
        *,
        log_label: str = "llama.cpp",
    ) -> conversation.ConversationResult:
        """Map a llama.cpp failure to a spoken error.

        A timeout is the only one worth a boundary counter: it says llama.cpp
        was reachable but too slow, which is a different operational problem
        from it being down or answering nonsense.
        """
        if isinstance(err, SaySoTimeoutError):
            return self.boundary_failure(
                BoundaryFailureCode.REQUEST_TIMEOUT,
                phase,
                speech=ERROR_REQUEST_TIMEOUT,
            )
        if not isinstance(err, SaySoError):
            raise err
        _LOGGER.debug(
            "trace_id=%s %s error: %s", self.trace.trace_id, log_label, err
        )
        return self.error(
            ERROR_MODEL_UNAVAILABLE,
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
        """Run one traced llama.cpp completion."""
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
        trace.close(span)
        return result

    async def execute(
        self,
        validated: list[tuple[ToolCall, dict[str, Any]]],
        tool_map: dict[str, llm.Tool],
        span: Any,
    ) -> tuple[bool, str | None]:
        """Run one validated batch through Home Assistant, in model order."""
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
        resolved_target: str | None = None
        async for result in self.chat_log.async_add_assistant_content(content):
            if is_tool_execution_failure(result.tool_result):
                batch_failed = True
                if ha_error is None:
                    error = result.tool_result.get("error")
                    if isinstance(error, str) and error:
                        ha_error = error
            elif resolved_target is None:
                resolved_target = first_target(result.tool_result)
        if resolved_target is not None:
            # Home Assistant's own resolution beats the name the model asked for.
            span.metadata["target"] = resolved_target
        return batch_failed, ha_error


class SaySoConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """SaySo conversation agent backed by llama.cpp."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: SaySoConfigEntry) -> None:
        """Initialize the conversation entity."""
        self._entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="SaySo",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def _runtime(self) -> SaySoRuntimeData:
        """Return runtime data for the config entry."""
        return self._entry.runtime_data

    @property
    @override
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return MATCH_ALL

    @override
    async def async_added_to_hass(self) -> None:
        """Register as the conversation agent for this config entry."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self._entry, self)

    @override
    async def async_will_remove_from_hass(self) -> None:
        """Unregister the conversation agent."""
        conversation.async_unset_agent(self.hass, self._entry)
        await super().async_will_remove_from_hass()

    @override
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle a user message with llama.cpp, tracing the whole turn."""
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
        """Provide context, route to a schema, and ask llama.cpp once."""
        chat_log, trace = turn.chat_log, turn.trace
        llm_context = turn.user_input.as_llm_context(DOMAIN)

        with trace.stage(Stage.CONTEXT) as span:
            try:
                await chat_log.async_provide_llm_data(
                    llm_context,
                    turn.runtime.llm_api,
                    _system_prompt_with_area(
                        turn.runtime.system_prompt, self.hass, turn.user_input
                    ),
                    turn.user_input.extra_system_prompt,
                )
            except conversation.ConverseError as err:
                trace.fail(Stage.CONTEXT, ErrorType.UNKNOWN, type(err).__name__)
                return err.as_conversation_result()

            try:
                turn.complete_schema = compile_llm_tools(chat_log.llm_api)
            except SaySoInvalidToolEnvelopeError:
                return turn.error(
                    ERROR_ACTION_FAILED,
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
            turn.active_schema = (
                select_schema_for_domain(
                    turn.complete_schema, chat_log.llm_api.tools, domain_hint
                )
                if turn.complete_schema is not None
                else None
            )
            messages = turn.messages
            span.metadata["domain_hint"] = domain_hint
            span.metadata["tools"] = (
                len(turn.active_schema.tools) if turn.active_schema is not None else 0
            )

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
        self, turn: _Turn, tool_calls: list[ToolCall]
    ) -> conversation.ConversationResult:
        """Execute tool calls sequentially until final text or iteration limit."""
        chat_log, trace = turn.chat_log, turn.trace
        if chat_log.llm_api is None:
            return turn.error(
                ERROR_ACTION_FAILED,
                stage=Stage.TOOL_PARSE,
                error_type=ErrorType.UNAVAILABLE_TOOL,
            )

        # Every tool Home Assistant currently exposes, versus the subset the
        # model was actually shown. A name in the first but not the second is a
        # routing miss and is recoverable; a name in neither never executes.
        available_tools = build_tool_availability_names(chat_log.llm_api.tools)
        offered_tools = (
            expand_compiled_tool_name_aliases(
                {tool["function"]["name"] for tool in turn.active_schema.tools}
            )
            if turn.active_schema is not None
            else available_tools
        )
        tool_map = build_tool_map(chat_log.llm_api.tools)
        iteration = 1
        tools_executed = False
        current = tool_calls
        phase = BoundaryPhase.INITIAL

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

            # A tool the active subset hid and an argument Home Assistant
            # rejects are the same failure — the model saw the wrong contract —
            # so they report identically and share one correction budget.
            misses = [call for call in current if call.name not in offered_tools]
            if misses:
                validated, failures = [], filtered_miss_failures(misses)
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
                # Close tool parsing before the correction request so correction
                # latency lands in the inference stage, not in tool_parse_ms.
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
                offered_tools = available_tools
                if not corrected.tool_calls:
                    return turn.error(
                        ERROR_ACTION_FAILED,
                        stage=Stage.TOOL_PARSE,
                        error_type=ErrorType.INVALID_MODEL_OUTPUT,
                    )
                current = corrected.tool_calls
                continue

            trace.close(parse_span)
            action_span = trace.open(
                Stage.HA_ACTION, **action_metadata(validated, tool_map)
            )
            batch_failed, ha_error = await turn.execute(
                validated, tool_map, action_span
            )
            apply_action_summary(trace, action_span.metadata)
            trace.close(action_span, success=not batch_failed)

            if batch_failed:
                return turn.boundary_failure(
                    BoundaryFailureCode.TOOL_EXECUTION_FAILED,
                    BoundaryPhase.EXECUTION,
                    ha_error=ha_error,
                )

            tools_executed = True
            if len(validated) == 1 and _is_action_tool(tool_map[validated[0][0].name]):
                return turn.text("Done.")

            phase = BoundaryPhase.FOLLOW_UP
            try:
                follow_up = await turn.complete(
                    turn.messages, tools=_tools_of(turn.active_schema)
                )
            except SaySoError as err:
                return turn.model_failure(
                    err, phase, log_label="llama.cpp follow-up"
                )

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
    """Return a compiled schema's tools, or nothing when there is no schema."""
    return schema.tools if schema is not None else None


def _system_prompt_with_area(
    system_prompt: str,
    hass: HomeAssistant,
    user_input: conversation.ConversationInput,
) -> str:
    """Add the requesting device's area to model context when available."""
    device_reg = dr.async_get(hass)
    device_ids = [user_input.device_id]
    satellite_id = getattr(user_input, "satellite_id", None)
    if satellite_id:
        satellite = er.async_get(hass).async_get(satellite_id)
        if satellite is not None and satellite.device_id is not None:
            device_ids.append(satellite.device_id)
    for device_id in device_ids:
        if not device_id:
            continue
        device = device_reg.async_get(device_id)
        if device is None or device.area_id is None:
            continue
        area = ar.async_get(hass).async_get_area(device.area_id)
        if area is not None:
            return f"{system_prompt}\narea={area.name}"
    return system_prompt


def _is_action_tool(tool: llm.Tool) -> bool:
    """Return whether a tool mutates Home Assistant state."""
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
    """Build a short spoken error result, finalizing the trace against ``stage``."""
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
