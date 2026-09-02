"""Conversation platform for SaySo."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, override

from homeassistant.components import conversation
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import SaySoConfigEntry, SaySoRuntimeData
from .client import ChatCompletionResult, ToolCall
from .const import (
    DOMAIN,
    ERROR_ACTION_FAILED,
    ERROR_EMPTY_RESPONSE,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_REQUEST_TIMEOUT,
    ERROR_TOOL_ITERATION_LIMIT,
)
from .exceptions import (
    SaySoConnectionError,
    SaySoError,
    SaySoHttpError,
    SaySoInvalidResponseError,
    SaySoTimeoutError,
)
from .diagnostics import (
    BoundaryFailureCode,
    BoundaryPhase,
    record_boundary_failure,
)
from .routing import (
    build_routing_catalog,
    build_routing_preferences,
    build_routing_registries,
    identify_command_domain,
    select_tools_for_domain,
)
from .schema import (
    CompiledToolSchema,
    ToolArgumentFailureCode,
    ToolArgumentValidationError,
    build_tool_map,
    compile_llm_tools,
    format_synthetic_validation_error,
    validate_tool_arguments,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: SaySoConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the SaySo conversation entity."""
    async_add_entities([SaySoConversationEntity(config_entry)])


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
        """Handle a user message with llama.cpp."""
        runtime = self._runtime

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                runtime.llm_api,
                runtime.system_prompt,
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        compiled_schema = compile_llm_tools(chat_log.llm_api)
        llm_context = user_input.as_llm_context(DOMAIN)
        domain_hint = identify_command_domain(
            user_input.text,
            build_routing_catalog(self.hass, assistant=llm_context.assistant),
            registries=build_routing_registries(self.hass),
            preferences=build_routing_preferences(self.hass, llm_context),
        )
        active_tools = (
            select_tools_for_domain(
                compiled_schema.tools,
                chat_log.llm_api.tools,
                domain_hint,
            )
            if compiled_schema is not None
            else None
        )
        messages = _chat_log_to_messages(chat_log.content)

        try:
            result = await runtime.client.chat_completion(
                messages,
                model=runtime.model,
                tools=active_tools,
                temperature=runtime.temperature,
                max_tokens=runtime.max_output_tokens,
            )
        except (
            SaySoTimeoutError,
            SaySoConnectionError,
            SaySoHttpError,
            SaySoInvalidResponseError,
            SaySoError,
        ) as err:
            return _client_exception_result(
                self._entry.entry_id,
                BoundaryPhase.INITIAL,
                compiled_schema,
                user_input,
                chat_log,
                err,
            )

        if result.tool_calls:
            return await self._async_handle_tool_calls(
                user_input,
                chat_log,
                runtime,
                result.tool_calls,
                compiled_schema,
                active_tools,
            )

        error_message = _validate_text_completion(result)
        if error_message is not None:
            return _error_result(user_input, chat_log, error_message)

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=result.content,
            )
        )
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_handle_tool_calls(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
        runtime: SaySoRuntimeData,
        tool_calls: list[ToolCall],
        compiled_schema: CompiledToolSchema | None,
        active_tools: tuple[dict[str, Any], ...] | None,
    ) -> conversation.ConversationResult:
        """Execute tool calls sequentially until final text or iteration limit."""
        if chat_log.llm_api is None:
            return _error_result(user_input, chat_log, ERROR_ACTION_FAILED)

        complete_allowed_tools = {tool.name for tool in chat_log.llm_api.tools}
        validation_tool_names = (
            {tool["function"]["name"] for tool in active_tools}
            if active_tools is not None
            else complete_allowed_tools
        )
        request_tools = active_tools
        tool_map = build_tool_map(chat_log.llm_api.tools)
        iteration = 1
        correction_used = False
        tools_executed = False
        current_tool_calls = tool_calls
        phase = BoundaryPhase.INITIAL

        while True:
            if not _validate_tool_call_batch_structure(current_tool_calls):
                _record_boundary(
                    self._entry.entry_id,
                    BoundaryFailureCode.INVALID_ARGUMENTS,
                    phase,
                    compiled_schema,
                )
                return _error_result(user_input, chat_log, ERROR_ACTION_FAILED)

            unavailable_calls = [
                tool_call
                for tool_call in current_tool_calls
                if tool_call.name not in complete_allowed_tools
            ]
            if unavailable_calls:
                _record_boundary(
                    self._entry.entry_id,
                    BoundaryFailureCode.UNAVAILABLE_TOOL,
                    phase,
                    compiled_schema,
                )
                return _error_result(user_input, chat_log, ERROR_ACTION_FAILED)

            filtered_misses = [
                tool_call
                for tool_call in current_tool_calls
                if tool_call.name not in validation_tool_names
            ]
            if filtered_misses:
                if correction_used or tools_executed or compiled_schema is None:
                    _record_boundary(
                        self._entry.entry_id,
                        BoundaryFailureCode.SCHEMA_MISMATCH,
                        phase,
                        compiled_schema,
                    )
                    return _error_result(user_input, chat_log, ERROR_ACTION_FAILED)

                phase = BoundaryPhase.CORRECTION
                correction_messages = _build_filtered_miss_correction_messages(
                    _chat_log_to_messages(chat_log.content),
                    filtered_misses,
                    complete_allowed_tools,
                    compiled_schema.fingerprint,
                )
                try:
                    correction_response = await runtime.client.chat_completion(
                        correction_messages,
                        model=runtime.model,
                        tools=compiled_schema.tools,
                        temperature=runtime.temperature,
                        max_tokens=runtime.max_output_tokens,
                    )
                except (
                    SaySoTimeoutError,
                    SaySoConnectionError,
                    SaySoHttpError,
                    SaySoInvalidResponseError,
                    SaySoError,
                ) as err:
                    return _client_exception_result(
                        self._entry.entry_id,
                        BoundaryPhase.CORRECTION,
                        compiled_schema,
                        user_input,
                        chat_log,
                        err,
                        log_label="llama.cpp correction",
                    )

                correction_used = True
                validation_tool_names = complete_allowed_tools
                if not correction_response.tool_calls:
                    return _error_result(user_input, chat_log, ERROR_ACTION_FAILED)
                current_tool_calls = correction_response.tool_calls
                continue

            validated_tool_calls: list[tuple[ToolCall, dict[str, Any]]] = []
            validation_failures: list[
                tuple[ToolCall, ToolArgumentValidationError]
            ] = []
            for tool_call in current_tool_calls:
                tool = tool_map[tool_call.name]
                normalized_args, validation_error = validate_tool_arguments(
                    tool,
                    tool_call.arguments,
                )
                if validation_error is not None:
                    _LOGGER.debug(
                        "Tool argument validation failed for %s (%s): %s",
                        tool_call.name,
                        validation_error.code,
                        validation_error.message,
                    )
                    validation_failures.append((tool_call, validation_error))
                else:
                    validated_tool_calls.append((tool_call, normalized_args))

            if validation_failures:
                if (
                    validated_tool_calls
                    or correction_used
                    or tools_executed
                    or compiled_schema is None
                ):
                    _record_boundary(
                        self._entry.entry_id,
                        _validation_failure_code(validation_failures[0][1]),
                        phase,
                        compiled_schema,
                    )
                    return _error_result(user_input, chat_log, ERROR_ACTION_FAILED)

                phase = BoundaryPhase.CORRECTION
                correction_messages = _build_pre_execution_correction_messages(
                    _chat_log_to_messages(chat_log.content),
                    validation_failures,
                    complete_allowed_tools,
                    compiled_schema.fingerprint,
                )
                try:
                    correction_response = await runtime.client.chat_completion(
                        correction_messages,
                        model=runtime.model,
                        tools=compiled_schema.tools,
                        temperature=runtime.temperature,
                        max_tokens=runtime.max_output_tokens,
                    )
                except (
                    SaySoTimeoutError,
                    SaySoConnectionError,
                    SaySoHttpError,
                    SaySoInvalidResponseError,
                    SaySoError,
                ) as err:
                    return _client_exception_result(
                        self._entry.entry_id,
                        BoundaryPhase.CORRECTION,
                        compiled_schema,
                        user_input,
                        chat_log,
                        err,
                        log_label="llama.cpp correction",
                    )

                correction_used = True
                validation_tool_names = complete_allowed_tools
                if not correction_response.tool_calls:
                    return _error_result(user_input, chat_log, ERROR_ACTION_FAILED)
                current_tool_calls = correction_response.tool_calls
                continue

            assistant_content = conversation.AssistantContent(
                agent_id=self.entity_id,
                content=None,
                tool_calls=[
                    llm.ToolInput(
                        id=tool_call.id,
                        tool_name=tool_call.name,
                        tool_args=normalized_args,
                    )
                    for tool_call, normalized_args in validated_tool_calls
                ],
            )
            batch_failed = False
            async for _tool_result in chat_log.async_add_assistant_content(
                assistant_content
            ):
                if "error" in _tool_result.tool_result:
                    batch_failed = True

            if batch_failed:
                _record_boundary(
                    self._entry.entry_id,
                    BoundaryFailureCode.TOOL_EXECUTION_FAILED,
                    BoundaryPhase.EXECUTION,
                    compiled_schema,
                )
                return _error_result(user_input, chat_log, ERROR_ACTION_FAILED)

            tools_executed = True
            phase = BoundaryPhase.FOLLOW_UP

            messages = _chat_log_to_messages(chat_log.content)
            try:
                follow_up = await runtime.client.chat_completion(
                    messages,
                    model=runtime.model,
                    tools=request_tools,
                    temperature=runtime.temperature,
                    max_tokens=runtime.max_output_tokens,
                )
            except (
                SaySoTimeoutError,
                SaySoConnectionError,
                SaySoHttpError,
                SaySoInvalidResponseError,
                SaySoError,
            ) as err:
                return _client_exception_result(
                    self._entry.entry_id,
                    BoundaryPhase.FOLLOW_UP,
                    compiled_schema,
                    user_input,
                    chat_log,
                    err,
                    log_label="llama.cpp follow-up",
                )

            if follow_up.tool_calls:
                if iteration >= runtime.max_tool_iterations:
                    _record_boundary(
                        self._entry.entry_id,
                        BoundaryFailureCode.ITERATION_LIMIT,
                        BoundaryPhase.FOLLOW_UP,
                        compiled_schema,
                    )
                    return _error_result(
                        user_input, chat_log, ERROR_TOOL_ITERATION_LIMIT
                    )
                iteration += 1
                current_tool_calls = follow_up.tool_calls
                continue

            error_message = _validate_text_completion(follow_up)
            if error_message is not None:
                return _error_result(user_input, chat_log, error_message)

            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=self.entity_id,
                    content=follow_up.content,
                )
            )
            return conversation.async_get_result_from_chat_log(user_input, chat_log)


def _record_boundary(
    entry_id: str,
    code: BoundaryFailureCode,
    phase: BoundaryPhase,
    compiled_schema: CompiledToolSchema | None,
) -> None:
    """Record one boundary failure and log its stable code and phase."""
    fingerprint = compiled_schema.fingerprint if compiled_schema else None
    record_boundary_failure(
        entry_id,
        code,
        phase,
        fingerprint=fingerprint,
    )
    _LOGGER.debug("SaySo boundary failure: code=%s phase=%s", code.value, phase.value)


def _client_exception_result(
    entry_id: str,
    phase: BoundaryPhase,
    compiled_schema: CompiledToolSchema | None,
    user_input: conversation.ConversationInput,
    chat_log: conversation.ChatLog,
    err: BaseException,
    *,
    log_label: str = "llama.cpp",
) -> conversation.ConversationResult:
    """Map a SaySo client exception to a spoken error after recording timeout boundaries."""
    if isinstance(err, SaySoTimeoutError):
        _record_boundary(
            entry_id,
            BoundaryFailureCode.REQUEST_TIMEOUT,
            phase,
            compiled_schema,
        )
        return _error_result(user_input, chat_log, ERROR_REQUEST_TIMEOUT)
    if isinstance(err, SaySoConnectionError):
        return _error_result(user_input, chat_log, ERROR_MODEL_UNAVAILABLE)
    if isinstance(err, (SaySoHttpError, SaySoInvalidResponseError)):
        _LOGGER.debug("%s response error: %s", log_label, err)
        return _error_result(user_input, chat_log, ERROR_MODEL_UNAVAILABLE)
    if isinstance(err, SaySoError):
        _LOGGER.debug("SaySo error: %s", err)
        return _error_result(user_input, chat_log, ERROR_MODEL_UNAVAILABLE)
    raise err


def _validation_failure_code(
    error: ToolArgumentValidationError,
) -> BoundaryFailureCode:
    """Map a tool-argument validation error to a boundary diagnostic code."""
    if error.code == ToolArgumentFailureCode.SCHEMA_MISMATCH:
        return BoundaryFailureCode.SCHEMA_MISMATCH
    return BoundaryFailureCode.INVALID_ARGUMENTS


def _batch_validation_failure_code(
    tool_calls: list[ToolCall],
    allowed_tools: set[str],
) -> BoundaryFailureCode:
    """Map a batch prevalidation failure to a boundary diagnostic code."""
    for tool_call in tool_calls:
        if tool_call.name not in allowed_tools:
            return BoundaryFailureCode.UNAVAILABLE_TOOL
    return BoundaryFailureCode.INVALID_ARGUMENTS


def _build_filtered_miss_correction_messages(
    base_messages: list[dict[str, Any]],
    filtered_misses: list[ToolCall],
    allowed_tools: set[str],
    fingerprint: str,
) -> list[dict[str, Any]]:
    """Append a synthetic transcript for one filtered-schema correction request."""
    validation_failures = [
        (
            tool_call,
            ToolArgumentValidationError(
                code=ToolArgumentFailureCode.SCHEMA_MISMATCH,
                message=(
                    f"Tool {tool_call.name} is not available in the active schema subset"
                ),
                tool_name=tool_call.name,
            ),
        )
        for tool_call in filtered_misses
    ]
    return _build_pre_execution_correction_messages(
        base_messages,
        validation_failures,
        allowed_tools,
        fingerprint,
    )


def _build_pre_execution_correction_messages(
    base_messages: list[dict[str, Any]],
    validation_failures: list[tuple[ToolCall, ToolArgumentValidationError]],
    allowed_tools: set[str],
    fingerprint: str,
) -> list[dict[str, Any]]:
    """Append a synthetic assistant/tool transcript for one correction request."""
    messages = list(base_messages)
    allowed_tool_names = sorted(allowed_tools)
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(
                            tool_call.arguments, sort_keys=True
                        ),
                    },
                }
                for tool_call, _error in validation_failures
            ],
        }
    )
    for tool_call, validation_error in validation_failures:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    format_synthetic_validation_error(
                        validation_error,
                        allowed_tools=allowed_tool_names,
                        fingerprint=fingerprint,
                    )
                ),
            }
        )
    return messages


def _validate_tool_call_batch_structure(tool_calls: list[ToolCall]) -> bool:
    """Return whether every call in the batch is structurally well-formed."""
    seen_ids: set[str] = set()
    for tool_call in tool_calls:
        if not tool_call.id or not tool_call.name:
            return False
        if tool_call.id in seen_ids:
            return False
        seen_ids.add(tool_call.id)
        if not isinstance(tool_call.arguments, dict):
            return False
    return True


def _validate_tool_call_batch(
    tool_calls: list[ToolCall],
    allowed_tools: set[str],
) -> bool:
    """Return whether every call in the batch is well-formed and authorized."""
    if not _validate_tool_call_batch_structure(tool_calls):
        return False
    return all(tool_call.name in allowed_tools for tool_call in tool_calls)


def _validate_text_completion(result: ChatCompletionResult) -> str | None:
    """Return a user-facing error when the model output is unusable text."""
    content = (result.content or "").strip()
    if not content:
        return ERROR_EMPTY_RESPONSE

    return None


def _chat_log_to_messages(
    content: list[conversation.Content],
) -> list[dict[str, Any]]:
    """Convert Home Assistant chat log entries to llama.cpp messages."""
    messages: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, conversation.SystemContent):
            if item.content:
                messages.append({"role": "system", "content": item.content})
        elif isinstance(item, conversation.UserContent):
            messages.append({"role": "user", "content": item.content})
        elif isinstance(item, conversation.AssistantContent):
            message: dict[str, Any] = {
                "role": "assistant",
                "content": item.content or "",
            }
            if item.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.tool_args, sort_keys=True),
                        },
                    }
                    for tool_call in item.tool_calls
                ]
            messages.append(message)
        elif isinstance(item, conversation.ToolResultContent):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.tool_call_id,
                    "content": json.dumps(item.tool_result),
                }
            )
    return messages


def _error_result(
    user_input: conversation.ConversationInput,
    chat_log: conversation.ChatLog,
    speech: str,
) -> conversation.ConversationResult:
    """Build a short spoken error result."""
    intent_response = intent.IntentResponse(language=user_input.language)
    intent_response.async_set_error(
        intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
        speech,
    )
    return conversation.ConversationResult(
        response=intent_response,
        conversation_id=chat_log.conversation_id,
    )
