"""Conversation platform for SaySo."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, override

from homeassistant.components import conversation
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, intent
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import SaySoConfigEntry, SaySoRuntimeData
from .client import ChatCompletionResult
from .const import (
    DOMAIN,
    ERROR_EMPTY_RESPONSE,
    ERROR_MODEL_UNAVAILABLE,
    ERROR_REQUEST_TIMEOUT,
    ERROR_TOOL_CALLS_UNSUPPORTED,
)
from .exceptions import (
    SaySoConnectionError,
    SaySoError,
    SaySoHttpError,
    SaySoInvalidResponseError,
    SaySoTimeoutError,
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

        messages = _chat_log_to_messages(chat_log.content)
        try:
            result = await runtime.client.chat_completion(
                messages,
                model=runtime.model,
                temperature=runtime.temperature,
                max_tokens=runtime.max_output_tokens,
            )
        except SaySoTimeoutError:
            return _error_result(user_input, chat_log, ERROR_REQUEST_TIMEOUT)
        except SaySoConnectionError:
            return _error_result(user_input, chat_log, ERROR_MODEL_UNAVAILABLE)
        except (SaySoHttpError, SaySoInvalidResponseError) as err:
            _LOGGER.debug("llama.cpp response error: %s", err)
            return _error_result(user_input, chat_log, ERROR_MODEL_UNAVAILABLE)
        except SaySoError as err:
            _LOGGER.debug("SaySo error: %s", err)
            return _error_result(user_input, chat_log, ERROR_MODEL_UNAVAILABLE)

        error_message = _validate_completion(result)
        if error_message is not None:
            return _error_result(user_input, chat_log, error_message)

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=result.content,
            )
        )
        return conversation.async_get_result_from_chat_log(user_input, chat_log)


def _validate_completion(result: ChatCompletionResult) -> str | None:
    """Return a user-facing error when the model output is unusable."""
    if result.tool_calls:
        return ERROR_TOOL_CALLS_UNSUPPORTED

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
            messages.append({"role": "assistant", "content": item.content or ""})
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
