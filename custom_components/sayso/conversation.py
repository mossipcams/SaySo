"""SaySo Home Assistant Assist conversation entity."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import uuid
from typing import Any

from aiohttp import ClientError
from homeassistant.components.conversation import ConversationEntity
from homeassistant.components.conversation.models import (
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.core import HomeAssistant

from .const import API_VERSION, CONF_TOKEN, CONF_URL, DEVICE_NAME, DOMAIN

JsonRequest = Callable[
    [str, dict[str, str], dict[str, Any]],
    Awaitable[dict[str, Any]],
]


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the SaySo conversation agent."""

    async_add_entities([SaySoConversationEntity(entry)])


class SaySoConversationEntity(ConversationEntity):
    """Route Assist conversation turns through the SaySo model service."""

    _attr_has_entity_name = True
    _attr_name = "Conversation"

    def __init__(
        self,
        entry: ConfigEntry,
        *,
        request_json: JsonRequest | None = None,
    ) -> None:
        self._entry = entry
        self._request_json = request_json or self._async_request_json
        self._attr_unique_id = f"{entry.entry_id}-conversation"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=DEVICE_NAME,
            manufacturer="SaySo",
            model="Voice Assistant",
        )

    @property
    def supported_languages(self) -> list[str]:
        """Return languages accepted by the tuned model."""

        return ["en"]

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        _chat_log: ChatLog,
    ) -> ConversationResult:
        """Send Assist text to SaySo and return spoken response content."""

        correlation_id = user_input.conversation_id or uuid.uuid4().hex
        payload = {
            "version": API_VERSION,
            "type": "text",
            "correlation_id": correlation_id,
            "payload": {
                "satellite_id": user_input.satellite_id or "macbook",
                "text": user_input.text,
            },
        }
        response = intent.IntentResponse(language=user_input.language)
        try:
            result = await self._request_json(
                f"{self._entry.data[CONF_URL]}/api/v1/text",
                {"Authorization": f"Bearer {self._entry.data[CONF_TOKEN]}"},
                payload,
            )
            if not isinstance(result, dict) or result.get("type") != "text_response":
                raise ValueError("invalid SaySo response type")
            result_payload = result["payload"]
            if not isinstance(result_payload, dict):
                raise ValueError("invalid SaySo response payload")
            content = result_payload["response_content"]
            if not isinstance(content, str) or not content:
                raise ValueError("SaySo response has no speech content")
        except (ClientError, TimeoutError, KeyError, TypeError, ValueError):
            response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "SaySo could not complete that request.",
            )
            return ConversationResult(
                response=response,
                conversation_id=user_input.conversation_id,
            )

        response.async_set_speech("Done." if content == "\a" else content)
        return ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
            continue_conversation=(
                result_payload.get("plan", {}).get("outcome") == "clarification"
                if isinstance(result_payload.get("plan", {}), dict)
                else False
            ),
        )

    async def _async_request_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST one Assist turn to the configured SaySo service."""

        session = async_get_clientsession(self.hass)
        async with session.post(url, headers=headers, json=payload) as response:
            data = await response.json(content_type=None)
            if response.status >= 400:
                raise ValueError("SaySo request failed")
            if not isinstance(data, dict) or data.get("type") != "text_response":
                raise ValueError("invalid SaySo response")
            return data
