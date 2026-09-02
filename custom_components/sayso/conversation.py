"""SaySo Home Assistant Assist conversation entity."""

from __future__ import annotations

from homeassistant.components.conversation import ConversationEntity
from homeassistant.components.conversation.models import (
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers import intent
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.core import HomeAssistant

from .const import DEVICE_NAME, DOMAIN
from .coordinator import SaySoConnectionCoordinator, SaySoConversationError


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the SaySo conversation agent."""

    coordinator: SaySoConnectionCoordinator = hass.data[DOMAIN][entry.entry_id][
        "coordinator"
    ]
    async_add_entities([SaySoConversationEntity(entry, coordinator=coordinator)])


class SaySoConversationEntity(ConversationEntity):
    """Route Assist conversation turns through the SaySo model service."""

    _attr_has_entity_name = True
    _attr_name = "Conversation"

    def __init__(
        self,
        entry: ConfigEntry,
        *,
        coordinator: SaySoConnectionCoordinator | None = None,
    ) -> None:
        self._entry = entry
        self._coordinator = coordinator
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

    def _get_coordinator(self) -> SaySoConnectionCoordinator:
        if self._coordinator is not None:
            return self._coordinator
        return self.hass.data[DOMAIN][self._entry.entry_id]["coordinator"]

    async def async_prepare(self, language: str | None = None) -> None:
        """Verify resident SaySo runtime readiness before transcript handling."""

        if language is not None and language not in self.supported_languages:
            msg = f"Language {language} is not supported"
            raise ValueError(msg)

        await self._get_coordinator().async_request_prepare()

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        _chat_log: ChatLog,
    ) -> ConversationResult:
        """Send Assist text to SaySo and return spoken response content."""

        response = intent.IntentResponse(language=user_input.language)
        try:
            result_payload = await self._get_coordinator().async_request_conversation(
                transcript=user_input.text,
                device_id=user_input.device_id,
                satellite_id=user_input.satellite_id,
                context=user_input.context,
            )
            if not isinstance(result_payload, dict):
                raise ValueError("invalid SaySo response payload")
            content = result_payload.get("speech")
            if not isinstance(content, str) or not content:
                raise ValueError("SaySo response has no speech content")
        except (
            SaySoConversationError,
            TimeoutError,
            KeyError,
            TypeError,
            ValueError,
        ):
            response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "SaySo could not complete that request.",
            )
            return ConversationResult(
                response=response,
                conversation_id=user_input.conversation_id,
            )

        response.async_set_speech(content)
        return ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
            continue_conversation=(
                result_payload.get("response_type") == "clarification"
            ),
        )
