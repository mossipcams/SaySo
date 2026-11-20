from __future__ import annotations

import json
from typing import Any, override

from homeassistant.components.ai_task import (
    AITaskEntity,
    AITaskEntityFeature,
    GenDataTask,
    GenDataTaskResult,
)
from homeassistant.components.conversation import ChatLog
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import SaySoConfigEntry
from .const import (
    AI_TASK_SUBENTRY_TYPE,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
)
from .transcript import chat_log_to_messages

STRUCTURE_INSTRUCTION = (
    "Respond with only one JSON object that satisfies the requested structure. "
    "Do not add prose or code fences."
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: SaySoConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    for subentry in entry.subentries.values():
        if subentry.subentry_type == AI_TASK_SUBENTRY_TYPE:
            async_add_entities(
                [SaySoAITaskEntity(entry, subentry)],
                config_subentry_id=subentry.subentry_id,
            )


class SaySoAITaskEntity(AITaskEntity):
    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = AITaskEntityFeature.GENERATE_DATA

    def __init__(self, entry: SaySoConfigEntry, subentry: ConfigSubentry) -> None:
        self._entry = entry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="SaySo",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @override
    async def _async_generate_data(
        self, task: GenDataTask, chat_log: ChatLog
    ) -> GenDataTaskResult:
        if task.attachments:
            raise HomeAssistantError(
                "AI Task attachments are not supported",
                translation_domain=DOMAIN,
                translation_key="ai_task_attachments",
            )
        messages: list[dict[str, Any]] = chat_log_to_messages(chat_log.content)
        if task.structure is not None:
            messages.append({"role": "system", "content": STRUCTURE_INSTRUCTION})
        result = await self._entry.runtime_data.engine.async_chat_completion(
            messages,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )
        content = result.content or ""
        if task.structure is None:
            data: Any = content
        else:
            data = _validated_structure(task.structure, content)
        return GenDataTaskResult(conversation_id=chat_log.conversation_id, data=data)


def _invalid_structure(reason: str) -> HomeAssistantError:
    return HomeAssistantError(
        f"AI Task response does not match the requested structure: {reason}",
        translation_domain=DOMAIN,
        translation_key="ai_task_invalid",
        translation_placeholders={"reason": reason},
    )


def _validated_structure(structure: Any, content: str) -> Any:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as err:
        raise _invalid_structure("the response is not valid JSON") from err
    try:
        return structure(parsed)
    except Exception as err:
        raise _invalid_structure(str(err)) from err
