"""Tests for the SaySo Assist conversation entity."""

from __future__ import annotations

from typing import Any

import pytest
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import CONF_TOKEN, CONF_URL, DOMAIN
from custom_components.sayso.conversation import SaySoConversationEntity


@pytest.mark.asyncio
async def test_conversation_forwards_assist_text_and_returns_speech() -> None:
    calls: list[dict[str, Any]] = []

    async def request_json(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append({"url": url, "headers": headers, "payload": payload})
        return {
            "version": 1,
            "type": "text_response",
            "correlation_id": "turn-1",
            "payload": {
                "category": "completed",
                "response_mode": "earcon",
                "response_content": "\a",
            },
        }

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entity = SaySoConversationEntity(entry, request_json=request_json)
    user_input = ConversationInput(
        text="turn on the lamp",
        context=Context(),
        conversation_id="conversation-1",
        device_id=None,
        satellite_id="macbook",
        language="en",
        agent_id="sayso",
    )

    result = await entity._async_handle_message(user_input, None)  # noqa: SLF001

    assert calls == [
        {
            "url": "http://127.0.0.1:8765/api/v1/text",
            "headers": {"Authorization": "Bearer secret-token"},
            "payload": {
                "version": 1,
                "type": "text",
                "correlation_id": "conversation-1",
                "payload": {
                    "satellite_id": "macbook",
                    "text": "turn on the lamp",
                },
            },
        },
    ]
    assert result.conversation_id == "conversation-1"
    assert result.response.speech["plain"]["speech"] == "Done."


@pytest.mark.asyncio
async def test_clarification_keeps_assist_conversation_open() -> None:
    async def request_json(
        _url: str,
        _headers: dict[str, str],
        _payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "type": "text_response",
            "correlation_id": "turn-2",
            "payload": {
                "category": "no_action",
                "plan": {"outcome": "clarification"},
                "response_mode": "text",
                "response_content": "Which lamp do you mean?",
            },
        }

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entity = SaySoConversationEntity(entry, request_json=request_json)
    user_input = ConversationInput(
        text="turn on the lamp",
        context=Context(),
        conversation_id="conversation-2",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="sayso",
    )

    result = await entity._async_handle_message(user_input, None)  # noqa: SLF001

    assert result.continue_conversation is True
    assert result.response.speech["plain"]["speech"] == "Which lamp do you mean?"


@pytest.mark.asyncio
async def test_malformed_non_object_response_fails_closed() -> None:
    async def request_json(
        _url: str,
        _headers: dict[str, str],
        _payload: dict[str, Any],
    ) -> Any:
        return []

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entity = SaySoConversationEntity(entry, request_json=request_json)
    user_input = ConversationInput(
        text="turn on the lamp",
        context=Context(),
        conversation_id="conversation-3",
        device_id=None,
        satellite_id="macbook",
        language="en",
        agent_id="sayso",
    )

    result = await entity._async_handle_message(user_input, None)  # noqa: SLF001

    assert result.response.error_code is not None
    assert result.response.speech["plain"]["speech"] == (
        "SaySo could not complete that request."
    )
