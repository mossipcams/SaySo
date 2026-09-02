"""Tests for the SaySo Assist conversation entity."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.const import CONF_TOKEN, CONF_URL, DOMAIN
from custom_components.sayso.conversation import SaySoConversationEntity
from custom_components.sayso.coordinator import SaySoConversationError


def _make_entity(
    *,
    coordinator: AsyncMock | None = None,
) -> SaySoConversationEntity:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    return SaySoConversationEntity(entry, coordinator=coordinator)


def _user_input(**overrides: Any) -> ConversationInput:
    defaults: dict[str, Any] = {
        "text": "turn on the lamp",
        "context": Context(),
        "conversation_id": "conversation-1",
        "device_id": None,
        "satellite_id": None,
        "language": "en",
        "agent_id": "sayso",
    }
    defaults.update(overrides)
    return ConversationInput(**defaults)


@pytest.mark.asyncio
async def test_conversation_uses_coordinator_and_never_posts_to_text_api() -> None:
    calls: list[dict[str, Any]] = []
    coordinator = AsyncMock()

    async def request_conversation(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"speech": "Done.", "response_type": "action_done"}

    coordinator.async_request_conversation = request_conversation
    entity = _make_entity(coordinator=coordinator)

    turn_context = Context(user_id="caller-1")
    result = await entity._async_handle_message(  # noqa: SLF001
        _user_input(context=turn_context),
        None,
    )

    assert calls == [
        {
            "transcript": "turn on the lamp",
            "device_id": None,
            "satellite_id": None,
            "context": turn_context,
        },
    ]
    conversation_source = Path(__file__).with_name("conversation.py").read_text()
    assert "/api/v1/text" not in conversation_source
    assert 'or "macbook"' not in conversation_source
    assert "or 'macbook'" not in conversation_source
    assert result.conversation_id == "conversation-1"
    assert result.response.speech["plain"]["speech"] == "Done."


@pytest.mark.asyncio
async def test_missing_satellite_and_device_ids_stay_none() -> None:
    calls: list[dict[str, Any]] = []
    coordinator = AsyncMock()

    async def request_conversation(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"speech": "Done.", "response_type": "action_done"}

    coordinator.async_request_conversation = request_conversation
    entity = _make_entity(coordinator=coordinator)

    user_input = _user_input(device_id=None, satellite_id=None)
    await entity._async_handle_message(user_input, None)  # noqa: SLF001

    assert len(calls) == 1
    assert calls[0]["device_id"] is None
    assert calls[0]["satellite_id"] is None
    assert "macbook" not in calls[0].values()


@pytest.mark.asyncio
async def test_clarification_keeps_assist_conversation_open() -> None:
    coordinator = AsyncMock()
    coordinator.async_request_conversation.return_value = {
        "speech": "Which lamp do you mean?",
        "response_type": "clarification",
    }
    entity = _make_entity(coordinator=coordinator)

    user_input = _user_input(
        conversation_id="conversation-2",
        satellite_id=None,
    )
    result = await entity._async_handle_message(  # noqa: SLF001
        user_input,
        None,
    )

    assert result.continue_conversation is True
    assert result.response.speech["plain"]["speech"] == "Which lamp do you mean?"
    coordinator.async_request_conversation.assert_awaited_once_with(
        transcript="turn on the lamp",
        device_id=None,
        satellite_id=None,
        context=user_input.context,
    )


@pytest.mark.asyncio
async def test_conversation_passes_device_and_satellite_ids_to_coordinator(
    hass: HomeAssistant,
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_URL: "http://127.0.0.1:8765", CONF_TOKEN: "secret-token"},
    )
    entry.add_to_hass(hass)

    area_reg = ar.async_get(hass)
    living_room = area_reg.async_create("Living Room")
    device_reg = dr.async_get(hass)
    device = device_reg.async_get_or_create(
        config_entry_id=entry.entry_id,
        identifiers={("test", "living_room_satellite")},
    )
    device_reg.async_update_device(device.id, area_id=living_room.id)

    calls: list[dict[str, Any]] = []
    coordinator = AsyncMock()

    async def request_conversation(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"speech": "Done.", "response_type": "action_done"}

    coordinator.async_request_conversation = request_conversation
    entity = _make_entity(coordinator=coordinator)

    turn_context = Context(user_id="caller-1")
    result = await entity._async_handle_message(  # noqa: SLF001
        _user_input(
            context=turn_context,
            device_id=device.id,
            satellite_id=device.id,
        ),
        None,
    )

    assert calls == [
        {
            "transcript": "turn on the lamp",
            "device_id": device.id,
            "satellite_id": device.id,
            "context": turn_context,
        },
    ]
    assert result.response.speech["plain"]["speech"] == "Done."


@pytest.mark.asyncio
async def test_malformed_non_object_response_fails_closed() -> None:
    coordinator = AsyncMock()
    coordinator.async_request_conversation.return_value = []
    entity = _make_entity(coordinator=coordinator)

    result = await entity._async_handle_message(  # noqa: SLF001
        _user_input(conversation_id="conversation-3"),
        None,
    )

    assert result.response.error_code is not None
    assert result.response.speech["plain"]["speech"] == (
        "SaySo could not complete that request."
    )


@pytest.mark.asyncio
async def test_empty_speech_response_fails_closed() -> None:
    coordinator = AsyncMock()
    coordinator.async_request_conversation.return_value = {
        "speech": "",
        "response_type": "no_action",
    }
    entity = _make_entity(coordinator=coordinator)

    result = await entity._async_handle_message(_user_input(), None)  # noqa: SLF001

    assert result.response.error_code is not None
    assert result.response.speech["plain"]["speech"] == (
        "SaySo could not complete that request."
    )


@pytest.mark.asyncio
async def test_disconnected_coordinator_fails_closed() -> None:
    coordinator = AsyncMock()
    coordinator.async_request_conversation.side_effect = SaySoConversationError(
        "SaySo WebSocket is not connected",
    )
    entity = _make_entity(coordinator=coordinator)

    result = await entity._async_handle_message(_user_input(), None)  # noqa: SLF001

    assert result.response.error_code is not None
    assert result.response.speech["plain"]["speech"] == (
        "SaySo could not complete that request."
    )


@pytest.mark.asyncio
async def test_timed_out_conversation_request_fails_closed() -> None:
    coordinator = AsyncMock()
    coordinator.async_request_conversation.side_effect = asyncio.TimeoutError(
        "conversation request timed out",
    )
    entity = _make_entity(coordinator=coordinator)

    result = await entity._async_handle_message(_user_input(), None)  # noqa: SLF001

    assert result.response.error_code is not None
    assert result.response.speech["plain"]["speech"] == (
        "SaySo could not complete that request."
    )


@pytest.mark.asyncio
async def test_async_prepare_succeeds_when_coordinator_reports_ready() -> None:
    coordinator = AsyncMock()
    coordinator.async_request_prepare.return_value = {
        "connected": True,
        "graph_ready": True,
        "model_ready": True,
    }
    entity = _make_entity(coordinator=coordinator)

    await entity.async_prepare("en")

    coordinator.async_request_prepare.assert_awaited_once_with()
    coordinator.async_request_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_async_prepare_rejects_unsupported_language() -> None:
    coordinator = AsyncMock()
    entity = _make_entity(coordinator=coordinator)

    with pytest.raises(ValueError, match="not supported"):
        await entity.async_prepare("fr")

    coordinator.async_request_prepare.assert_not_called()


@pytest.mark.asyncio
async def test_async_prepare_raises_when_coordinator_not_connected() -> None:
    coordinator = AsyncMock()
    coordinator.async_request_prepare.side_effect = SaySoConversationError(
        "SaySo WebSocket is not connected",
    )
    entity = _make_entity(coordinator=coordinator)

    with pytest.raises(SaySoConversationError, match="not connected"):
        await entity.async_prepare("en")

    coordinator.async_request_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_async_prepare_raises_on_timeout_without_conversation() -> None:
    coordinator = AsyncMock()
    coordinator.async_request_prepare.side_effect = asyncio.TimeoutError(
        "prepare request timed out",
    )
    entity = _make_entity(coordinator=coordinator)

    with pytest.raises(asyncio.TimeoutError):
        await entity.async_prepare("en")

    coordinator.async_request_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_async_prepare_raises_when_server_not_ready() -> None:
    coordinator = AsyncMock()
    coordinator.async_request_prepare.side_effect = SaySoConversationError(
        "SaySo is not ready",
    )
    entity = _make_entity(coordinator=coordinator)

    with pytest.raises(SaySoConversationError, match="not ready"):
        await entity.async_prepare("en")

    coordinator.async_request_conversation.assert_not_called()
