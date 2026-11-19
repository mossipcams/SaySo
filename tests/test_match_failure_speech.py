from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component
from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.client import ToolCall
from custom_components.sayso.const import ERROR_ACTION_FAILED
from custom_components.sayso.conversation import (
    SaySoConversationEntity,
    _Turn,
    execution_failure_speech,
    record_boundary,
)
from custom_components.sayso.diagnostics import BoundaryFailureCode, BoundaryPhase
from custom_components.sayso.tracing import ErrorType, Stage, TraceContext


async def _setup(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {"conversation": {}})


def _calls(args_list: list[dict]) -> list[tuple[ToolCall, dict]]:
    return [
        (ToolCall(id=f"call_{index}", name="HassTurnOn", arguments=args), args)
        for index, args in enumerate(args_list)
    ]


async def test_area_without_exposed_domain_says_absence(hass: HomeAssistant) -> None:
    await _setup(hass)
    ar.async_get(hass).async_create("Bedroom")

    speech = execution_failure_speech(
        hass,
        conversation.DOMAIN,
        _calls([{"area": "Bedroom", "domain": "light"}]),
        "MatchFailedError",
    )

    assert speech == "The bedroom has no lights available."


async def test_area_with_exposed_domain_keeps_stock(hass: HomeAssistant) -> None:
    await _setup(hass)
    area = ar.async_get(hass).async_create("Bedroom")
    registry = er.async_get(hass)
    entry = registry.async_get_or_create(
        "light", "test", "bed_light", suggested_object_id="bedroom_light"
    )
    registry.async_update_entity(entry.entity_id, area_id=area.id)
    async_expose_entity(hass, conversation.DOMAIN, entry.entity_id, True)
    hass.states.async_set(entry.entity_id, "off")

    speech = execution_failure_speech(
        hass,
        conversation.DOMAIN,
        _calls([{"area": "Bedroom", "domain": "light"}]),
        "MatchFailedError",
    )

    assert speech == ERROR_ACTION_FAILED


async def test_area_with_unexposed_domain_says_absence(hass: HomeAssistant) -> None:
    await _setup(hass)
    area = ar.async_get(hass).async_create("Bedroom")
    registry = er.async_get(hass)
    entry = registry.async_get_or_create(
        "light", "test", "bed_light", suggested_object_id="bedroom_light"
    )
    registry.async_update_entity(entry.entity_id, area_id=area.id)
    async_expose_entity(hass, conversation.DOMAIN, entry.entity_id, False)
    hass.states.async_set(entry.entity_id, "off")

    speech = execution_failure_speech(
        hass,
        conversation.DOMAIN,
        _calls([{"area": "Bedroom", "domain": "light"}]),
        "MatchFailedError",
    )

    assert speech == "The bedroom has no lights available."


async def test_name_not_found_keeps_stock(hass: HomeAssistant) -> None:
    await _setup(hass)

    speech = execution_failure_speech(
        hass, conversation.DOMAIN, _calls([{"name": "Hall Lamp"}]), "MatchFailedError"
    )

    assert speech == ERROR_ACTION_FAILED


async def test_non_match_failure_keeps_stock(hass: HomeAssistant) -> None:
    await _setup(hass)

    speech = execution_failure_speech(
        hass,
        conversation.DOMAIN,
        _calls([{"area": "Bedroom", "domain": "light"}]),
        "ServiceNotFound",
    )

    assert speech == ERROR_ACTION_FAILED


async def test_multiple_calls_keep_stock(hass: HomeAssistant) -> None:
    await _setup(hass)
    ar.async_get(hass).async_create("Bedroom")

    speech = execution_failure_speech(
        hass,
        conversation.DOMAIN,
        _calls(
            [
                {"area": "Bedroom", "domain": "light"},
                {"area": "Kitchen", "domain": "fan"},
            ]
        ),
        "MatchFailedError",
    )

    assert speech == ERROR_ACTION_FAILED


async def test_unknown_area_keeps_stock(hass: HomeAssistant) -> None:
    await _setup(hass)

    speech = execution_failure_speech(
        hass,
        conversation.DOMAIN,
        _calls([{"area": "Attic", "domain": "light"}]),
        "MatchFailedError",
    )

    assert speech == ERROR_ACTION_FAILED


async def test_device_area_entity_counts_toward_area(hass: HomeAssistant) -> None:
    await _setup(hass)
    area = ar.async_get(hass).async_create("Bedroom")
    registry = er.async_get(hass)
    config_entry = MockConfigEntry(domain="sayso", unique_id="bed-lamp")
    config_entry.add_to_hass(hass)
    device = dr.async_get(hass).async_get_or_create(
        config_entry_id=config_entry.entry_id,
        identifiers={("bed_lamp", "bed-lamp-device")},
    )
    dr.async_get(hass).async_update_device(device.id, area_id=area.id)
    entry = registry.async_get_or_create(
        "fan", "test", "bed_fan", suggested_object_id="bedroom_fan"
    )
    registry.async_update_entity(entry.entity_id, device_id=device.id)
    async_expose_entity(hass, conversation.DOMAIN, entry.entity_id, True)
    hass.states.async_set(entry.entity_id, "off")

    speech = execution_failure_speech(
        hass,
        conversation.DOMAIN,
        _calls([{"area": "Bedroom", "domain": "fan"}]),
        "MatchFailedError",
    )

    assert speech == ERROR_ACTION_FAILED


@pytest.mark.parametrize(
    ("args", "error", "expected"),
    [
        (
            {"area": "Bedroom", "domain": "light"},
            "MatchFailedError",
            "The bedroom has no lights available.",
        ),
        (
            {"name": "Hall Lamp"},
            "MatchFailedError",
            ERROR_ACTION_FAILED,
        ),
        (
            {"name": "Hall Lamp", "area": "Bedroom", "domain": "light"},
            "MatchFailedError",
            ERROR_ACTION_FAILED,
        ),
        (
            {"area": "Bedroom", "domain": "light"},
            "IntentHandleError",
            ERROR_ACTION_FAILED,
        ),
    ],
)
async def test_execution_failure_path(
    hass: HomeAssistant, args, error, expected
) -> None:
    await _setup(hass)
    ar.async_get(hass).async_create("Bedroom")
    tool = SimpleNamespace(
        name="HassTurnOn",
        parameters=vol.Schema(
            {
                vol.Optional("area"): str,
                vol.Optional("domain"): str,
                vol.Optional("name"): str,
            }
        ),
    )
    tool_result = {
        "error": error,
        "error_text": "No entities matched the domain (given area 'Bedroom', domains light)",
    }
    api = SimpleNamespace(
        tools=[tool], async_call_tool=AsyncMock(return_value=tool_result)
    )
    chat_log = conversation.ChatLog(hass, "match-failure", llm_api=api)
    trace = TraceContext("match-failure")
    turn = _Turn(
        entry_id="test-entry",
        agent_id="conversation.sayso",
        runtime=SimpleNamespace(),
        user_input=conversation.ConversationInput(
            text="Turn on the bedroom lights",
            context=Context(),
            conversation_id=None,
            language="en",
            agent_id="conversation.sayso",
            device_id=None,
            satellite_id=None,
        ),
        chat_log=chat_log,
        trace=trace,
    )
    agent = SaySoConversationEntity(MockConfigEntry(domain="sayso"))
    agent.hass = hass

    with patch(
        "custom_components.sayso.conversation.record_boundary", wraps=record_boundary
    ) as recorded:
        result = await agent._async_run_tool_calls(turn, [_calls([args])[0][0]])

    api.async_call_tool.assert_awaited_once()
    assert api.async_call_tool.await_args.args[0].tool_args == args
    assert chat_log.content[-1].tool_result == tool_result
    assert result.response.speech["plain"]["speech"] == expected
    assert result.response.response_type == intent.IntentResponseType.ERROR
    recorded.assert_called_once_with(
        "test-entry",
        BoundaryFailureCode.TOOL_EXECUTION_FAILED,
        BoundaryPhase.EXECUTION,
        None,
        ha_error=error,
        trace=trace,
    )
    assert trace.error_type == ErrorType.HA_ACTION_FAILED
    assert trace.error_stage == Stage.HA_ACTION
