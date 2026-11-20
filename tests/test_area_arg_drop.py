from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.client import ChatCompletionResult, ToolCall
from custom_components.sayso.conversation import SaySoConversationEntity, _Turn
from custom_components.sayso.schema import (
    CompiledToolSchema,
    compile_tools,
    schema_fingerprint,
)
from custom_components.sayso.tracing import TraceContext


class _HassTurnOn(llm.IntentTool):
    def __init__(self) -> None:
        self.name = "HassTurnOn"
        self.description = "Turn on"
        self.parameters = vol.Schema(
            {
                vol.Optional("name"): str,
                vol.Optional("area"): str,
                vol.Optional("domain"): vol.Any(str, [str]),
            }
        )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, str]:
        return {"success": "true"}


async def _setup_home(hass: HomeAssistant) -> str:
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    assert await async_setup_component(hass, "intent", {})

    living = ar.async_get(hass).async_create("Living Room")
    kitchen = ar.async_get(hass).async_create("Kitchen")

    config_entry = MockConfigEntry(domain="sayso", unique_id="area-drop")
    config_entry.add_to_hass(hass)
    satellite = dr.async_get(hass).async_get_or_create(
        config_entry_id=config_entry.entry_id,
        identifiers={("sayso", "satellite")},
    )
    dr.async_get(hass).async_update_device(satellite.id, area_id=kitchen.id)

    registry = er.async_get(hass)
    entry = registry.async_get_or_create("media_player", "test", "sofa_tv")
    registry.async_update_entity(entry.entity_id, area_id=living.id, name="Sofa TV")
    hass.states.async_set(
        entry.entity_id, "off", attributes={"friendly_name": "Sofa TV"}
    )
    async_expose_entity(hass, conversation.DOMAIN, entry.entity_id, True)

    return satellite.id


@pytest.fixture
async def area_drop(hass: HomeAssistant):
    satellite_id = await _setup_home(hass)
    turn_on = _HassTurnOn()
    schema = CompiledToolSchema(
        tools=compile_tools([turn_on]),
        fingerprint=schema_fingerprint(list(compile_tools([turn_on]))),
    )
    api = SimpleNamespace(
        tools=[turn_on],
        async_call_tool=AsyncMock(return_value={"success": True}),
    )
    agent = SaySoConversationEntity(
        MockConfigEntry(domain="sayso", entry_id="area-drop")
    )
    agent.hass = hass
    engine = AsyncMock()

    def make_turn(text: str, *, conversation_id: str = "area-drop-conv") -> _Turn:
        return _Turn(
            entry_id="area-drop",
            agent_id="conversation.sayso",
            runtime=SimpleNamespace(
                engine=engine,
                model="test",
                temperature=0,
                max_output_tokens=160,
                max_tool_iterations=1,
                llm_api=None,
                system_prompt="",
            ),
            user_input=conversation.ConversationInput(
                text=text,
                context=Context(),
                conversation_id=conversation_id,
                language="en",
                agent_id="conversation.sayso",
                device_id=satellite_id,
                satellite_id=None,
            ),
            chat_log=conversation.ChatLog(hass, conversation_id, llm_api=api),
            trace=TraceContext("area-drop"),
        )

    async def answer(
        turn: _Turn, tool_calls: list[ToolCall] | None = None
    ) -> conversation.ConversationResult:
        model = (
            ChatCompletionResult(content=None, tool_calls=tool_calls or [])
            if tool_calls
            else ChatCompletionResult(content="Hello.", tool_calls=[])
        )
        engine.reset_mock()
        engine.async_chat_completion = AsyncMock(return_value=model)
        with (
            patch.object(
                conversation.ChatLog, "async_provide_llm_data", new=AsyncMock()
            ),
            patch(
                "custom_components.sayso.conversation.compile_llm_tools",
                return_value=schema,
            ),
        ):
            return await agent._async_answer(turn)

    return SimpleNamespace(
        agent=agent, api=api, engine=engine, make_turn=make_turn, answer=answer
    )


async def test_satellite_area_dropped_from_named_device_call(area_drop) -> None:
    call = ToolCall(
        id="call", name="HassTurnOn", arguments={"name": "Sofa TV", "area": "Kitchen"}
    )
    turn = area_drop.make_turn("Mute the Sofa TV")

    result = await area_drop.answer(turn, [call])

    assert area_drop.api.async_call_tool.await_count == 1
    tool_args = area_drop.api.async_call_tool.await_args.args[0].tool_args
    assert tool_args == {"name": "Sofa TV"}
    assert result.response.speech["plain"]["speech"] == "Done."
