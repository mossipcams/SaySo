from __future__ import annotations

import time
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
from homeassistant.helpers import intent, llm
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.client import ChatCompletionResult, ToolCall
from custom_components.sayso.const import DOMAIN, ERROR_CANCELLED
from custom_components.sayso.conversation import (
    SaySoConversationEntity,
    _Turn,
    _grounding_home,
    _Held,
)
from custom_components.sayso.core.grounding import clarify
from custom_components.sayso.schema import CompiledToolSchema, compile_tools, schema_fingerprint
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


def _speech(result: conversation.ConversationResult) -> str:
    return result.response.speech["plain"]["speech"]


async def _setup_home(hass: HomeAssistant) -> tuple[str, str]:
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    assert await async_setup_component(hass, "intent", {})

    living = ar.async_get(hass).async_create("Living Room")
    ar.async_get(hass).async_create("Bedroom")
    ar.async_get(hass).async_create("Kitchen")

    config_entry = MockConfigEntry(domain="sayso", unique_id="grounding-flow")
    config_entry.add_to_hass(hass)
    satellite = dr.async_get(hass).async_get_or_create(
        config_entry_id=config_entry.entry_id,
        identifiers={("sayso", "satellite")},
    )
    dr.async_get(hass).async_update_device(satellite.id, area_id=living.id)

    registry = er.async_get(hass)
    bedroom = ar.async_get(hass).async_get_area_by_name("Bedroom")
    for domain, unique, name, area_id in (
        ("switch", "corner_lamp", "Corner lamp", living.id),
        ("switch", "shelf_lamp", "Shelf lamp", living.id),
        ("light", "nightstand", "Nightstand", bedroom.id),
    ):
        entry = registry.async_get_or_create(domain, "test", unique)
        registry.async_update_entity(entry.entity_id, area_id=area_id, name=name)
        hass.states.async_set(entry.entity_id, "off", attributes={"friendly_name": name})
        async_expose_entity(hass, conversation.DOMAIN, entry.entity_id, True)

    return satellite.id, living.id


@pytest.fixture
async def grounding_flow(hass: HomeAssistant):
    satellite_id, _living_id = await _setup_home(hass)
    turn_on = _HassTurnOn()
    schema = CompiledToolSchema(
        tools=compile_tools([turn_on]),
        fingerprint=schema_fingerprint(list(compile_tools([turn_on]))),
    )
    api = SimpleNamespace(
        tools=[turn_on],
        async_call_tool=AsyncMock(return_value={"success": True}),
    )
    agent = SaySoConversationEntity(MockConfigEntry(domain="sayso", entry_id="grounding-flow"))
    agent.hass = hass
    engine = AsyncMock()

    def make_turn(
        text: str,
        *,
        conversation_id: str = "grounding-conv",
        voice: bool = True,
    ) -> _Turn:
        return _Turn(
            entry_id="grounding-flow",
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
                device_id=satellite_id if voice else None,
                satellite_id=None,
            ),
            chat_log=conversation.ChatLog(hass, conversation_id, llm_api=api),
            trace=TraceContext("grounding-flow"),
        )

    async def answer(
        turn: _Turn,
        tool_calls: list[ToolCall] | None = None,
        *,
        model: ChatCompletionResult | None = None,
    ):
        if model is None:
            model = (
                ChatCompletionResult(content=None, tool_calls=tool_calls or [])
                if tool_calls
                else ChatCompletionResult(content="Hello.", tool_calls=[])
            )
        engine.reset_mock()
        engine.async_chat_completion = AsyncMock(return_value=model)
        with (
            patch.object(conversation.ChatLog, "async_provide_llm_data", new=AsyncMock()),
            patch(
                "custom_components.sayso.conversation.compile_llm_tools",
                return_value=schema,
            ),
        ):
            return await agent._async_answer(turn)

    return SimpleNamespace(
        agent=agent,
        api=api,
        engine=engine,
        make_turn=make_turn,
        answer=answer,
        satellite_id=satellite_id,
    )


def _clarify_speech(flow, text: str, tool_call: ToolCall, *, voice: bool = True) -> str:
    turn = flow.make_turn(text, voice=voice)
    home = _grounding_home(
        turn.chat_log.hass,
        turn.user_input,
        turn.user_input.as_llm_context(DOMAIN).assistant,
    )
    return clarify(tool_call.name, tool_call.arguments, text, home).speech


async def test_voice_grounded_executes(grounding_flow) -> None:
    call = ToolCall(id="call", name="HassTurnOn", arguments={"name": "Corner lamp"})
    turn = grounding_flow.make_turn("turn on the corner lamp")
    result = await grounding_flow.answer(turn, [call])

    grounding_flow.api.async_call_tool.assert_awaited_once()
    assert grounding_flow.api.async_call_tool.await_args.args[0].tool_args["name"] == (
        "Corner lamp"
    )
    assert _speech(result) == "Done."
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE


async def test_voice_ungrounded_clarifies(grounding_flow) -> None:
    call = ToolCall(id="call", name="HassTurnOn", arguments={"name": "Shelf lamp"})
    text = "turn on the cornet lamp"
    turn = grounding_flow.make_turn(text)
    expected = _clarify_speech(grounding_flow, text, call)
    assert expected == "Did you mean the Corner lamp?"

    result = await grounding_flow.answer(turn, [call])

    grounding_flow.api.async_call_tool.assert_not_awaited()
    assert _speech(result) == expected
    assert result.continue_conversation is True
    assert "grounding-conv" in grounding_flow.agent._held


async def test_follow_up_yes_executes(grounding_flow) -> None:
    call = ToolCall(id="call", name="HassTurnOn", arguments={"name": "Shelf lamp"})
    first = grounding_flow.make_turn("turn on the cornet lamp")
    await grounding_flow.answer(first, [call])
    grounding_flow.api.async_call_tool.reset_mock()

    second = grounding_flow.make_turn("yes")
    result = await grounding_flow.answer(second, [])

    grounding_flow.engine.async_chat_completion.assert_not_awaited()
    grounding_flow.api.async_call_tool.assert_awaited_once()
    assert grounding_flow.api.async_call_tool.await_args.args[0].tool_args["name"] == (
        "Corner lamp"
    )
    assert _speech(result) == "Done."


async def test_follow_up_no_then_yes_does_not_replay(grounding_flow) -> None:
    call = ToolCall(id="call", name="HassTurnOn", arguments={"name": "Shelf lamp"})
    first = grounding_flow.make_turn("turn on the cornet lamp")
    await grounding_flow.answer(first, [call])

    second = grounding_flow.make_turn("no")
    cancel = await grounding_flow.answer(second, [])
    grounding_flow.api.async_call_tool.assert_not_awaited()
    assert _speech(cancel) == ERROR_CANCELLED

    third = grounding_flow.make_turn("yes")
    replay = await grounding_flow.answer(
        third,
        model=ChatCompletionResult(content="Hello.", tool_calls=[]),
    )
    grounding_flow.api.async_call_tool.assert_not_awaited()
    assert grounding_flow.engine.async_chat_completion.await_count == 1
    assert _speech(replay) == "Hello."


async def test_two_option_clarification_named_choice_executes(grounding_flow) -> None:
    text = "turn on the corner or shelf lamp"
    call = ToolCall(id="call", name="HassTurnOn", arguments={"name": "Lamp"})
    first = grounding_flow.make_turn(text)
    clarify_result = await grounding_flow.answer(first, [call])
    assert " or the " in _speech(clarify_result)

    second = grounding_flow.make_turn("shelf lamp")
    result = await grounding_flow.answer(second, [])

    grounding_flow.api.async_call_tool.assert_awaited_once()
    assert grounding_flow.api.async_call_tool.await_args.args[0].tool_args["name"] == (
        "Shelf lamp"
    )
    assert _speech(result) == "Done."


async def test_unrelated_follow_up_clears_held(grounding_flow) -> None:
    call = ToolCall(id="call", name="HassTurnOn", arguments={"name": "Shelf lamp"})
    first = grounding_flow.make_turn("turn on the cornet lamp")
    await grounding_flow.answer(first, [call])

    second = grounding_flow.make_turn("what is the weather")
    result = await grounding_flow.answer(
        second,
        model=ChatCompletionResult(content="Sunny.", tool_calls=[]),
    )

    grounding_flow.api.async_call_tool.assert_not_awaited()
    assert grounding_flow.engine.async_chat_completion.await_count == 1
    assert _speech(result) == "Sunny."
    assert "grounding-conv" not in grounding_flow.agent._held


async def test_expired_held_ignored(grounding_flow, monkeypatch) -> None:
    call = ToolCall(id="call", name="HassTurnOn", arguments={"name": "Shelf lamp"})
    first = grounding_flow.make_turn("turn on the cornet lamp")
    await grounding_flow.answer(first, [call])

    anchor = 1_000.0
    held = grounding_flow.agent._held["grounding-conv"]
    monkeypatch.setattr(time, "monotonic", lambda: anchor + 301)
    grounding_flow.agent._held["grounding-conv"] = _Held(
        held.calls, held.index, held.options, anchor
    )

    second = grounding_flow.make_turn("yes")
    result = await grounding_flow.answer(
        second,
        model=ChatCompletionResult(content="Okay then.", tool_calls=[]),
    )

    grounding_flow.api.async_call_tool.assert_not_awaited()
    assert grounding_flow.engine.async_chat_completion.await_count == 1
    assert _speech(result) == "Okay then."


async def test_non_voice_executes_without_clarification(grounding_flow) -> None:
    call = ToolCall(id="call", name="HassTurnOn", arguments={"name": "Shelf lamp"})
    turn = grounding_flow.make_turn("turn on the cornet lamp", voice=False)
    result = await grounding_flow.answer(turn, [call])

    grounding_flow.api.async_call_tool.assert_awaited_once()
    assert _speech(result) == "Done."
    assert "grounding-conv" not in grounding_flow.agent._held