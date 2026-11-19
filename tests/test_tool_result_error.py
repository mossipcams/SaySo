from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.helpers import llm
from homeassistant.core import Context, HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.boundary import tool_result_failed
from custom_components.sayso.client import ToolCall
from custom_components.sayso.conversation import SaySoConversationEntity, _Turn
from custom_components.sayso.tracing import ErrorType, Stage, TraceContext


class _NewToolResult:
    def __init__(self, data: dict, error: bool) -> None:
        self.data = data
        self.error = error


class _NewShapeContent:
    def __init__(self, data: dict, error: bool) -> None:
        self.result = _NewToolResult(data=data, error=error)


class _OldShapeResult:
    def __init__(self, tool_result: dict) -> None:
        self.tool_result = tool_result


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
    ) -> dict[str, Any]:
        return {"success": "true"}


def test_new_shape_flagged_error_is_failed() -> None:
    assert tool_result_failed(_NewShapeContent({"success": True}, True)) is True


def test_new_shape_unflagged_success_is_not_failed() -> None:
    assert tool_result_failed(_NewShapeContent({"success": True}, False)) is False


def test_old_shape_error_dict_is_failed() -> None:
    assert tool_result_failed(_OldShapeResult({"error": "ServiceNotFound"})) is True


def test_old_shape_success_dict_is_not_failed() -> None:
    assert tool_result_failed(_OldShapeResult({"success": True})) is False


@pytest.fixture
async def error_flag_turn(hass: HomeAssistant):
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    tool = _HassTurnOn()
    api = SimpleNamespace(tools=[tool], async_call_tool=AsyncMock())
    turn = _Turn(
        entry_id="tool-result-error",
        agent_id="conversation.sayso",
        runtime=SimpleNamespace(
            engine=SimpleNamespace(async_chat_completion=AsyncMock()),
            model="test",
            temperature=0,
            max_output_tokens=160,
            max_tool_iterations=1,
            llm_api=None,
            system_prompt="",
        ),
        user_input=conversation.ConversationInput(
            text="Turn on the Hall LAMP",
            context=Context(),
            conversation_id=None,
            language="en",
            agent_id="conversation.sayso",
            device_id=None,
            satellite_id=None,
        ),
        chat_log=conversation.ChatLog(hass, "tool-result-error", llm_api=api),
        trace=TraceContext("tool-result-error"),
    )
    agent = SaySoConversationEntity(MockConfigEntry(domain="sayso"))
    agent.hass = hass
    return agent, turn


async def test_flagged_error_is_spoken_as_failure(error_flag_turn):
    agent, turn = error_flag_turn

    async def fake_add_assistant_content(self, content, tool_call_tasks=None):
        yield _NewShapeContent({"success": True}, True)

    with (
        patch.object(
            conversation.ChatLog,
            "async_add_assistant_content",
            fake_add_assistant_content,
        ),
    ):
        result = await agent._async_run_tool_calls(
            turn,
            [ToolCall(id="call", name="HassTurnOn", arguments={"name": "Hall Lamp"})],
        )

    speech = result.response.speech["plain"]["speech"]
    assert speech == "I couldn't complete that action."
    assert speech != "Done."
    assert turn.trace.error_stage == Stage.HA_ACTION
    assert turn.trace.error_type == ErrorType.HA_ACTION_FAILED


async def test_unflagged_success_stays_done(error_flag_turn):
    agent, turn = error_flag_turn

    async def fake_add_assistant_content(self, content, tool_call_tasks=None):
        yield _NewShapeContent({"success": True}, False)

    with (
        patch.object(
            conversation.ChatLog,
            "async_add_assistant_content",
            fake_add_assistant_content,
        ),
    ):
        result = await agent._async_run_tool_calls(
            turn,
            [ToolCall(id="call", name="HassTurnOn", arguments={"name": "Hall Lamp"})],
        )

    assert result.response.speech["plain"]["speech"] == "Done."
