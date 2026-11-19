from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_expose_entity
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar, entity_registry as er, intent
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.client import (
    ChatCompletionResult,
    ToolCall,
    _read_json_body,
)
from custom_components.sayso.completion import extract_tool_calls, parse_tool_calls
from custom_components.sayso.conversation import (
    SaySoConversationEntity,
    _Turn,
    execution_failure_speech,
)
from custom_components.sayso.exceptions import (
    SaySoConnectionError,
    SaySoInvalidResponseError,
    SaySoInvalidToolCallsError,
    SaySoInvalidToolEnvelopeError,
    SaySoTimeoutError,
)
from custom_components.sayso.schema import CompiledToolSchema
from custom_components.sayso.tracing import ErrorType, Stage, TraceContext


@pytest.fixture
async def failure_turn(hass: HomeAssistant):
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    tool = SimpleNamespace(
        name="HassTurnOn",
        parameters=vol.Schema(
            {
                vol.Optional("name"): str,
                vol.Optional("domain"): vol.Any(str, [str]),
                vol.Optional("area"): str,
            }
        ),
    )
    api = SimpleNamespace(tools=[tool], async_call_tool=AsyncMock())
    turn = _Turn(
        entry_id="failure-speech",
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
        chat_log=conversation.ChatLog(hass, "failure-speech", llm_api=api),
        trace=TraceContext("failure-speech"),
    )
    agent = SaySoConversationEntity(MockConfigEntry(domain="sayso"))
    agent.hass = hass
    return agent, turn, api


def _match_text(reason, text_format):
    if text_format == "repr":
        return str(
            intent.MatchFailedError(
                intent.MatchTargetsResult(False, intent.MatchFailedReason[reason]),
                intent.MatchTargetsConstraints(),
            )
        )
    return {
        "NAME": "No entities matched the name (given name 'Hall Lamp')",
        "STATE": "No entities were in the required state (given name 'Hall Lamp')",
        "DOMAIN": "No entities matched the domain (given area 'Bedroom', domains light)",
        "AREA": "No entities matched the area",
    }[reason]


def _assert_reply(result, turn, speech, stage, error_type):
    assert result.response.speech["plain"]["speech"] == speech
    assert not speech.endswith("?")
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert turn.trace.error_stage == stage
    assert turn.trace.error_type == error_type


@pytest.mark.parametrize("text_format", ["repr", "sentence"])
@pytest.mark.parametrize(
    ("tool_name", "args", "reason", "utterance", "expected"),
    [
        (
            "HassTurnOn",
            {"name": "Hall Lamp"},
            "NAME",
            "turn on the HALL lamp",
            "I couldn't find Hall Lamp.",
        ),
        (
            "HassTurnOn",
            {"name": "Hall Lamp"},
            "NAME",
            "turn on the light",
            "Sorry, I didn't understand that.",
        ),
        (
            "HassTurnOn",
            {},
            "NAME",
            "turn on the light",
            "Sorry, I didn't understand that.",
        ),
        (
            "HassMediaPause",
            {"name": "Living Room TV", "domain": "media_player"},
            "STATE",
            "pause the TV",
            "The Living Room TV isn't playing.",
        ),
        (
            "HassMediaUnpause",
            {"name": "Living Room TV", "domain": ["media_player"]},
            "STATE",
            "resume the TV",
            "The Living Room TV isn't paused.",
        ),
        (
            "HassTurnOn",
            {"name": "Hall Lamp", "domain": "light"},
            "STATE",
            "turn on the lamp",
            "I couldn't complete that action.",
        ),
        (
            "media_player__HassMediaPause",
            {"name": "Living Room TV"},
            "STATE",
            "pause the TV",
            "The Living Room TV isn't playing.",
        ),
        (
            "HassMediaPause",
            {"domain": "media_player"},
            "STATE",
            "pause",
            "I couldn't complete that action.",
        ),
        (
            "media_player__HassMediaUnpause",
            {"name": "Living Room TV"},
            "STATE",
            "resume the TV",
            "The Living Room TV isn't paused.",
        ),
        (
            "HassTurnOn",
            {"name": "Hall Lamp"},
            "AREA",
            "turn on Hall Lamp",
            "I couldn't complete that action.",
        ),
        (
            "HassTurnOn",
            {"area": "Bedroom", "domain": "light"},
            "DOMAIN",
            "turn on bedroom lights",
            "The bedroom has no lights available.",
        ),
    ],
)
async def test_ha_failure_speech(
    failure_turn, text_format, tool_name, args, reason, utterance, expected
):
    agent, turn, api = failure_turn
    ar.async_get(agent.hass).async_create("Bedroom")
    turn.user_input.text = utterance
    api.tools[0].name = tool_name
    error_text = _match_text(reason, text_format)
    api.async_call_tool.return_value = {
        "error": "MatchFailedError",
        "error_text": error_text,
    }
    call = ToolCall(id="call", name=tool_name, arguments=args)
    with patch(
        "custom_components.sayso.conversation.execution_failure_speech",
        wraps=execution_failure_speech,
    ) as choose:
        result = await agent._async_run_tool_calls(turn, [call])
    assert choose.call_args.args[-3:] == ("MatchFailedError", error_text, utterance)
    assert turn.chat_log.content[-1].tool_result == api.async_call_tool.return_value
    api.async_call_tool.assert_awaited_once()
    _assert_reply(result, turn, expected, Stage.HA_ACTION, ErrorType.HA_ACTION_FAILED)


@pytest.mark.parametrize(
    "tool_name",
    [
        "HassStartTimer",
        "HassCancelTimer",
        "HassCancelAllTimers",
        "HassIncreaseTimer",
        "HassDecreaseTimer",
        "HassPauseTimer",
        "HassUnpauseTimer",
        "HassTimerStatus",
    ],
)
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("namespace", ["", "intent__"])
async def test_timer_failure_speech(failure_turn, tool_name, named, namespace):
    agent, turn, api = failure_turn
    tool_name = namespace + tool_name
    api.tools[0].name = tool_name
    api.async_call_tool.return_value = {"error": "TimerNotFoundError", "error_text": ""}
    call = ToolCall(
        id="timer", name=tool_name, arguments={"name": "pasta"} if named else {}
    )
    result = await agent._async_run_tool_calls(turn, [call])
    expected = (
        "I couldn't complete that action." if named else "There are no timers running."
    )
    _assert_reply(result, turn, expected, Stage.HA_ACTION, ErrorType.HA_ACTION_FAILED)


@pytest.mark.parametrize(
    ("error", "error_text"),
    [
        ("MatchFailedError", None),
        ("MatchFailedError", "unknown reason"),
        ("MatchFailedError", {"reason": "NAME"}),
        ("ServiceNotFound", "MatchFailedReason.NAME"),
        ("IntentHandleError", "No entities matched the name"),
        ("TimerNotFoundError", ""),
    ],
)
async def test_other_errors_keep_generic_speech(failure_turn, error, error_text):
    agent, turn, api = failure_turn
    api.async_call_tool.return_value = {"error": error, "error_text": error_text}
    result = await agent._async_run_tool_calls(
        turn, [ToolCall(id="call", name="HassTurnOn", arguments={"name": "Hall Lamp"})]
    )
    _assert_reply(
        result,
        turn,
        "I couldn't complete that action.",
        Stage.HA_ACTION,
        ErrorType.HA_ACTION_FAILED,
    )


@pytest.mark.parametrize(
    ("tool_name", "error", "error_text"),
    [
        ("HassCancelTimer", "TimerNotFoundError", ""),
        ("HassTurnOn", "MatchFailedError", "No entities matched the name"),
        (
            "HassMediaPause",
            "MatchFailedError",
            "No entities were in the required state",
        ),
    ],
)
async def test_multiple_calls_keep_generic_speech(
    failure_turn, tool_name, error, error_text
):
    agent, turn, api = failure_turn
    api.tools[0].name = tool_name
    api.async_call_tool.return_value = {"error": error, "error_text": error_text}
    args = (
        {}
        if tool_name == "HassCancelTimer"
        else {"name": "Hall Lamp", "domain": "media_player"}
    )
    calls = [
        ToolCall(id=str(index), name=tool_name, arguments=args) for index in range(2)
    ]
    result = await agent._async_run_tool_calls(turn, calls)
    _assert_reply(
        result,
        turn,
        "I couldn't complete that action.",
        Stage.HA_ACTION,
        ErrorType.HA_ACTION_FAILED,
    )
    assert api.async_call_tool.await_count == 2


@pytest.mark.parametrize("exposed", [False, True])
async def test_area_domain_respects_exposure(failure_turn, exposed):
    agent, turn, api = failure_turn
    area = ar.async_get(agent.hass).async_create("Bedroom")
    registry = er.async_get(agent.hass)
    entity = registry.async_get_or_create("light", "test", "bedroom")
    registry.async_update_entity(entity.entity_id, area_id=area.id)
    agent.hass.states.async_set(entity.entity_id, "off")
    async_expose_entity(agent.hass, conversation.DOMAIN, entity.entity_id, exposed)
    api.async_call_tool.return_value = {
        "error": "MatchFailedError",
        "error_text": _match_text("DOMAIN", "repr"),
    }
    result = await agent._async_run_tool_calls(
        turn,
        [
            ToolCall(
                id="call",
                name="HassTurnOn",
                arguments={"area": "Bedroom", "domain": "light"},
            )
        ],
    )
    expected = (
        "I couldn't complete that action."
        if exposed
        else "The bedroom has no lights available."
    )
    _assert_reply(result, turn, expected, Stage.HA_ACTION, ErrorType.HA_ACTION_FAILED)


@pytest.mark.parametrize(
    ("call", "error_type"),
    [
        (
            ToolCall(id="call", name="Unavailable", arguments={}),
            ErrorType.UNAVAILABLE_TOOL,
        ),
        (
            ToolCall(id="call", name="HassTurnOn", arguments={"name": []}),
            ErrorType.INVALID_ARGUMENTS,
        ),
        (
            ToolCall(id="call", name="HassTurnOn", arguments={"unexpected": True}),
            ErrorType.SCHEMA_MISMATCH,
        ),
        (ToolCall(id="", name="HassTurnOn", arguments={}), ErrorType.INVALID_ARGUMENTS),
        (
            ToolCall(id="call", name="HassTurnOn", arguments=[]),
            ErrorType.INVALID_ARGUMENTS,
        ),
    ],
)
async def test_model_tool_rejections(failure_turn, call, error_type):
    agent, turn, api = failure_turn
    result = await agent._async_run_tool_calls(turn, [call])
    _assert_reply(
        result, turn, "Sorry, I didn't understand that.", Stage.TOOL_PARSE, error_type
    )
    api.async_call_tool.assert_not_awaited()


async def test_tool_without_api_is_rejected(failure_turn):
    agent, turn, api = failure_turn
    turn.chat_log.llm_api = None
    result = await agent._async_run_tool_calls(
        turn, [ToolCall(id="call", name="HassTurnOn", arguments={})]
    )
    _assert_reply(
        result,
        turn,
        "Sorry, I didn't understand that.",
        Stage.TOOL_PARSE,
        ErrorType.UNAVAILABLE_TOOL,
    )
    api.async_call_tool.assert_not_awaited()


async def test_invalid_tool_envelope(failure_turn):
    agent, turn, api = failure_turn
    with (
        patch.object(conversation.ChatLog, "async_provide_llm_data", new=AsyncMock()),
        patch(
            "custom_components.sayso.conversation.compile_llm_tools",
            side_effect=SaySoInvalidToolEnvelopeError("invalid envelope"),
        ),
    ):
        result = await agent._async_answer(turn)
    _assert_reply(
        result,
        turn,
        "Sorry, I didn't understand that.",
        Stage.CONTEXT,
        ErrorType.SCHEMA_MISMATCH,
    )
    turn.runtime.engine.async_chat_completion.assert_not_awaited()
    api.async_call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    ("error", "speech", "error_type"),
    [
        (
            SaySoInvalidToolCallsError("invalid tool calls"),
            "Sorry, I didn't understand that.",
            ErrorType.INVALID_MODEL_OUTPUT,
        ),
        (
            SaySoInvalidResponseError("llama.cpp returned invalid JSON"),
            "The local model is unavailable.",
            ErrorType.INVALID_MODEL_OUTPUT,
        ),
        (
            SaySoInvalidResponseError("Local inference failed: backend crash"),
            "The local model is unavailable.",
            ErrorType.INVALID_MODEL_OUTPUT,
        ),
        (
            SaySoTimeoutError("timeout"),
            "That request took too long.",
            ErrorType.MODEL_TIMEOUT,
        ),
        (
            SaySoConnectionError("unavailable"),
            "The local model is unavailable.",
            ErrorType.MODEL_UNAVAILABLE,
        ),
    ],
)
async def test_inference_failure_speech(failure_turn, error, speech, error_type):
    agent, turn, api = failure_turn
    turn.runtime.engine.async_chat_completion.side_effect = error
    with (
        patch.object(conversation.ChatLog, "async_provide_llm_data", new=AsyncMock()),
        patch(
            "custom_components.sayso.conversation.compile_llm_tools",
            return_value=None,
        ),
    ):
        result = await agent._async_answer(turn)
    _assert_reply(result, turn, speech, Stage.INFERENCE, error_type)
    api.async_call_tool.assert_not_awaited()


@pytest.mark.parametrize("content", [None, "I cannot do that."])
async def test_correction_without_calls_is_rejected(failure_turn, content):
    agent, turn, api = failure_turn
    turn.complete_schema = CompiledToolSchema(tools=(), fingerprint="test")
    turn.runtime.engine.async_chat_completion.return_value = ChatCompletionResult(
        content=content, tool_calls=[]
    )
    result = await agent._async_run_tool_calls(
        turn, [ToolCall(id="call", name="HassTurnOn", arguments={"name": []})]
    )
    _assert_reply(
        result,
        turn,
        "Sorry, I didn't understand that.",
        Stage.TOOL_PARSE,
        ErrorType.INVALID_MODEL_OUTPUT,
    )
    assert turn.correction_used
    turn.runtime.engine.async_chat_completion.assert_awaited_once()
    api.async_call_tool.assert_not_awaited()


async def test_iteration_limit_keeps_existing_speech(failure_turn):
    agent, turn, api = failure_turn
    call = ToolCall(id="call", name="HassTurnOn", arguments={"name": "Hall Lamp"})
    api.async_call_tool.return_value = {"success": True}
    turn.runtime.engine.async_chat_completion.return_value = ChatCompletionResult(
        content=None, tool_calls=[call]
    )
    result = await agent._async_run_tool_calls(turn, [call])
    _assert_reply(
        result,
        turn,
        "I couldn't complete that action.",
        Stage.INFERENCE,
        ErrorType.ITERATION_LIMIT,
    )
    api.async_call_tool.assert_awaited_once()


@pytest.mark.parametrize(
    "raw",
    [
        {},
        [None],
        [{}],
        [{"id": "call", "function": {"name": "", "arguments": {}}}],
        [{"function": {"name": "HassTurnOn", "arguments": {}}}],
        [{"id": "call", "function": {"name": "HassTurnOn", "arguments": "{"}}],
        [{"id": "call", "function": {"name": "HassTurnOn", "arguments": "[]"}}],
    ],
)
async def test_invalid_tool_call_parser_selects_misunderstood(failure_turn, raw):
    with pytest.raises(SaySoInvalidToolCallsError) as caught:
        parse_tool_calls(raw, subject="llama.cpp")
    await test_inference_failure_speech(
        failure_turn,
        caught.value,
        "Sorry, I didn't understand that.",
        ErrorType.INVALID_MODEL_OUTPUT,
    )


async def test_invalid_local_tool_call_selects_misunderstood(failure_turn):
    with pytest.raises(SaySoInvalidToolCallsError) as caught:
        extract_tool_calls("<|tool_call_start|>HassTurnOn(name=<|tool_call_end|>")
    await test_inference_failure_speech(
        failure_turn,
        caught.value,
        "Sorry, I didn't understand that.",
        ErrorType.INVALID_MODEL_OUTPUT,
    )


@pytest.mark.parametrize(
    "json_result",
    [ValueError("invalid JSON"), [], {"error": "model failed"}],
)
async def test_invalid_backend_response_keeps_unavailable(failure_turn, json_result):
    response = SimpleNamespace(json=AsyncMock())
    if isinstance(json_result, Exception):
        response.json.side_effect = json_result
    else:
        response.json.return_value = json_result
    with pytest.raises(SaySoInvalidResponseError) as caught:
        await _read_json_body(response)
    assert not isinstance(caught.value, SaySoInvalidToolCallsError)
    await test_inference_failure_speech(
        failure_turn,
        caught.value,
        "The local model is unavailable.",
        ErrorType.INVALID_MODEL_OUTPUT,
    )
