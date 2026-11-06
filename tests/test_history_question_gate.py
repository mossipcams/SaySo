from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from homeassistant.components import conversation
from homeassistant.helpers import llm

from custom_components.sayso.conversation import _Turn
from sayso_contract import history

AGENT_ID = "conversation.sayso"
ROOT = Path(__file__).resolve().parents[1]
SYSTEM = {"role": "system", "content": "You are SaySo."}


def _user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _assistant(text: str | None = "") -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _tool_result(call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)}


def _volume_turn(text: str, call_id: str) -> list[dict[str, Any]]:
    return [
        _user(text),
        _tool_call(call_id, "HassSetVolumeRelative", {"volume_step": "up"}),
        _tool_result(call_id, {"success": True}),
        _assistant("Done."),
    ]


def test_tonight_loop_sends_only_current_command() -> None:
    current = _user("Turn the living room lights on.")
    messages = [
        SYSTEM,
        *_volume_turn("Turn the TV volume up.", "call_1"),
        _user("Come live your lights on."),
        _tool_call("call_2", "HassSetVolumeRelative", {"entity": "Living room lights"}),
        _tool_result("call_2", {"error": "MatchFailedError"}),
        _user("Turn living room lights on."),
        _tool_call("call_3", "HassSetVolumeRelative", {"entity": "Living room lights"}),
        _tool_result("call_3", {"error": "MatchFailedError"}),
        current,
    ]

    assert history.model_history(messages) == [SYSTEM, current]


def test_acknowledgement_after_action_sends_only_current_command() -> None:
    current = _user("Thank you.")
    messages = [SYSTEM, *_volume_turn("Turn the TV volume up.", "call_1"), current]

    assert history.model_history(messages) == [SYSTEM, current]


def test_clarification_answer_keeps_question_turn() -> None:
    previous_user = _user("Turn off the TV.")
    question = _assistant("Did you mean the Samsung Tv or the TV?")
    current = _user("The Samsung one.")
    messages = [SYSTEM, previous_user, question, current]

    assert history.model_history(messages) == [SYSTEM, previous_user, question, current]


def test_question_two_turns_back_is_dropped_after_completed_turn() -> None:
    current = _user("Turn the kitchen light on.")
    messages = [
        SYSTEM,
        _user("Turn off the TV."),
        _assistant("Did you mean the Samsung Tv or the TV?"),
        *_volume_turn("Turn the TV volume up.", "call_1"),
        current,
    ]

    assert history.model_history(messages) == [SYSTEM, current]


def test_current_turn_tool_call_and_result_are_kept_after_question() -> None:
    previous_user = _user("Turn off the TV.")
    question = _assistant("Did you mean the Samsung Tv or the TV?")
    current = _user("The Samsung one.")
    current_call = _tool_call("call_9", "HassTurnOff", {"name": "Samsung Tv"})
    current_result = _tool_result("call_9", {"success": True})
    messages = [
        SYSTEM,
        *_volume_turn("Turn the TV volume up.", "call_1"),
        previous_user,
        question,
        current,
        current_call,
        current_result,
    ]

    assert history.model_history(messages) == [
        SYSTEM,
        previous_user,
        question,
        current,
        current_call,
        current_result,
    ]


def test_current_turn_tool_call_and_result_are_kept_without_question() -> None:
    current = _user("Turn the kitchen light on.")
    current_call = _tool_call("call_9", "HassTurnOn", {"name": "Kitchen"})
    current_result = _tool_result("call_9", {"success": True})
    messages = [
        SYSTEM,
        *_volume_turn("Turn the TV volume up.", "call_1"),
        current,
        current_call,
        current_result,
    ]

    assert history.model_history(messages) == [
        SYSTEM,
        current,
        current_call,
        current_result,
    ]


@pytest.mark.parametrize(
    "question_text",
    [
        "Did you mean the Samsung Tv or the TV？",
        "Which one;",
        "Which one;  \n",
        "Which one?  \n",
    ],
)
def test_question_endings_count_as_question(question_text: str) -> None:
    previous_user = _user("Turn off the TV.")
    question = _assistant(question_text)
    current = _user("The Samsung one.")
    messages = [SYSTEM, previous_user, question, current]

    assert history.model_history(messages) == [SYSTEM, previous_user, question, current]


@pytest.mark.parametrize("content", [None, "", "Done.", "Which one;"])
def test_non_question_assistant_content_drops_earlier_turn(content: str | None) -> None:
    current = _user("The Samsung one.")
    messages = [
        SYSTEM,
        _user("Turn off the TV."),
        _assistant(content),
        current,
    ]

    assert history.model_history(messages) == [SYSTEM, current]


def test_no_user_message_returns_messages_unchanged() -> None:
    messages = [SYSTEM, _assistant("Hello.")]

    assert history.model_history(messages) is messages


def test_turn_messages_send_only_current_command_after_completed_action() -> None:
    chat_log = SimpleNamespace(
        content=[
            conversation.SystemContent(content="You are SaySo."),
            conversation.UserContent(content="Turn the TV volume up."),
            conversation.AssistantContent(
                agent_id=AGENT_ID,
                content=None,
                tool_calls=[
                    llm.ToolInput(
                        id="call_1",
                        tool_name="HassSetVolumeRelative",
                        tool_args={"volume_step": "up"},
                    )
                ],
            ),
            conversation.ToolResultContent(
                agent_id=AGENT_ID,
                tool_call_id="call_1",
                tool_name="HassSetVolumeRelative",
                tool_result={"success": True},
            ),
            conversation.AssistantContent(agent_id=AGENT_ID, content="Done."),
            conversation.UserContent(content="Thank you."),
        ]
    )
    turn = _Turn(
        entry_id="entry",
        agent_id=AGENT_ID,
        runtime=None,
        user_input=None,
        chat_log=chat_log,
        trace=None,
    )

    assert turn.messages == [
        {"role": "system", "content": "You are SaySo."},
        {"role": "user", "content": "Thank you."},
    ]


def test_history_module_has_no_homeassistant_import() -> None:
    tree = ast.parse((ROOT / "custom_components/sayso/history.py").read_text())
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]

    assert not [name for name in imported if name.startswith("homeassistant")]
