from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from custom_components.sayso.core.completion import ChatCompletionResult, ToolCall
from custom_components.sayso.core.grounding import GroundingEntity, GroundingHome
from custom_components.sayso.core.post_model import (
    ExecutionResult,
    TurnInput,
    ValidationResult,
    run_turn,
)

TURN_ON = {
    "type": "function",
    "function": {
        "name": "intent__HassTurnOn",
        "description": "Turn on",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "area": {"type": "string"},
            },
        },
    },
}

DESK_LAMP = GroundingEntity("light.desk_lamp", "Desk Lamp", (), "light", "Office", None)
ATTIC_FAN = GroundingEntity("fan.attic_fan", "Attic Fan", (), "fan", "Attic", None)
HOME = GroundingHome(
    entities=(DESK_LAMP, ATTIC_FAN),
    areas={"Office": (), "Attic": (), "Kitchen": ()},
    satellite_area="Kitchen",
    satellite_device_id=None,
)
UTTERANCE = "Turn on the fan and the lamp."


class _Model:
    def __init__(self, *replies: ChatCompletionResult) -> None:
        self.replies = list(replies)

    async def __call__(self, messages: list[dict[str, Any]], *, correction: bool):
        return self.replies.pop(0)


class _Recorder:
    def open(self, stage: Any, **metadata: Any) -> SimpleNamespace:
        return SimpleNamespace(metadata=dict(metadata))

    def close(self, span: Any, *, success: bool = True) -> None:
        return None


def _turn(**overrides: Any) -> TurnInput:
    base: dict[str, Any] = {
        "user_text": UTTERANCE,
        "transcript": lambda: [{"role": "user", "content": UTTERANCE}],
        "tools": (TURN_ON,),
        "schema_fingerprint": "fp",
        "available_names": {"intent__HassTurnOn"},
        "action_tools": frozenset({"intent__HassTurnOn"}),
        "exposed_domains": frozenset({"light", "fan"}),
        "home": lambda: HOME,
        "tool_api": True,
        "iteration_limit": lambda: 3,
        "now": 0.0,
        "held_ttl": 300.0,
    }
    base.update(overrides)
    return TurnInput(**base)


async def _clarify_then_confirm(fan_name: str, model: _Model):
    executed: list[tuple[ToolCall, dict[str, Any]]] = []

    async def execute(validated):
        executed.extend(validated)
        return ExecutionResult(failed=False)

    def validate(calls):
        return ValidationResult([(call, call.arguments) for call in calls], [])

    batch = ChatCompletionResult(
        content=None,
        tool_calls=[
            ToolCall(
                id="lamp", name="intent__HassTurnOn", arguments={"name": "Reading Lamp"}
            ),
            ToolCall(id="fan", name="intent__HassTurnOn", arguments={"name": fan_name}),
        ],
    )
    first = await run_turn(
        _turn(),
        complete=_Model(batch),
        validate=validate,
        execute=execute,
        recorder=_Recorder(),
    )
    assert first.held is not None
    assert first.held.index == 0
    assert executed == []

    second = await run_turn(
        _turn(user_text="yes", held=first.held),
        complete=model,
        validate=validate,
        execute=execute,
        recorder=_Recorder(),
    )
    return second, executed


async def test_confirmed_replay_clarifies_when_other_call_is_ungrounded() -> None:
    second, executed = await _clarify_then_confirm("Ceiling Fan", _Model())
    assert second.held_replayed is True
    assert second.clarified is True
    assert second.held is not None
    assert second.held.index == 1
    assert second.executed == ()
    assert executed == []


async def test_confirmed_replay_grounds_other_calls_and_executes_batch() -> None:
    second, executed = await _clarify_then_confirm(
        "Attic Fan", _Model(ChatCompletionResult(content="Both are on.", tool_calls=[]))
    )
    assert second.held_replayed is True
    assert second.clarified is False
    assert second.speech == "Both are on."
    assert [(c.id, args) for c, args in executed] == [
        ("lamp", {"name": "Desk Lamp"}),
        ("fan", {"name": "Attic Fan"}),
    ]
