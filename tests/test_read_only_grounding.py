from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from custom_components.sayso.boundary import tool_is_read_only
from custom_components.sayso.conversation import _is_read_only
from custom_components.sayso.core.completion import ChatCompletionResult, ToolCall
from custom_components.sayso.core.grounding import GroundingEntity, GroundingHome
from custom_components.sayso.core.post_model import (
    ExecutionResult,
    TurnInput,
    ValidationResult,
    run_turn,
)
from custom_components.sayso.core.tool_contract import READ_ONLY_FALLBACK

TURN_ON = {
    "type": "function",
    "function": {
        "name": "intent__HassTurnOn",
        "description": "Turn on",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
    },
}

TURN_OFF = {
    "type": "function",
    "function": {
        "name": "intent__HassTurnOff",
        "description": "Turn off",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
    },
}

DESK_LAMP = GroundingEntity("light.desk_lamp", "Desk Lamp", (), "light", "Office", None)
FLOOR_LAMP = GroundingEntity("light.floor_lamp", "Floor Lamp", (), "light", "Den", None)


def _home(*entities: GroundingEntity) -> GroundingHome:
    return GroundingHome(
        entities=entities,
        areas={"Office": (), "Den": (), "Kitchen": ()},
        satellite_area="Kitchen",
        satellite_device_id=None,
    )


def _call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


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


def _turn(calls: list[ToolCall], **overrides: Any) -> TurnInput:
    base: dict[str, Any] = {
        "user_text": "turn on the bedside lamp",
        "transcript": lambda: [{"role": "user", "content": "turn on the bedside lamp"}],
        "tools": (TURN_ON, TURN_OFF),
        "schema_fingerprint": "fp",
        "available_names": {"intent__HassTurnOn", "intent__HassTurnOff"},
        "action_tools": frozenset({"intent__HassTurnOn", "intent__HassTurnOff"}),
        "exposed_domains": frozenset({"light"}),
        "home": lambda: _home(DESK_LAMP, FLOOR_LAMP),
        "tool_api": True,
        "iteration_limit": lambda: 3,
        "now": 0.0,
        "held_ttl": 300.0,
        "calls": tuple(calls),
    }
    base.update(overrides)
    return TurnInput(**base)


async def _run(turn: TurnInput, model: _Model):
    executed: list[str] = []

    async def execute(validated):
        executed.extend(call.name for call, _ in validated)
        return ExecutionResult(failed=False)

    outcome = await run_turn(
        turn,
        complete=model,
        validate=lambda calls: ValidationResult(
            [(call, call.arguments) for call in calls], []
        ),
        execute=execute,
        recorder=_Recorder(),
    )
    return outcome, executed


async def test_read_only_call_that_would_fail_grounding_executes() -> None:
    turn = _turn(
        [_call("intent__HassTurnOn", {"name": "Bedside Lamp"})],
        read_only=frozenset({"intent__HassTurnOn"}),
    )
    outcome, executed = await _run(turn, _Model())
    assert outcome.clarified is False
    assert outcome.speech == "Done."
    assert executed == ["intent__HassTurnOn"]


async def test_same_call_without_read_only_clarifies() -> None:
    turn = _turn([_call("intent__HassTurnOn", {"name": "Bedside Lamp"})])
    outcome, executed = await _run(turn, _Model())
    assert outcome.clarified is True
    assert outcome.executed == ()
    assert executed == []


async def test_mixed_batch_grounds_only_non_read_only_calls() -> None:
    turn = _turn(
        [
            _call("intent__HassTurnOn", {"name": "Bedside Lamp"}, call_id="c1"),
            _call("intent__HassTurnOff", {"name": "Bedside Lamp"}, call_id="c2"),
        ],
        read_only=frozenset({"intent__HassTurnOn"}),
    )
    outcome, executed = await _run(turn, _Model())
    assert outcome.clarified is True
    assert outcome.executed == ()
    assert executed == []


async def test_mixed_batch_executes_grounded_non_read_only_call() -> None:
    turn = _turn(
        [
            _call("intent__HassTurnOn", {"name": "Bedside Lamp"}, call_id="c1"),
            _call("intent__HassTurnOff", {"name": "Desk Lamp"}, call_id="c2"),
        ],
        user_text="turn on the bedside lamp and turn off the desk lamp",
        read_only=frozenset({"intent__HassTurnOn"}),
    )
    model = _Model(ChatCompletionResult(content="Both done.", tool_calls=[]))
    outcome, executed = await _run(turn, model)
    assert outcome.clarified is False
    assert outcome.speech == "Both done."
    assert executed == ["intent__HassTurnOn", "intent__HassTurnOff"]


def test_read_only_defaults_to_empty() -> None:
    assert _turn([]).read_only == frozenset()


@pytest.mark.parametrize(
    ("annotations", "expected"),
    [
        (SimpleNamespace(read_only=True), True),
        (SimpleNamespace(read_only=False), False),
    ],
)
def test_tool_is_read_only_reads_annotation(
    annotations: SimpleNamespace, expected: bool
) -> None:
    assert tool_is_read_only(SimpleNamespace(annotations=annotations)) is expected


def test_tool_is_read_only_missing_annotations_is_none() -> None:
    assert tool_is_read_only(SimpleNamespace()) is None


def test_annotated_read_only_false_overrides_fallback_name() -> None:
    tool = SimpleNamespace(annotations=SimpleNamespace(read_only=False))
    assert "homeassistant__GetLiveContext" in READ_ONLY_FALLBACK
    assert _is_read_only("homeassistant__GetLiveContext", tool) is False


def test_annotated_read_only_true_applies_outside_fallback() -> None:
    tool = SimpleNamespace(annotations=SimpleNamespace(read_only=True))
    assert "intent__HassTurnOn" not in READ_ONLY_FALLBACK
    assert _is_read_only("intent__HassTurnOn", tool) is True


def test_missing_annotations_fall_back_to_name_set() -> None:
    assert _is_read_only("llm__GetDateTime", SimpleNamespace()) is True
    assert _is_read_only("intent__HassTurnOn", SimpleNamespace()) is False
