from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from custom_components.sayso.core.completion import ChatCompletionResult, ToolCall
from custom_components.sayso.core.grounding import GroundingEntity, GroundingHome
from custom_components.sayso.core.held import HeldState
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

LIVE = {
    "type": "function",
    "function": {
        "name": "homeassistant__GetLiveContext",
        "description": "Live state",
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


def _reply(*calls: ToolCall, content: str | None = None) -> ChatCompletionResult:
    return ChatCompletionResult(content=content, tool_calls=list(calls))


class _Model:
    def __init__(self, *replies: ChatCompletionResult) -> None:
        self.replies = list(replies)
        self.requests: list[bool] = []

    async def __call__(self, messages: list[dict[str, Any]], *, correction: bool):
        self.requests.append(correction)
        return self.replies.pop(0)


class _Recorder:
    def open(self, stage: Any, **metadata: Any) -> SimpleNamespace:
        return SimpleNamespace(metadata=dict(metadata))

    def close(self, span: Any, *, success: bool = True) -> None:
        return None


def _turn(**overrides: Any) -> TurnInput:
    base: dict[str, Any] = {
        "user_text": "turn on the desk lamp",
        "transcript": lambda: [{"role": "user", "content": "turn on the desk lamp"}],
        "tools": (TURN_ON, LIVE),
        "schema_fingerprint": "fp",
        "available_names": {"intent__HassTurnOn", "homeassistant__GetLiveContext"},
        "action_tools": frozenset({"intent__HassTurnOn"}),
        "exposed_domains": frozenset({"light"}),
        "home": None,
        "tool_api": True,
        "iteration_limit": lambda: 3,
        "now": 0.0,
        "held_ttl": 300.0,
    }
    base.update(overrides)
    return TurnInput(**base)


async def _run(turn: TurnInput, model: _Model):
    executed: list[tuple[ToolCall, dict[str, Any]]] = []

    async def execute(validated):
        executed.extend(validated)
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


async def test_plain_speech_sets_no_flags() -> None:
    model = _Model(_reply(content="Hello there."))
    outcome, executed = await _run(_turn(), model)
    assert outcome.speech == "Hello there."
    assert outcome.executed == ()
    assert executed == []
    assert not (
        outcome.correction_used
        or outcome.clarified
        or outcome.done_shortcut
        or outcome.area_dropped
        or outcome.held_replayed
    )
    assert outcome.first is not None
    assert outcome.first.content == "Hello there."


async def test_done_shortcut_records_executed_call() -> None:
    model = _Model(_reply(_call("intent__HassTurnOn", {"name": "Desk Lamp"})))
    outcome, executed = await _run(_turn(), model)
    assert outcome.speech == "Done."
    assert outcome.done_shortcut is True
    assert outcome.executed == (
        ToolCall(id="c1", name="intent__HassTurnOn", arguments={"name": "Desk Lamp"}),
    )
    assert [call.name for call, _ in executed] == ["intent__HassTurnOn"]
    assert model.requests == [False]


async def test_follow_up_after_status_query_executes_and_speaks() -> None:
    model = _Model(
        _reply(_call("homeassistant__GetLiveContext", {"name": "Desk Lamp"})),
        _reply(content="The desk lamp is on."),
    )
    outcome, _ = await _run(_turn(user_text="is the desk lamp on"), model)
    assert outcome.speech == "The desk lamp is on."
    assert outcome.done_shortcut is False
    assert [call.name for call in outcome.executed] == ["homeassistant__GetLiveContext"]
    assert model.requests == [False, False]


async def test_correction_used_when_first_pass_fails_contract() -> None:
    model = _Model(
        _reply(_call("intent__HassTurnOn", {"name": "Desk Lamp", "bogus": 1})),
        _reply(_call("intent__HassTurnOn", {"name": "Desk Lamp"})),
    )
    outcome, _ = await _run(_turn(), model)
    assert outcome.correction_used is True
    assert outcome.done_shortcut is True
    assert outcome.executed == (
        ToolCall(id="c1", name="intent__HassTurnOn", arguments={"name": "Desk Lamp"}),
    )
    assert model.requests == [False, True]
    assert outcome.first.tool_calls[0].arguments == {"name": "Desk Lamp", "bogus": 1}


async def test_area_dropped_when_model_invents_unspoken_area() -> None:
    model = _Model(
        _reply(_call("intent__HassTurnOn", {"name": "Desk Lamp", "area": "Kitchen"}))
    )
    turn = _turn(home=lambda: _home(DESK_LAMP))
    outcome, _ = await _run(turn, model)
    assert outcome.area_dropped is True
    assert outcome.speech == "Done."
    assert outcome.executed[0].arguments == {"name": "Desk Lamp"}


async def test_clarified_turn_executes_nothing() -> None:
    model = _Model(_reply(_call("intent__HassTurnOn", {"name": "Bedside Lamp"})))
    turn = _turn(
        user_text="turn on the bedside lamp", home=lambda: _home(DESK_LAMP, FLOOR_LAMP)
    )
    outcome, executed = await _run(turn, model)
    assert outcome.clarified is True
    assert isinstance(outcome.speech, str) and outcome.speech
    assert outcome.executed == ()
    assert executed == []
    assert outcome.done_shortcut is False


async def test_held_option_replay_sets_flag() -> None:
    held = HeldState(
        calls=[_call("intent__HassTurnOn", {"name": "Desk Lamp", "area": "Office"})],
        index=0,
        options=(DESK_LAMP,),
        expires=10.0,
    )
    model = _Model()
    turn = _turn(user_text="yes", held=held, home=lambda: _home(DESK_LAMP))
    outcome, _ = await _run(turn, model)
    assert outcome.held_replayed is True
    assert outcome.speech == "Done."
    assert outcome.executed == (
        ToolCall(id="c1", name="intent__HassTurnOn", arguments={"name": "Desk Lamp"}),
    )
    assert outcome.first is None
    assert model.requests == []


async def test_supplied_calls_have_no_first_completion() -> None:
    model = _Model()
    turn = _turn(
        calls=(_call("intent__HassTurnOn", {"name": "Desk Lamp"}),), confirmed=True
    )
    outcome, _ = await _run(turn, model)
    assert outcome.first is None
    assert outcome.done_shortcut is True
    assert len(outcome.executed) == 1
    assert model.requests == []
