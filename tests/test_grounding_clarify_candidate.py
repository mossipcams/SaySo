from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from custom_components.sayso.core.completion import ChatCompletionResult, ToolCall
from custom_components.sayso.core.grounding import (
    GroundingEntity,
    GroundingHome,
    clarify,
    is_grounded,
)
from custom_components.sayso.core.post_model import (
    ExecutionResult,
    TurnInput,
    ValidationResult,
    run_turn,
)

TURN_OFF = {
    "type": "function",
    "function": {
        "name": "intent__HassTurnOff",
        "description": "Turn off",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "area": {"type": "string"},
                "domain": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

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
                "domain": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

LOUNGE_TV = GroundingEntity(
    "media_player.lounge_television",
    "Lounge Television",
    (),
    "media_player",
    "Living Room",
    None,
)
BEDROOM_TV = GroundingEntity(
    "media_player.bedroom_tv", "Bedroom TV", (), "media_player", "Bedroom", None
)
HOUSEHOLD_TWO = GroundingHome(
    entities=(LOUNGE_TV, BEDROOM_TV),
    areas={"Living Room": (), "Bedroom": ()},
    satellite_area="Living Room",
    satellite_device_id=None,
)

LIBRARIAN_TV_ARGS = {
    "domain": ["media_player"],
    "area": "Living Room",
    "name": "Lounge Television",
}


def _fan_home(
    *entities: GroundingEntity, satellite_area: str = "Living Room"
) -> GroundingHome:
    return GroundingHome(
        entities=entities,
        areas={"Attic": (), "Den": (), "Living Room": ()},
        satellite_area=satellite_area,
        satellite_device_id=None,
    )


ATTIC_FAN = GroundingEntity("fan.attic_fan", "Attic Fan", (), "fan", "Attic", None)
DEN_FAN = GroundingEntity("fan.den_fan", "Den Fan", (), "fan", "Den", None)


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


def _turn(text: str, **overrides: Any) -> TurnInput:
    base: dict[str, Any] = {
        "user_text": text,
        "transcript": lambda: [{"role": "user", "content": text}],
        "tools": (TURN_OFF, TURN_ON),
        "schema_fingerprint": "fp",
        "available_names": {"intent__HassTurnOff", "intent__HassTurnOn"},
        "action_tools": frozenset({"intent__HassTurnOff", "intent__HassTurnOn"}),
        "exposed_domains": frozenset({"media_player", "fan"}),
        "home": lambda: HOUSEHOLD_TWO,
        "tool_api": True,
        "iteration_limit": lambda: 3,
        "now": 0.0,
        "held_ttl": 300.0,
    }
    base.update(overrides)
    return TurnInput(**base)


async def test_librarian_tv_clarifies_with_satellite_candidate() -> None:
    text = "Turn off librarian reading TV."
    assert (
        is_grounded("intent__HassTurnOff", LIBRARIAN_TV_ARGS, text, HOUSEHOLD_TWO)
        is False
    )
    clar = clarify("intent__HassTurnOff", LIBRARIAN_TV_ARGS, text, HOUSEHOLD_TWO)
    assert clar.speech == "Did you mean the Lounge Television?"
    assert clar.options == (LOUNGE_TV,)


async def test_yes_replays_held_librarian_tv_call() -> None:
    executed: list[tuple[ToolCall, dict[str, Any]]] = []

    async def execute(validated):
        executed.extend(validated)
        return ExecutionResult(failed=False)

    def validate(calls):
        return ValidationResult([(call, call.arguments) for call in calls], [])

    call = ToolCall(
        id="c1", name="intent__HassTurnOff", arguments=dict(LIBRARIAN_TV_ARGS)
    )
    first = await run_turn(
        _turn("Turn off librarian reading TV."),
        complete=_Model(ChatCompletionResult(content=None, tool_calls=[call])),
        validate=validate,
        execute=execute,
        recorder=_Recorder(),
    )
    assert first.speech == "Did you mean the Lounge Television?"
    assert first.executed == ()
    assert executed == []
    assert first.held is not None
    assert first.held.options == (LOUNGE_TV,)

    second = await run_turn(
        _turn("yes", held=first.held),
        complete=_Model(),
        validate=validate,
        execute=execute,
        recorder=_Recorder(),
    )
    assert second.held_replayed is True
    assert second.speech == "Done."
    assert [(c.name, args) for c, args in executed] == [
        ("intent__HassTurnOff", {"name": "Lounge Television"})
    ]


async def test_no_domain_word_keeps_not_understood() -> None:
    text = "Turn off librarian."
    clar = clarify("intent__HassTurnOff", LIBRARIAN_TV_ARGS, text, HOUSEHOLD_TWO)
    assert clar.speech == "Sorry, I didn't understand that."
    assert clar.options == ()


async def test_ungrounded_fan_with_none_in_satellite_area_does_not_execute() -> None:
    home = _fan_home(ATTIC_FAN)
    args = {"domain": ["fan"], "name": "Ceiling Fan"}
    text = "Turn on the attic fan."
    assert is_grounded("intent__HassTurnOn", args, text, home) is False
    clar = clarify("intent__HassTurnOn", args, text, home)
    assert clar.options == (ATTIC_FAN,)
    assert clar.speech == "Did you mean the Attic Fan?"

    executed: list[tuple[ToolCall, dict[str, Any]]] = []

    async def execute(validated):
        executed.extend(validated)
        return ExecutionResult(failed=False)

    call = ToolCall(id="c1", name="intent__HassTurnOn", arguments=dict(args))
    outcome = await run_turn(
        _turn(
            text,
            home=lambda: home,
            tools=(TURN_ON,),
            available_names={"intent__HassTurnOn"},
            action_tools=frozenset({"intent__HassTurnOn"}),
            exposed_domains=frozenset({"fan"}),
        ),
        complete=_Model(ChatCompletionResult(content=None, tool_calls=[call])),
        validate=lambda calls: ValidationResult([(c, c.arguments) for c in calls], []),
        execute=execute,
        recorder=_Recorder(),
    )
    assert executed == []
    assert outcome.executed == ()
    assert outcome.speech == "Did you mean the Attic Fan?"


async def test_no_satellite_candidate_keeps_not_understood() -> None:
    home = _fan_home(ATTIC_FAN, DEN_FAN)
    args = {"domain": ["fan"], "name": "Garage Fan"}
    clar = clarify("intent__HassTurnOn", args, "Turn on the garage fan.", home)
    assert clar.speech == "Sorry, I didn't understand that."
    assert clar.options == ()
