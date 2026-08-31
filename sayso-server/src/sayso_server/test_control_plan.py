"""ControlPlan validation and round-trip tests."""

import json

import pytest
from pydantic import ValidationError

from sayso_server.control_plan import ControlPlan


def test_action_plan_round_trips() -> None:
    payload = {
        "outcome": "action",
        "intent": "turn off living room lights except the lamp",
        "domain": "light",
        "scope": {"kind": "current_area"},
        "exclude": ["lamp"],
        "state": "off",
    }
    plan = ControlPlan.model_validate(payload)
    dumped = plan.model_dump(mode="json")
    assert ControlPlan.model_validate(dumped) == plan
    assert json.loads(json.dumps(dumped)) == dumped


def test_query_plan_round_trips() -> None:
    payload = {
        "outcome": "query",
        "intent": "check if any lights are on",
        "domain": "light",
        "scope": {"kind": "named_area", "name": "kitchen"},
    }
    plan = ControlPlan.model_validate(payload)
    dumped = plan.model_dump(mode="json")
    assert ControlPlan.model_validate(dumped) == plan


def test_clarification_plan_round_trips() -> None:
    payload = {
        "outcome": "clarification",
        "intent": "turn on the lamp",
        "reason": "multiple lamps match",
    }
    plan = ControlPlan.model_validate(payload)
    dumped = plan.model_dump(mode="json")
    assert ControlPlan.model_validate(dumped) == plan


def test_unsupported_plan_round_trips() -> None:
    payload = {
        "outcome": "unsupported",
        "intent": "play music on spotify",
        "reason": "media playback is not supported",
    }
    plan = ControlPlan.model_validate(payload)
    dumped = plan.model_dump(mode="json")
    assert ControlPlan.model_validate(dumped) == plan


def test_no_action_plan_round_trips() -> None:
    payload = {
        "outcome": "no-action",
        "intent": "unknown request",
        "reason": "could not interpret command",
    }
    plan = ControlPlan.model_validate(payload)
    dumped = plan.model_dump(mode="json")
    assert ControlPlan.model_validate(dumped) == plan


def test_action_with_brightness_value_round_trips() -> None:
    payload = {
        "outcome": "action",
        "intent": "dim the ceiling light",
        "domain": "light",
        "targets": ["ceiling light"],
        "value": 40,
    }
    plan = ControlPlan.model_validate(payload)
    dumped = plan.model_dump(mode="json")
    assert ControlPlan.model_validate(dumped) == plan


def test_action_with_climate_mode_round_trips() -> None:
    payload = {
        "outcome": "action",
        "intent": "set thermostat to heat",
        "domain": "climate",
        "targets": ["thermostat"],
        "mode": "heat",
        "value": 72,
    }
    plan = ControlPlan.model_validate(payload)
    dumped = plan.model_dump(mode="json")
    assert ControlPlan.model_validate(dumped) == plan


def test_rejects_missing_intent() -> None:
    with pytest.raises(ValidationError, match="intent"):
        ControlPlan.model_validate(
            {
                "outcome": "action",
                "domain": "light",
                "state": "on",
            }
        )


def test_rejects_empty_intent() -> None:
    with pytest.raises(ValidationError, match="intent"):
        ControlPlan.model_validate(
            {
                "outcome": "query",
                "intent": "",
                "domain": "light",
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "outcome": "action",
            "intent": "turn off",
            "domain": "light",
            "state": "off",
            "value": 50,
        },
        {
            "outcome": "action",
            "intent": "toggle",
            "domain": "light",
            "state": "toggle",
            "value": 10,
        },
        {
            "outcome": "action",
            "intent": "set brightness",
            "domain": "light",
            "state": "invalid_state",
        },
    ],
)
def test_rejects_invalid_state_value_pairs(payload: dict) -> None:
    with pytest.raises(ValidationError):
        ControlPlan.model_validate(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("targets", ["light.living_room"]),
        ("include", ["switch.kitchen"]),
        ("exclude", ["scene.movie_time"]),
        ("targets", ["floor lamp", "light.bedroom"]),
    ],
)
def test_rejects_entity_ids_in_semantic_targets(field: str, value: list[str]) -> None:
    payload = {
        "outcome": "action",
        "intent": "control devices",
        "domain": "light",
        "state": "on",
        field: value,
    }
    with pytest.raises(ValidationError, match="entity"):
        ControlPlan.model_validate(payload)
