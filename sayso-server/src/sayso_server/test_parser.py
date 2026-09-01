"""Strict model-output parser tests."""

import json

from sayso_server.control_plan import ControlPlan, NoActionPlan
from sayso_server.parser import parse_model_output


def test_valid_json_control_plan_round_trips() -> None:
    payload = {
        "outcome": "query",
        "intent": "check if any lights are on",
        "domain": "light",
        "scope": {"kind": "named_area", "name": "kitchen"},
    }
    text = json.dumps(payload)

    plan = parse_model_output(text, intent="check if any lights are on")

    assert plan == ControlPlan.model_validate(payload)
    dumped = plan.model_dump(mode="json")
    assert ControlPlan.model_validate(dumped) == plan


def test_fenced_json_control_plan_round_trips() -> None:
    payload = {
        "outcome": "action",
        "intent": "turn off living room lights",
        "domain": "light",
        "state": "off",
    }
    text = f"```json\n{json.dumps(payload)}\n```"

    plan = parse_model_output(text, intent="turn off living room lights")

    assert plan == ControlPlan.model_validate(payload)


def test_malformed_json_becomes_no_action() -> None:
    plan = parse_model_output('{"outcome": "action",', intent="turn off")

    assert isinstance(plan, NoActionPlan)
    assert plan.outcome == "no-action"
    assert plan.reason == "model_output_invalid"
    assert plan.intent == "turn off"


def test_tool_call_wrapper_becomes_no_action() -> None:
    wrapped = {
        "name": "control_plan",
        "arguments": {
            "outcome": "action",
            "intent": "turn off",
            "domain": "light",
            "state": "off",
        },
    }

    plan = parse_model_output(json.dumps(wrapped), intent="turn off")

    assert isinstance(plan, NoActionPlan)
    assert plan.reason == "model_output_invalid"


def test_openai_tool_calls_wrapper_becomes_no_action() -> None:
    wrapped = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "control_plan",
                    "arguments": json.dumps(
                        {
                            "outcome": "query",
                            "intent": "status",
                            "domain": "light",
                        }
                    ),
                },
            }
        ]
    }

    plan = parse_model_output(json.dumps(wrapped), intent="status")

    assert isinstance(plan, NoActionPlan)
    assert plan.reason == "model_output_invalid"


def test_invalid_control_plan_schema_becomes_no_action() -> None:
    invalid = {
        "outcome": "action",
        "domain": "light",
        "state": "on",
    }

    plan = parse_model_output(json.dumps(invalid), intent="turn on")

    assert isinstance(plan, NoActionPlan)
    assert plan.reason == "model_output_invalid"


def test_malformed_json_with_empty_intent_becomes_no_action() -> None:
    plan = parse_model_output('{"outcome": "action",', intent="")

    assert isinstance(plan, NoActionPlan)
    assert plan.outcome == "no-action"
    assert plan.reason == "model_output_invalid"
    assert plan.intent == "unknown"
