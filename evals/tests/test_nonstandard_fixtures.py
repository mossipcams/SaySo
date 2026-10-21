from __future__ import annotations

import json

import yaml

from evals.cases import load_home, select_cases
from evals.runner import render_case
from sayso_contract import tool_contract

LIVE_CONTEXT_HEADER = "Live Context: An overview of the areas and the devices in this smart home:\n"


def nonstandard_cases():
    return select_cases(tag="nonstandard")


def tool_messages(case):
    return [
        message
        for message in [*case.history, *case.after_utterance]
        if message["role"] == "tool"
    ]


def brightness_pct(raw: int) -> int:
    return max(1, round(raw / 255 * 100)) if raw > 0 else 0


def test_nonstandard_cases_pass_the_tool_contract() -> None:
    for case in nonstandard_cases():
        rendered = render_case(case)
        calls = [(call["name"], call["arguments"]) for call in case.expected["calls"]]
        assert not tool_contract.check_tool_calls(
            calls, rendered["tools"], rendered["exposed_domains"]
        ), case.id


def test_history_tool_calls_pass_the_tool_contract() -> None:
    for case in nonstandard_cases():
        rendered = render_case(case)
        for message in [*case.history, *case.after_utterance]:
            for call in message.get("tool_calls") or []:
                arguments = json.loads(call["function"]["arguments"])
                assert not tool_contract.check_tool_calls(
                    [(call["function"]["name"], arguments)],
                    rendered["tools"],
                    rendered["exposed_domains"],
                ), (case.id, call)


def test_action_results_use_the_verified_intent_shape() -> None:
    for case in nonstandard_cases():
        home = load_home(case.household)
        ids = {entity["name"]: entity["entity_id"] for entity in home["entities"]}
        for message in tool_messages(case):
            payload = json.loads(message["content"])
            if "response_type" not in payload:
                continue
            assert payload["response_type"] == "action_done", case.id
            assert payload == {
                "speech": {},
                "response_type": "action_done",
                "data": {
                    "success": payload["data"]["success"],
                    "failed": [],
                },
            }, case.id
            for item in payload["data"]["success"]:
                assert item == {
                    "name": item["name"],
                    "type": "entity",
                    "id": ids[item["name"]],
                }, case.id


def test_live_context_results_use_the_verified_shape() -> None:
    for case in nonstandard_cases():
        for message in tool_messages(case):
            payload = json.loads(message["content"])
            if "result" not in payload or "response_type" in payload:
                continue
            assert set(payload) == {"result"}, case.id
            assert payload["result"].startswith(LIVE_CONTEXT_HEADER), case.id
            entries = yaml.safe_load(payload["result"][len(LIVE_CONTEXT_HEADER):])
            for entry in entries:
                attributes = entry.get("attributes") or {}
                if entry["domain"] == "light" and "brightness" in attributes:
                    raw = attributes["brightness"]
                    assert isinstance(raw, str) and 0 <= int(raw) <= 255, case.id
                    pct = attributes["brightness_pct"]
                    assert isinstance(pct, str), case.id
                    assert int(pct) == brightness_pct(int(raw)), case.id
                if entry["domain"] == "media_player" and "volume_level" in attributes:
                    assert isinstance(attributes["volume_level"], float), case.id


def test_satellite_volume_is_relative_only() -> None:
    for case in nonstandard_cases():
        home = load_home(case.household)
        speakers = {
            entity["name"]
            for entity in home["entities"]
            if entity.get("device_class") == "speaker"
        }
        messages = [*case.history, *case.after_utterance]
        calls = [
            (call["function"]["name"], json.loads(call["function"]["arguments"]))
            for message in messages
            for call in message.get("tool_calls") or []
        ]
        calls += [(call["name"], call["arguments"]) for call in case.expected["calls"]]
        for name, arguments in calls:
            if arguments.get("name") not in speakers:
                continue
            assert name == "media_player__HassSetVolumeRelative", case.id
            assert arguments["volume_step"] in ("up", "down"), case.id


def test_relative_brightness_labels_follow_brightness_pct() -> None:
    cases = {case.id: case for case in nonstandard_cases()}
    apply_step = cases["nonstandard_16_apply_step"]
    live = json.loads(tool_messages(apply_step)[0]["content"])["result"]
    pct = int(yaml.safe_load(live[len(LIVE_CONTEXT_HEADER):])[0]["attributes"]["brightness_pct"])
    assert apply_step.expected["calls"][0]["arguments"]["brightness"] == pct + 20

    restore = cases["nonstandard_22_lamp_brightness"]
    live = json.loads(tool_messages(restore)[0]["content"])["result"]
    pct = int(yaml.safe_load(live[len(LIVE_CONTEXT_HEADER):])[0]["attributes"]["brightness_pct"])
    assert restore.expected["calls"][0]["arguments"]["brightness"] == pct


def test_family_four_is_relative_satellite_only() -> None:
    family_four = [
        case for case in nonstandard_cases() if "nonstandard_family_04" in case.tags
    ]
    assert family_four
    for case in family_four:
        for call in case.expected["calls"]:
            assert call["name"] == "media_player__HassSetVolumeRelative", case.id
            assert call["arguments"]["name"] == "Kitchen Satellite", case.id
            assert call["arguments"]["volume_step"] in ("up", "down"), case.id
