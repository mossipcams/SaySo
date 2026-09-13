"""Unit tests for the live command smoke harness (no network)."""

from __future__ import annotations

from scripts.live_command_smoke import (
    MATRIX,
    Command,
    base_tool_name,
    build_utterance,
    classify,
    discover_agent,
    speech_from_response,
    targets_by_domain,
    unwrap_service_response,
)


def _state(entity_id: str, state: str = "on", name: str | None = None) -> dict:
    attributes = {"friendly_name": name} if name else {}
    return {"entity_id": entity_id, "state": state, "attributes": attributes}


def _response(speech: str = "Done.", response_type: str = "action_done") -> dict:
    return {
        "response": {
            "response_type": response_type,
            "speech": {"plain": {"speech": speech}},
        },
        "conversation_id": "c1",
    }


def _trace(tool: str | None = "HassTurnOn", *, success=True, target="light.kitchen") -> dict:
    summary: dict = {"success": success, "trace_id": "t1"}
    if tool:
        summary["tool"] = tool
    if target:
        summary["target"] = target
    if not success:
        summary["error_stage"] = "ha_action"
        summary["error_type"] = "ha_action_failed"
    return {"summary": summary, "events": []}


def test_unwrap_service_response_strips_the_ha_envelope() -> None:
    inner = {"traces": [{"trace_id": "t1"}]}
    wrapped = {"changed_states": [], "service_response": inner}
    assert unwrap_service_response(wrapped) is inner
    # A bare dict is returned unchanged (older HA or a direct call).
    assert unwrap_service_response(inner) == inner
    assert unwrap_service_response(None) == {}


def test_matrix_has_no_duplicate_tools() -> None:
    tools = [command.tool for command in MATRIX]
    assert len(tools) == len(set(tools))


def test_matrix_covers_all_expected_families() -> None:
    tools = {command.tool for command in MATRIX}
    assert {
        "HassTurnOn",
        "HassTurnOff",
        "HassLightSet",
        "HassFanSetSpeed",
        "HassClimateSetTemperature",
        "HassStartTimer",
        "HassCancelTimer",
        "GetLiveContext",
        "GetDateTime",
    } <= tools


def test_discover_agent_picks_the_only_non_default() -> None:
    states = [_state("conversation.home_assistant"), _state("conversation.sayso_llama")]
    assert discover_agent(states) == "conversation.sayso_llama"


def test_discover_agent_rejects_ambiguous_home() -> None:
    states = [_state("conversation.sayso_a"), _state("conversation.sayso_b")]
    try:
        discover_agent(states)
    except RuntimeError as err:
        assert "SAYSO_AGENT" in str(err)
    else:  # pragma: no cover - explicit failure
        raise AssertionError("expected RuntimeError")


def test_targets_by_domain_skips_unavailable_and_conversation() -> None:
    states = [
        _state("light.kitchen", "on", "Kitchen Light"),
        _state("light.spare", "unavailable", "Spare Light"),
        _state("conversation.sayso"),
    ]
    targets = targets_by_domain(states)
    assert targets["light"] == ["Kitchen Light"]


def test_targets_by_domain_ranks_user_facing_before_diagnostics() -> None:
    states = [
        _state("light.apollo_msr_2_rgb_light", "off", "Apollo MSR-2 RGB Light"),
        _state("light.kitchen_light", "off", "Kitchen light"),
        _state("light.nightstand", "off", "Nightstand"),
    ]
    assert targets_by_domain(states)["light"] == ["Nightstand", "Kitchen light", "Apollo MSR-2 RGB Light"]


def test_build_utterance_fills_target_and_area() -> None:
    targets = {"light": ["Kitchen Light"], "vacuum": ["Robo"]}
    command = Command("HassTurnOn", "Turn on {target}", target_domains=("light",))
    assert build_utterance(command, targets, []) == "Turn on Kitchen Light"

    area_command = Command("HassVacuumCleanArea", "Vacuum the {area}", needs_area=True, requires="vacuum")
    assert build_utterance(area_command, targets, ["Kitchen"]) == "Vacuum the Kitchen"


def test_build_utterance_skips_when_domain_missing() -> None:
    command = Command("HassTurnOn", "Turn on {target}", target_domains=("light",), requires="light")
    assert build_utterance(command, {"fan": ["Fan"]}, []) is None


def test_speech_from_response_reads_plain_speech() -> None:
    assert speech_from_response(_response("All set.")) == "All set."
    assert speech_from_response(None) == ""
    assert speech_from_response({}) == ""


def test_classify_pass() -> None:
    command = Command("HassTurnOn", "Turn on {target}")
    status, detail = classify(command, response=_response(), trace=_trace("HassTurnOn"))
    assert status == "PASS"
    assert "HassTurnOn" in detail


def test_classify_fails_on_ha_error() -> None:
    command = Command("HassTurnOn", "Turn on {target}")
    status, _ = classify(command, response=_response("Nope.", "error"), trace=_trace())
    assert status == "FAIL"


def test_classify_fails_when_trace_reports_failure() -> None:
    command = Command("HassTurnOn", "Turn on {target}")
    status, detail = classify(
        command, response=_response(), trace=_trace("HassTurnOn", success=False)
    )
    assert status == "FAIL"
    assert "ha_action_failed" in detail


def test_classify_fails_action_without_tool_call() -> None:
    command = Command("HassTurnOn", "Turn on {target}")
    status, detail = classify(command, response=_response(), trace=_trace(None, target=None))
    assert status == "FAIL"
    assert "no tool call" in detail


def test_classify_allows_query_without_tool() -> None:
    command = Command("GetDateTime", "What time is it?")
    # A query tool must still be called; a bare spoken answer is not enough.
    status, detail = classify(
        command, response=_response("It is noon."), trace=_trace(None, target=None)
    )
    assert status == "FAIL"
    assert "no tool call" in detail
    ok, _ = classify(
        command, response=_response("It is noon."), trace=_trace("GetDateTime", target=None)
    )
    assert ok == "PASS"


def test_classify_fails_when_action_tool_resolves_no_entity() -> None:
    command = Command("HassTurnOn", "Turn on {target}")
    status, detail = classify(
        command, response=_response(), trace=_trace("HassTurnOn", target=None)
    )
    assert status == "FAIL"
    assert "resolved no entity" in detail


def test_classify_allows_timer_without_entity_target() -> None:
    command = Command("HassStartTimer", "Set a 5 minute timer")
    status, _ = classify(
        command, response=_response("Timer started."), trace=_trace("HassStartTimer", target=None)
    )
    assert status == "PASS"


def test_classify_strips_ha_namespace_prefix() -> None:
    command = Command("HassTurnOn", "Turn on {target}")
    status, detail = classify(
        command,
        response=_response(),
        trace=_trace("light__HassTurnOn", target="light.kitchen"),
    )
    assert status == "PASS"
    assert "light__HassTurnOn" in detail


def test_base_tool_name_strips_namespace() -> None:
    assert base_tool_name("media_player__HassMediaNext") == "HassMediaNext"
    assert base_tool_name("HassTurnOff") == "HassTurnOff"
    assert base_tool_name(None) == ""


def test_classify_rejects_wrong_tool_by_default() -> None:
    command = Command("HassTurnOff", "Turn off {target}")
    status, detail = classify(
        command, response=_response(), trace=_trace("HassTurnOn")
    )
    assert status == "FAIL"
    assert "expected HassTurnOff" in detail
