"""Gates that stop a corpus from claiming supervision it does not carry.

Four properties, one test group each:

1. a refusal cannot fill a positive operation quota;
2. missing supervision fails the dataset audit rather than the training run;
3. an action the entity cannot perform is rejected before it becomes a label;
4. changing the entity graph changes the expected target or outcome.
"""

from __future__ import annotations

import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.schema import v2_openai_tools
from generators import grounding
from generators.audit import audit_rows
from generators.capability_registry import (
    CAPABILITIES,
    TRAINING_COVERAGE_EXCLUDED,
    covered_tool_names,
    entity_supports,
)
from generators.config import GeneratorConfig
from generators.coverage import classify_row, expected_tool
from generators.duplicates import DuplicateTracker
from generators.pipeline import generate_row, run_generation
from generators.sampling import QuotaTracker, build_quota_targets
from generators.scenarios import build_scenario
from generators.labels import render_example, scenario_to_spec
from generators.validate import validate_spec

SMOKE = GeneratorConfig(count=300, seed=31337, paraphrase_enabled=False)


def _row(tool_calls, *, capability="media_players", operation="turn_on", tier=1, reason=None,
         tools=None, final="Done."):
    messages = [
        {"role": "system", "content": "ctx"},
        {"role": "user", "content": "turn on the living room tv"},
    ]
    if tool_calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
    messages.append({"role": "assistant", "content": final})
    if tools is None:
        called = {call["function"]["name"] for call in tool_calls or []}
        tools = [t for t in v2_openai_tools() if t["function"]["name"] in called]
    return {
        "messages": messages,
        "tools": tools,
        "metadata": {
            "candidate_id": f"row_{id(messages)}",
            "tier": tier,
            "capability": capability,
            "operation": operation,
            "no_action_reason": reason,
        },
    }


def _call(name, arguments):
    return {"function": {"name": name, "arguments": json.dumps(arguments)}}


# 1. Refusals cannot fill a positive quota -------------------------------------


def test_a_refusal_does_not_count_as_positive_supervision():
    refusal = _row([], reason="unsupported", final="I can't do that with the available tools.")
    facets = classify_row(refusal)
    assert facets["outcome"] == "unsupported"
    assert facets["positive"] is False


def test_refusal_only_fills_the_negative_budget_of_its_own_bucket():
    tracker = QuotaTracker(400, seed=1)
    key = (1, "media_players", "turn_on")
    assert tracker.targets["positive"][key] > 0

    refusal = _row([], reason="unsupported")
    allowance = tracker._negative_allowance(key)
    for _ in range(allowance):
        assert tracker.wants(classify_row(refusal)) is None
        tracker.record_accept(refusal)
    # The bucket still needs every one of its positive rows.
    assert tracker.accepted_positive[key] == 0
    assert tracker.shortfall()["positive"][str(key)] == tracker.targets["positive"][key]
    # And it will not take another refusal.
    assert tracker.wants(classify_row(refusal)) == "quota_negative_full"


def test_a_call_on_the_wrong_domain_is_not_positive_supervision():
    light_call = [_call("HassTurnOn", {"name": "Kitchen Light", "domain": ["light"]})]
    assert classify_row(_row(light_call, capability="media_players"))["positive"] is False
    media_call = [_call("HassTurnOn", {"name": "TV", "domain": ["media_player"]})]
    assert classify_row(_row(media_call, capability="media_players"))["positive"] is True


def test_offering_a_tool_is_not_coverage():
    """A row that offers HassMediaPause but calls nothing has not taught it."""
    tools = [{"function": {"name": "HassMediaPause", "description": "", "parameters": {}}}]
    row = _row([], capability="media_players", operation="pause", reason="clarify", tools=tools)
    assert classify_row(row)["positive"] is False


def test_a_nonexistent_target_is_not_positive_supervision():
    present, *_ = grounding.media_presence_family(
        prefix="test_quota_target",
        target_name="TV",
        renamed="Cinema",
        area="Living Room",
        elsewhere="Bedroom",
        entity_id="media_player.living_room_tv",
    )
    row = grounding.build_row(present, seed=9)
    call = next(call for message in row["messages"] for call in message.get("tool_calls") or [])
    call["function"]["arguments"] = json.dumps(
        {"domain": ["media_player"], "name": "Missing TV"}
    )
    assert classify_row(row)["positive"] is False


def test_generated_positive_counts_never_exceed_accepted_rows():
    result = run_generation(SMOKE)
    quota = result["stats"]["quota"]
    positives = sum(quota["achieved"]["positive"].values())
    negatives = sum(quota["achieved"]["negative"].values())
    assert positives + negatives == SMOKE.count
    assert not any(quota["shortfall"].values())


def test_impossible_quota_configurations_fail_before_generating():
    with pytest.raises(ValueError, match="count must be at least 1"):
        build_quota_targets(0)
    with pytest.raises(ValueError, match="negative_rate"):
        build_quota_targets(100, negative_rate=1.0)
    with pytest.raises(ValueError, match="sum to 1"):
        build_quota_targets(100, {1: 0.5, 2: 0.2})


# 2. Missing supervision fails the audit ---------------------------------------


def test_audit_fails_when_a_required_operation_has_no_positive_row():
    rows = [_row([], capability="media_players", operation="turn_on", reason="unsupported")]
    with pytest.raises(ValueError, match="missing positive supervision for operations"):
        audit_rows(
            rows,
            required_operations={(1, "media_players", "turn_on")},
            min_positive_per_operation=1,
            min_positive_per_tool=0,
        )


def test_audit_fails_when_a_required_tool_has_no_positive_row():
    rows = [_row([_call("HassTurnOn", {"name": "TV", "domain": ["media_player"]})])]
    with pytest.raises(ValueError, match="missing positive supervision for tools"):
        audit_rows(
            rows,
            required_operations={(1, "media_players", "turn_on"), (1, "media_players", "pause")},
            min_positive_per_operation=0,
            min_positive_per_tool=1,
        )


def test_audit_caps_absence_answers():
    rows = [
        _row([], capability="lights", operation="turn_on", reason="area_unavailable",
             final="The den has no lights available.")
        for _ in range(4)
    ] + [_row([_call("HassTurnOn", {"name": "TV", "domain": ["media_player"]})])]
    with pytest.raises(ValueError, match="absence answers"):
        audit_rows(rows, max_absence_rate=0.10)
    # The same distribution passes when the cap allows it: 209 absence rows are a
    # distribution to look at, not proof of anything on their own.
    assert audit_rows(rows, max_absence_rate=0.90)["absence_rate"] == 0.8


def test_generation_covers_every_tool_inside_declared_coverage():
    report = run_generation(SMOKE)["stats"]["quality_audit"]
    covered = {tool for tool in covered_tool_names() if tool != "__script__"}
    missing = covered - set(report["positive_by_tool"])
    assert not missing, missing
    assert report["positive_by_tool"].get("__script__", 0) > 0


def test_enabling_grounding_produces_grounding_supervision():
    report = run_generation(
        GeneratorConfig(count=1200, seed=20260911, paraphrase_enabled=False)
    )["stats"]["grounding"]
    required = {variant["family"] for variant in grounding.required_training_variants()}
    assert required <= set(report["by_family"])


def test_tools_outside_declared_coverage_are_not_offered():
    rows = run_generation(SMOKE)["rows"]
    offered = {tool["function"]["name"] for row in rows for tool in row["tools"]}
    assert not offered & TRAINING_COVERAGE_EXCLUDED


# 3. Unsupported entity actions are rejected -----------------------------------


def test_an_entity_without_the_feature_cannot_be_given_the_action():
    speaker = grounding.entity(
        "Hall Speaker", "media_players", "Hallway", features=("play", "pause")
    )
    assert not entity_supports(speaker, "media_players", "turn_on")
    spec = {
        "capability": "media_players",
        "operation": "turn_on",
        "home": {"entities": [speaker], "sayso_entity_area": "Hallway"},
        "expected": {
            "kind": "action",
            "calls": [{"name": "HassTurnOn",
                       "arguments": {"name": "Hall Speaker", "domain": ["media_player"]}}],
        },
    }
    assert validate_spec(spec) == "entity_lacks_capability"


def test_a_device_unsupported_refusal_must_match_the_entity_graph():
    capable = grounding.entity(
        "Study TV", "media_players", "Study",
        features=("on", "off", "play", "pause", "volume", "volume_step", "mute"),
    )
    spec = {
        "capability": "media_players",
        "operation": "volume_set",
        "home": {"entities": [capable], "sayso_entity_area": "Study"},
        "expected": {"kind": "no_action", "response": "device_unsupported", "calls": [],
                     "unsupported_names": ["Study TV"]},
    }
    assert validate_spec(spec) == "contradictory_device_support"


@pytest.mark.parametrize("operation", ["turn_on", "pause", "volume_set", "mute", "search_and_play"])
def test_generated_media_scenarios_only_target_capable_devices(operation):
    """Scenarios carry the home, so this checks the label against the real graph."""
    checked = 0
    for index in range(60):
        scenario = build_scenario(
            index=index, seed=909, capability="media_players", operation=operation, home_size=32
        )
        spec = scenario_to_spec(scenario)
        assert validate_spec(spec) is None, (operation, validate_spec(spec))
        entities = {e["name"]: e for e in scenario["home"]["entities"]}
        for call in spec["expected"].get("calls") or []:
            entity = entities.get((call["arguments"] or {}).get("name"))
            if entity is None:
                continue
            matches = [
                op for op in CAPABILITIES[entity["capability"]].operations
                if op.tool_name == call["name"]
            ]
            assert any(
                entity_supports(entity, entity["capability"], op.name) for op in matches
            ), (call["name"], entity["name"], entity["capabilities"])
            checked += 1
    assert checked, f"no {operation} labels produced"


# 4. The entity graph decides the answer ---------------------------------------


def _grounding(variant):
    return grounding.build_spec(variant, seed=77)


def test_the_same_request_changes_answer_with_the_entity_graph():
    present, renamed, moved, distractors = grounding.media_presence_family(
        prefix="test_media",
        target_name="TV",
        renamed="Cinema",
        area="Living Room",
        elsewhere="Bedroom",
        entity_id="media_player.living_room_tv",
    )
    specs = {v["family"]: _grounding(v) for v in (present, renamed, moved, distractors)}

    # One request for all four graphs.
    prompts = {spec["utterance"] for spec in specs.values()}
    assert len(prompts) == 1, prompts

    def call(family):
        calls = specs[family]["expected"].get("calls") or []
        return calls[0] if calls else None

    assert call("test_media_present") == {
        "name": "HassTurnOn", "arguments": {"name": "TV", "domain": ["media_player"]}
    }
    assert call("test_media_renamed")["arguments"]["name"] == "Cinema"
    assert call("test_media_distractors")["arguments"]["name"] == "TV"
    moved_expected = specs["test_media_moved"]["expected"]
    assert moved_expected["kind"] == "no_action"
    assert moved_expected["response"] == "area_unavailable"
    assert not moved_expected["calls"]


def test_distractors_do_not_steal_the_target():
    *_, distractors = grounding.media_presence_family(
        prefix="test_distract", target_name="TV", renamed="Cinema", area="Living Room",
        elsewhere="Bedroom", entity_id="media_player.living_room_tv",
    )
    spec = _grounding(distractors)
    names = {e["name"] for e in spec["home"]["entities"]}
    assert {"Floor Lamp", "Console Outlet", "Bedroom Speaker"} <= names
    assert spec["expected"]["calls"][0]["arguments"]["name"] == "TV"


def test_one_light_acts_and_two_lights_ask():
    single, pair = grounding.ambiguity_family(
        prefix="test_amb", area="Kitchen", names=("Sink Light", "Island Pendants")
    )
    assert _grounding(single)["expected"]["kind"] == "action"
    ambiguous = _grounding(pair)["expected"]
    assert ambiguous["kind"] == "no_action" and ambiguous["response"] == "clarify"


def test_supported_actions_decide_between_acting_and_refusing():
    capable, incapable = grounding.supported_action_family(
        prefix="test_features", area="Porch", name="Porch Speaker"
    )
    capable_spec = grounding.build_spec(capable, seed=77, index=12)
    incapable_spec = grounding.build_spec(incapable, seed=77, index=13)
    acted = capable_spec["expected"]
    refused = incapable_spec["expected"]
    assert acted["kind"] == "action" and acted["calls"][0]["name"] == "HassSetVolume"
    assert refused["kind"] == "no_action" and refused["response"] == "device_unsupported"
    assert capable_spec["utterance"] == incapable_spec["utterance"]
    assert acted["calls"] == refused["requested"]["calls"]


def test_grounding_rows_render_in_the_production_format():
    variant = grounding.training_variants()[0]
    row = grounding.build_row(variant, seed=5)
    assert row["messages"][0]["role"] == "system"
    assert row["tools"] and all("function" in tool for tool in row["tools"])
    called = {
        call["function"]["name"]
        for message in row["messages"]
        for call in message.get("tool_calls") or []
    }
    assert called <= {tool["function"]["name"] for tool in row["tools"]}


def test_grounding_eval_prompts_are_held_out_of_training():
    from evals.grounding_eval import grounding_user_prompts
    from generators.pipeline import _check_quality_eval_overlap

    prompts = grounding_user_prompts()
    assert prompts
    for prompt in prompts:
        assert _check_quality_eval_overlap(prompt), prompt


def test_the_living_room_tv_regression_is_the_production_contract():
    from evals.grounding_eval import build_grounding_examples

    row = next(
        example for example in build_grounding_examples()
        if example["metadata"]["grounding_family"] == "grounding_livingroom_tv_present"
    )
    calls = [c for m in row["messages"] for c in m.get("tool_calls") or []]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "HassTurnOn"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "domain": ["media_player"], "name": "TV"
    }
    assert next(message["content"] for message in row["messages"] if message["role"] == "user") == (
        "Turn on the living room media player"
    )
    assert "HassTurnOn" in {tool["function"]["name"] for tool in row["tools"]}
