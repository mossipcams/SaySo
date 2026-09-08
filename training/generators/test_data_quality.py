"""Behavioral regressions from the failed 40k run (colocated, not in tests/)."""

import random
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.config import GeneratorConfig
from generators.duplicates import DuplicateTracker
from generators.pipeline import generate_row
from generators.pipeline import _unique_no_action_hint
from generators.scenarios import build_scenario
from generators.labels import scenario_to_spec, render_example
from generators.utterances import expand_utterance
from generators.validate import validate_row, validate_spec
from generators.pipeline import run_generation


def test_request_case_does_not_reveal_call_decision():
    for difficulty in ("ordinary", "unsupported"):
        cases = set()
        for index in range(60):
            with patch("generators.pipeline.pick_robustness", return_value=difficulty):
                row, _ = generate_row(
                    {"index": index, "capability": "lights", "operation": "turn_on", "home_size": 8},
                    GeneratorConfig(), random.Random(index), excluded=set(),
                    dup_tracker=DuplicateTracker(),
                )
            if row:
                request = next(m["content"] for m in row["messages"] if m["role"] == "user")
                cases.add(request[0].isupper())
        assert cases == {False, True}, (difficulty, cases)


def test_unavailable_request_preserves_the_requested_device_type():
    spec = {"capability": "climate", "operation": "set_temperature",
            "home": {"sayso_entity_area": "Workshop"},
            "expected": {"kind": "no_action", "response": "area_unavailable"}}
    for seed in range(20):
        request = _unique_no_action_hint(spec, random.Random(seed))
        assert "Workshop" in request and "thermostat" in request.lower(), request


def test_unsupported_means_requested_tool_is_actually_unavailable():
    scenario = build_scenario(index=5, seed=19, capability="climate",
                              operation="set_temperature", home_size=8,
                              robustness="unsupported")
    expected = scenario["expected"]
    assert expected.get("unavailable_tools") == ["HassClimateSetTemperature"]
    spec = scenario_to_spec(scenario)
    spec["utterance"] = _unique_no_action_hint(spec, random.Random(7))
    row = render_example(spec)
    assert "HassClimateSetTemperature" not in {t["function"]["name"] for t in row["tools"]}
    assert not any(m.get("tool_calls") for m in row["messages"])
    assert scenario["target_entity"]["name"] in spec["utterance"]


def test_multi_action_and_exclusion_keep_every_target():
    for capability in ("lights", "scripts"):
        scenario = build_scenario(index=7, seed=29, capability=capability,
                                  operation="run" if capability == "scripts" else "turn_on",
                                  home_size=64, targeting="multiple", robustness="exclusion")
        spec = scenario_to_spec(scenario)
        spec["utterance"] = expand_utterance({**spec, "category": "clean_direct"})
        assert len(spec["expected"]["calls"]) >= 2
        assert len(spec["target_names"]) == len(spec["expected"]["calls"])
        for name in spec["target_names"] + spec["excluded_names"]:
            assert name.lower() in spec["utterance"].lower(), spec["utterance"]
        assert "leave" in spec["utterance"].lower()
        assert validate_row(spec) is None


def test_multi_action_includes_different_device_capabilities():
    scenario = build_scenario(index=7, seed=29, capability="lights", operation="turn_on",
                              home_size=64, targeting="multiple", robustness="multi_action")
    assert len({e["capability"] for e in scenario["target_entities"]}) > 1


def test_area_request_keeps_area_and_setting():
    spec = {"expected": {"kind": "action", "calls": [
        {"name": "HassLightSet", "arguments": {"area": "Office", "domain": ["light"], "brightness": 35}}
    ]}, "target_names": [], "category": "clean_direct"}
    request = expand_utterance(spec)
    assert "Office" in request and "35" in request and "light" in request


def test_training_requests_use_aliases_and_conversational_wording():
    aliases = conversational = 0
    for index in range(80):
        scenario = build_scenario(index=index, seed=32, capability="lights", operation="turn_on",
                                  home_size=16, robustness="alias_distractor")
        aliases += bool(scenario.get("spoken_targets"))
        with patch("generators.pipeline.pick_robustness", return_value="ordinary"):
            row, _ = generate_row(
                {"index": index, "capability": "lights", "operation": "turn_on", "home_size": 8},
                GeneratorConfig(), random.Random(index), excluded=set(), dup_tracker=DuplicateTracker(),
            )
        if row:
            user = next(m["content"].lower() for m in row["messages"] if m["role"] == "user")
            conversational += any(word in user for word in ("please", "could you", "can you"))
    assert aliases >= 40, aliases
    assert conversational >= 15, conversational


def test_absence_claim_cannot_contradict_the_home():
    spec = {"home": {"entities": [{"name": "Workshop Thermostat", "domain": "climate", "area": "Workshop"}]},
            "expected": {"kind": "no_action", "response": "area_unavailable", "calls": [],
                         "unavailable": {"area": "workshop", "type": "thermostats"}}}
    assert validate_spec(spec) == "contradictory_absence"


def test_behavior_tags_describe_the_accepted_rows():
    rows = run_generation(GeneratorConfig(count=400, seed=77))["rows"]
    for row in rows:
        metadata = row["metadata"]
        calls = [c for m in row["messages"] for c in m.get("tool_calls", [])]
        if metadata["category"] == "multi_action":
            assert len(calls) >= 2
        if metadata["category"] == "exclusion":
            assert metadata.get("excluded_names")
            assert len(calls) >= 2


def test_audit_rejects_label_dependent_casing():
    from generators.audit import audit_rows
    rows = run_generation(GeneratorConfig(count=400, seed=17))["rows"]
    for row in rows:
        user = next(m for m in row["messages"] if m["role"] == "user")
        text = user["content"]
        called = any(m.get("tool_calls") for m in row["messages"])
        user["content"] = text[:1].upper() + text[1:] if called else text.lower()
    with pytest.raises(ValueError, match="casing"):
        audit_rows(rows, expected_count=400)


def test_stt_noise_does_not_claim_an_unchanged_request():
    from generators.stt_noise import apply_stt_noise
    request = "Run Morning Routine"
    changed, kind = apply_stt_noise(request, random.Random(1), force_transform=True)
    assert changed != request or kind is None


def test_area_color_keeps_its_operation():
    from generators.tools import build_area_call
    call = build_area_call("lights", "set_color", "Office", rng=random.Random(1))
    assert "color" in call["arguments"] and "brightness" not in call["arguments"]


def test_polite_status_requests_are_grammatical():
    from generators.utterances import vary_training_utterance
    for seed in range(30):
        request = vary_training_utterance("what is the status of Kitchen Light", random.Random(seed))
        assert not any(bad in request for bad in ("please what", "could you what", "can you what"))


def test_polite_variants_do_not_bypass_eval_exclusion():
    from generators.pipeline import _check_quality_eval_overlap, _quality_eval_prompts
    for prompt in _quality_eval_prompts():
        assert _check_quality_eval_overlap(f"Can you {prompt} for me?")
        assert _check_quality_eval_overlap(f"Please tell me {prompt}")


def test_generation_reports_independently_audited_behavior():
    result = run_generation(GeneratorConfig(count=100, seed=17))
    assert result["stats"]["quality_audit"]["rows"] == 100


def test_timer_requests_use_the_named_timer_argument():
    for operation in ("pause", "status"):
        scenario = build_scenario(index=9, seed=51, capability="timers", operation=operation, home_size=8)
        call = scenario["expected"]["calls"][0]
        assert call["arguments"].get("name")
        spec = scenario_to_spec(scenario)
        spec["utterance"] = expand_utterance(spec)
        assert call["arguments"]["name"] in spec["utterance"]
        assert validate_row(spec) is None


def test_multi_actions_do_not_label_state_queries_as_done():
    for index in range(50):
        scenario = build_scenario(index=index, seed=29, capability="lights", operation="turn_on",
                                  home_size=64, targeting="multiple", robustness="multi_action")
        assert all(c["name"] != "GetLiveContext" for c in scenario["expected"]["calls"])


def test_exclusion_validation_rejects_a_missing_excluded_name():
    from generators.validate import validate_utterance
    spec = {"expected": {"kind": "action", "calls": []},
            "excluded_names": ["Workshop Lamp"],
            "utterance": "turn on Kitchen Light but leave the other one alone"}
    assert validate_utterance(spec) == "missing_exclusion"


def test_homes_have_plausible_rooms_devices_and_aliases():
    from generators.homes import generate_home
    outdoor = {"Garden", "Backyard", "Front Yard", "Driveway", "Patio", "Deck", "Porch", "Balcony"}
    for seed in range(60):
        home = generate_home(seed, 64, random.Random(seed))
        assert len({e["area"] for e in home["entities"]}) <= 14
        floors = {}
        for entity in home["entities"]:
            area = entity["area"]
            assert floors.setdefault(area, entity["floor"]) == entity["floor"]
            if entity["domain"] in {"media_player", "climate", "todo"}:
                assert area not in outdoor, entity
            for alias in entity["aliases"]:
                words = alias.lower().split()
                assert all(a != b for a, b in zip(words, words[1:])), alias


def test_names_describe_devices_and_routines_instead_of_random_qualities():
    from generators.homes import _random_entity_name
    for capability in ("lights", "scripts", "scenes", "todo_lists", "climate"):
        for seed in range(100):
            name = _random_entity_name(capability, "Kitchen", 0, seed, random.Random(seed))
            words = set(name.lower().split())
            assert not words & {"old", "spare", "second", "smart", "new", "portable"}, name
            if capability == "scripts":
                assert "script" not in words, name
            if capability == "scenes":
                assert "scene" not in words, name


def test_ordinary_speech_changes_sentence_structure_and_preserves_settings():
    from generators.utterances import vary_training_utterance
    on = [vary_training_utterance("turn on Kitchen Lamp", random.Random(seed)) for seed in range(80)]
    brightness = [vary_training_utterance("set Kitchen Lamp brightness to 35 percent", random.Random(seed)) for seed in range(80)]
    status = [vary_training_utterance("what is the status of Kitchen Lamp", random.Random(seed)) for seed in range(80)]
    assert any("Kitchen Lamp on" in request for request in on)
    assert any("dim Kitchen Lamp to 35 percent" in request for request in brightness)
    assert any("what's" in request or "check" in request for request in status)
    assert all("Kitchen Lamp" in request and "35" in request for request in brightness)


def test_routine_names_match_the_room_activity():
    from generators.homes import _random_entity_name
    for seed in range(50):
        scene = _random_entity_name("scenes", "Bathroom", 0, seed, random.Random(seed))
        assert not any(word in scene.lower() for word in ("movie", "dinner", "reading")), scene
        shopping = _random_entity_name("todo_lists", "Kitchen", 0, seed, random.Random(seed))
        assert "Kitchen" not in shopping, shopping


def test_large_homes_are_mostly_lights_and_plugs_not_thermostats():
    from collections import Counter
    from generators.homes import generate_home
    home = generate_home(12, 64, random.Random(19))
    counts = Counter(e["domain"] for e in home["entities"])
    assert counts["climate"] <= 2
    assert counts["light"] + counts["switch"] >= 38


def test_possessive_names_come_from_one_household():
    from generators.homes import generate_home
    home = generate_home(4, 64, random.Random(4))
    owners = {e["name"].split()[0].casefold() for e in home["entities"] if "'" in e["name"]}
    assert len(owners) <= 2


def test_speech_avoids_conflicting_verbs_and_supplies_kelvin_units():
    from generators.utterances import vary_training_utterance
    for seed in range(60):
        rng = random.Random(seed)
        assert "turn on" not in vary_training_utterance("run Bathroom Lights Off", rng)
        assert "dim" not in vary_training_utterance("set Kitchen Lamp brightness to 100 percent", rng)
        assert "kelvin" in vary_training_utterance("set Kitchen Lamp color temperature to 2700", rng)


def test_eval_name_exclusion_ignores_capitalization():
    from generators.homes import _random_entity_name
    with patch("generators.homes._eval_entity_names", return_value=frozenset({"kitchen lamp"})), patch("generators.homes._roles", return_value=("Lamp",)):
        for seed in range(50):
            name = _random_entity_name("lights", "Kitchen", 0, seed, random.Random(seed))
            assert name.casefold() != "kitchen lamp"


def test_group_requests_use_spoken_device_types():
    from generators.utterances import vary_training_utterance
    for seed in range(20):
        request = vary_training_utterance("set the climates in Hallway temperature to 70 degrees", random.Random(seed))
        assert "thermostats" in request and "climates" not in request
        request = vary_training_utterance("turn off the switchs in Kitchen", random.Random(seed))
        assert "outlets" in request and "switchs" not in request


def test_held_out_names_do_not_leak_through_aliases():
    from generators.homes import generate_home
    with patch("generators.homes._eval_entity_names", return_value=frozenset({"living room tv", "kitchen light"})):
        home = generate_home(9, 64, random.Random(9))
        assert not {"living room tv", "kitchen light"} & {
            alias.casefold() for entity in home["entities"] for alias in entity["aliases"]
        }


def test_generic_script_requests_clarify_without_hidden_area_assumptions():
    for seed in range(20):
        scenario = build_scenario(index=seed, seed=seed, capability="scripts", operation="run",
                                  home_size=32, targeting="area", robustness="ambiguity")
        assert scenario["expected"]["kind"] == "no_action"
        assert scenario["expected"]["response"] == "clarify"
        spec = scenario_to_spec(scenario)
        spec["utterance"] = _unique_no_action_hint(spec, random.Random(seed))
        assert "routine" in spec["utterance"]
        assert render_example(spec)["messages"][-1]["content"] == "Which routine did you mean?"


@pytest.mark.parametrize("operation,tool", [
    ("start", "HassStartTimer"), ("cancel_all", "HassCancelAllTimers"),
    ("pause", "HassPauseTimer"), ("status", "HassTimerStatus"),
])
def test_timer_semantics_do_not_depend_on_device_inventory(operation, tool):
    for robustness in ("ordinary", "ambiguity", "unsupported"):
        scenario = build_scenario(index=1, seed=19, capability="timers", operation=operation,
                                  home_size=8, targeting="area", robustness=robustness)
        expected = scenario["expected"]
        if robustness == "unsupported":
            assert expected["response"] == "unsupported"
            assert expected["unavailable_tools"] == [tool]
        else:
            assert expected["kind"] == "action", expected
            assert expected["calls"][0]["name"] == tool
        spec = scenario_to_spec(scenario)
        spec["utterance"] = (_unique_no_action_hint(spec, random.Random(0))
                             if robustness == "unsupported" else expand_utterance(spec))
        assert validate_row(spec) is None
        row = render_example(spec)
        assert (tool in {t["function"]["name"] for t in row["tools"]}) == (robustness != "unsupported")


def test_validator_rejects_false_timer_refusals_and_unsupported_claims():
    scenario = build_scenario(index=1, seed=19, capability="timers", operation="start",
                              home_size=8, robustness="unsupported")
    spec = scenario_to_spec(scenario)
    assert validate_spec(spec) is None
    spec["expected"] = {"kind": "no_action", "response": "area_unavailable", "calls": [],
                        "unavailable": {"area": "sitting room", "type": "timers"}}
    assert validate_spec(spec) == "timer_inventory_refusal"
    spec["expected"] = {"kind": "no_action", "response": "unsupported", "calls": []}
    assert validate_spec(spec) == "unsupported_without_withheld_tool"


@pytest.mark.parametrize("offered", [True, False])
def test_final_audit_rejects_timer_inventory_refusals(offered):
    from generators.audit import audit_rows
    from adapters.schema import v2_openai_tools
    row = {"metadata": {"candidate_id": "bad-timer", "capability": "timers", "operation": "start"},
           "messages": [{"role": "user", "content": "start a 20 minute Sitting Room timer"},
                        {"role": "assistant", "content": "The sitting room has no devices available."}],
           "tools": [t for t in v2_openai_tools() if offered and t["function"]["name"] == "HassStartTimer"]}
    with pytest.raises(ValueError, match="timer"):
        audit_rows([row])
    if offered:
        row["messages"][-1]["content"] = "I can't do that with the available Home Assistant tools."
        with pytest.raises(ValueError, match="timer"):
            audit_rows([row])


def test_frozen_realistic_eval_prompts_stay_held_out():
    from generators.pipeline import _check_quality_eval_overlap
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "realistic_eval_20260908_v2.json"
    for case in json.loads(fixture.read_text())["cases"]:
        prompt = next(m["content"] for m in case["messages"] if m["role"] == "user")
        for variant in (prompt, prompt.upper(), f"Please {prompt}", f"Could you {prompt} for me?"):
            assert _check_quality_eval_overlap(variant), variant


def test_unambiguous_status_query_keeps_state_and_status_label():
    from generators.gold import gold_from_scenario
    entity = {"name": "Entryway Ceiling Lights", "capability": "lights", "domain": "light",
              "area": "Entryway", "state": "on"}
    scenario = {"home": {"entities": [entity], "sayso_entity_area": "Entryway"},
                "capability": "lights", "operation": "query_state", "robustness": "ambiguity"}
    expected = gold_from_scenario(scenario, random.Random(1))
    assert expected["kind"] == "status"
    assert expected["state"] == "on"


def test_final_audit_rejects_state_queries_labeled_done():
    from generators.audit import audit_rows
    scenario = build_scenario(index=2, seed=19, capability="lights", operation="query_state", home_size=8)
    spec = scenario_to_spec(scenario)
    spec["utterance"] = expand_utterance(spec)
    row = render_example(spec)
    row["messages"][-1]["content"] = "Done."
    with pytest.raises(ValueError, match="state query"):
        audit_rows([row])


def test_canonical_targets_do_not_collide_with_other_entity_aliases():
    from generators.homes import generate_home
    for seed in range(30):
        home = generate_home(seed, 64, random.Random(seed))
        for entity in home["entities"]:
            assert not any(entity["name"].casefold() in {alias.casefold() for alias in other["aliases"]}
                           for other in home["entities"] if other is not entity), entity["name"]


@pytest.mark.parametrize("operation", ["turn_off", "query_state", "set_brightness"])
def test_unique_room_device_requests_keep_generic_wording_and_operation(operation):
    for index in range(100):
        scenario = build_scenario(index=index, seed=17, capability="lights", operation=operation,
                                  home_size=8, targeting="area", robustness="ambiguity")
        if scenario["expected"]["kind"] not in {"action", "status"}:
            continue
        with patch("generators.pipeline.build_scenario", return_value=scenario), patch("generators.pipeline.pick_robustness", return_value="ambiguity"):
            row, reason = generate_row(
                {"index": index, "capability": "lights", "operation": operation, "home_size": 8},
                GeneratorConfig(), random.Random(index), excluded=set(), dup_tracker=DuplicateTracker())
        if reason == "quality_eval_overlap":
            continue
        assert row, reason
        user = next(m["content"].lower() for m in row["messages"] if m["role"] == "user")
        assert f"the {scenario['home']['sayso_entity_area'].lower()} light" in user, user
        if operation == "turn_off":
            assert "off" in user and "on" not in user.split()
        if operation == "query_state":
            assert "status" in user or "state" in user
        return
    pytest.fail("no unique room-device scenario accepted")


def test_floor_requests_do_not_silently_narrow_to_one_room():
    from generators.gold import gold_from_scenario
    scenario = build_scenario(index=7, seed=29, capability="lights", operation="set_brightness",
                              home_size=64, targeting="floor")
    expected = scenario["expected"]
    assert "floor" in expected["calls"][0]["arguments"]
    assert "area" not in expected["calls"][0]["arguments"]
    spec = scenario_to_spec(scenario)
    spec["utterance"] = expand_utterance(spec)
    assert scenario["floor"].lower() in spec["utterance"].lower()
    assert "brightness" in spec["utterance"]
    assert validate_row(spec) is None
    spec["utterance"] = "set the lights brightness to 35 percent"
    assert validate_row(spec) == "missing_expected_scope"


def test_floor_requests_use_spoken_device_types():
    from generators.utterances import vary_training_utterance
    for technical, spoken in (("climates", "thermostats"), ("switchs", "outlets"), ("covers", "blinds")):
        request = vary_training_utterance(f"turn off the {technical} on Upstairs", random.Random(1))
        assert spoken in request and technical not in request
