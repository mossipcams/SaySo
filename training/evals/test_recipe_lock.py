"""Tests for human-locked recipe quality eval gold rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evals.recipe_lock import (  # noqa: E402
    assert_quality_eval_contract,
    build_quality_eval_examples,
    expected_tool_calls,
    locked_specs,
    quality_eval_user_prompts,
    score_quality_gold,
)


def test_locked_specs_cover_all_yes_rows_without_thermostat() -> None:
    specs = locked_specs()
    assert len(specs) == 38
    recipes = {spec["recipe"] for spec in specs}
    assert recipes == {1, 2, 3, 4, 5, 6, 7, 8}
    assert not any("thermostat" in spec["utterance"].casefold() for spec in specs)
    assert sum(spec["recipe"] == 7 for spec in specs) == 9
    assert sum(spec["recipe"] == 8 for spec in specs) == 3


def test_quality_eval_includes_sayso_entity_area_context_for_recipe_seven() -> None:
    examples = build_quality_eval_examples()
    area_rows = [row for row in examples if row["metadata"]["recipe"] == 7]
    assert len(area_rows) == 9
    for example in area_rows:
        system = example["messages"][0]["content"]
        sayso_area = next(
            spec["home"]["sayso_entity_area"]
            for spec in locked_specs()
            if spec["candidate_id"] == example["metadata"]["candidate_id"]
        )
        assert "This SaySo conversation entity area is" in system
        assert sayso_area in system


def test_kitchen_no_lights_row_has_area_unavailable_next_action() -> None:
    example = next(
        row
        for row in build_quality_eval_examples()
        if row["metadata"]["recipe"] == 7 and row["metadata"]["recipe_row"] == "i"
    )
    assert not expected_tool_calls(example)
    final = example["messages"][-1]["content"]
    assert "kitchen has no lights available" in final.casefold()


def test_quality_eval_labels_match_locked_tool_and_no_call_behavior() -> None:
    examples = build_quality_eval_examples()
    by_id = {example["metadata"]["candidate_id"]: example for example in examples}
    for spec in locked_specs():
        example = by_id[spec["candidate_id"]]
        assert_quality_eval_contract(example)
        expected_calls = spec["expected"].get("calls") or []
        actual_calls = expected_tool_calls(example)
        assert len(actual_calls) == len(expected_calls)
        for expected_call, actual_call in zip(expected_calls, actual_calls):
            assert actual_call["function"]["name"] == expected_call["name"]
            assert json.loads(actual_call["function"]["arguments"]) == expected_call["arguments"]
        if not expected_calls:
            assert not any(message.get("tool_calls") for message in example["messages"])


def test_score_quality_gold_checks_no_call_when_expected() -> None:
    clarify = next(
        row
        for row in build_quality_eval_examples()
        if row["metadata"]["recipe"] == 7 and row["metadata"]["recipe_row"] == "b"
    )
    assert score_quality_gold(clarify, [])["pass"]
    bad = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "HassTurnOn",
                        "arguments": json.dumps({"name": "Kitchen Sink Cool Light", "domain": ["light"]}),
                    },
                }
            ],
        }
    ]
    scored = score_quality_gold(clarify, bad)
    assert not scored["no_call_when_expected"]
    assert not scored["pass"]


def test_locked_recipe_three_through_six_utterances_match_human_lock() -> None:
    expected = {
        3: {
            "a": "Open the patio blinds",
            "b": "Lock the patio door",
            "c": "turn on office main light",
            "d": "unlock joe's guest room door lock",
            "e": "Turn on Joe's Kitchen Light",
            "f": "Close O'Malley's Study Blinds",
            "g": "Turn off Kids' Room Light",
            "h": "Turn on Kitchen North Light",
        },
        4: {
            "a": (
                "Set Kitchen Ceiling Cool Light to 40 percent and turn off Hallway East Outlet, "
                "but leave Office Main Light alone"
            ),
            "b": "Open Patio South Blinds and lock Patio Side Door Lock, but leave Garage West Fan alone",
            "c": (
                "Turn on Nursery East Outlet and turn off Living Room Ceiling Fan, "
                "but leave Joe's Kitchen Light alone"
            ),
            "d": (
                "Open Joe's Workshop Blinds, close Primary Bedroom Corner Garage Door, "
                "and lock Patio Side Door Lock, but leave Garage Ceiling Fan alone"
            ),
        },
        5: {
            "garage_van": "Turn on the garage west van",
            "a": "Uh unlock basement door lok please",
            "b": "tern on office main lite",
            "c": "close the patio south blends",
            "d": "lok joe's guest room door",
            "e": "turn off basement van",
        },
        6: {
            "a": "Check the status of Patio South Blinds",
            "b": "Is the Workshop West Fan running?",
            "c": "What's Joe's Guest Room Door Lock doing?",
            "d": "Is Kitchen North Light off?",
        },
    }
    by_recipe: dict[int, dict[str, str]] = {recipe: {} for recipe in expected}
    for spec in locked_specs():
        recipe = spec["recipe"]
        if recipe in expected:
            by_recipe[recipe][spec["recipe_row"]] = spec["utterance"]
    assert by_recipe == expected
    assert "Turn on the kitchen light" not in by_recipe[3].values()


def test_quality_eval_prompts_are_unique_enough_for_leakage_guard() -> None:
    prompts = quality_eval_user_prompts()
    assert len(prompts) >= 30
    assert "Turn on Office Main Light" in prompts
    assert "Play music in the garage" in prompts
