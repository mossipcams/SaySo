"""Tests for v3 quality eval gold and shadow rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evals.recipe_lock import quality_eval_user_prompts as recipe_lock_prompts  # noqa: E402
from evals.v3_quality import (  # noqa: E402
    _normalized,
    assert_quality_eval_contract,
    build_gold_examples,
    build_shadow_examples,
    build_shadow_specs,
    excluded_train_prompts,
    expected_tool_calls,
    gold_specs,
    gold_user_prompts,
    score_quality_gold,
)


def test_gold_specs_cover_v3_domains_and_ordinary_categories() -> None:
    specs = gold_specs()
    categories = {spec["category"] for spec in specs}
    required = {
        "climate_setpoint",
        "media_play",
        "media_pause",
        "media_volume",
        "media_mute",
        "timer_start",
        "timer_pause",
        "timer_status",
        "timer_cancel",
        "vacuum_start",
        "vacuum_return",
        "vacuum_clean_area",
        "scene_activate",
        "script_run",
        "ordinary_on",
        "ordinary_off",
        # 12.3% of training calls; no v3 suite covered either before 2026-09-06
        "light_brightness",
        "fan_speed",
        "status",
        "ambiguity",
        "unsupported_no_action",
    }
    assert required.issubset(categories)
    assert len(specs) >= 25


def test_gold_labels_validate_and_do_not_overlap_recipe_lock_prompts() -> None:
    examples = build_gold_examples()
    recipe_norm = {_normalized(text) for text in recipe_lock_prompts()}
    for example in examples:
        assert_quality_eval_contract(example)
        user = next(message for message in example["messages"] if message.get("role") == "user")
        assert _normalized(str(user["content"])) not in recipe_norm


def test_gold_labels_match_locked_tool_and_no_call_behavior() -> None:
    examples = build_gold_examples()
    by_id = {example["metadata"]["candidate_id"]: example for example in examples}
    for spec in gold_specs():
        example = by_id[spec["candidate_id"]]
        expected_calls = spec["expected"].get("calls") or []
        actual_calls = expected_tool_calls(example)
        assert len(actual_calls) == len(expected_calls)
        for expected_call, actual_call in zip(expected_calls, actual_calls):
            assert actual_call["function"]["name"] == expected_call["name"]
            assert json.loads(actual_call["function"]["arguments"]) == expected_call["arguments"]
        if not expected_calls:
            assert not any(message.get("tool_calls") for message in example["messages"])


def test_shadow_specs_use_fresh_areas_and_do_not_reuse_gold_utterances() -> None:
    gold_norm = {_normalized(text) for text in gold_user_prompts()}
    gold_entities = {entity["name"] for spec in gold_specs() for entity in spec["home"]["entities"]}
    specs = build_shadow_specs(seed=20260906, count=100)
    shadow_norm: set[str] = set()
    for spec in specs:
        assert spec["utterance"]
        norm = _normalized(spec["utterance"])
        assert norm not in gold_norm
        assert norm not in shadow_norm
        shadow_norm.add(norm)
        for entity in spec["home"]["entities"]:
            assert entity["name"] not in gold_entities
        assert spec["home"]["sayso_entity_area"] in {
            "Annex",
            "Atrium",
            "Conservatory",
            "Solarium",
            "Studio",
            "Terrace",
            "Balcony",
            "Cellar",
            "Attic",
            "Porch",
            "Veranda",
            "Courtyard",
        }


def test_shadow_examples_validate_and_mark_metadata() -> None:
    rows = build_shadow_examples(seed=20260906, count=100)
    assert len(rows) == 100
    for row in rows:
        assert row["metadata"]["shadow_eval"] is True
        assert row["metadata"]["v3_quality_shadow"] is True
        assert_quality_eval_contract(row)


def test_shadow_script_rows_call_the_per_script_tool() -> None:
    rows = build_shadow_examples(seed=20260906, count=100)
    script_rows = [row for row in rows if row["metadata"]["category"] == "script_run"]
    assert script_rows
    for row in script_rows:
        offered = {tool["function"]["name"] for tool in row["tools"]}
        for call in expected_tool_calls(row):
            name = call["function"]["name"]
            assert name.endswith("_script"), name
            assert name in offered
            assert json.loads(call["function"]["arguments"]) == {}


def test_light_and_fan_rows_use_the_argument_the_domain_requires() -> None:
    """Run 008's top failure was `temperature=` where `brightness=` was wanted.
    These rows only catch it if the labels themselves are right."""
    rows = build_gold_examples() + build_shadow_examples(seed=20260906, count=100)
    seen = set()
    for row in rows:
        for call in expected_tool_calls(row):
            name = call["function"]["name"]
            if name not in {"HassLightSet", "HassFanSetSpeed"}:
                continue
            seen.add(name)
            args = json.loads(call["function"]["arguments"])
            if name == "HassLightSet":
                assert "brightness" in args and "temperature" not in args, args
                assert args["domain"] == ["light"]
            else:
                assert "percentage" in args and "brightness" not in args, args
                assert args["domain"] == ["fan"]
    assert seen == {"HassLightSet", "HassFanSetSpeed"}


def test_contrast_rows_put_both_a_light_and_a_fan_in_the_home() -> None:
    """A contrast row is only a contrast if the wrong device is also present."""
    contrast = [
        spec
        for spec in gold_specs() + build_shadow_specs(seed=20260906, count=100)
        if spec["subcategory"] == "light_fan_contrast"
    ]
    assert contrast
    for spec in contrast:
        domains = {entity["domain"] for entity in spec["home"]["entities"]}
        assert {"light", "fan"}.issubset(domains), spec["candidate_id"]


def test_contract_rejects_a_target_the_prompt_never_names() -> None:
    example = next(
        row
        for row in build_shadow_examples(seed=20260906, count=100)
        if row["metadata"]["category"] == "climate_setpoint"
    )
    for message in example["messages"]:
        for call in message.get("tool_calls") or []:
            arguments = json.loads(call["function"]["arguments"])
            arguments["name"] = "Nowhere Thermostat"
            call["function"]["arguments"] = json.dumps(arguments)
    with pytest.raises(ValueError, match="absent from the prompt"):
        assert_quality_eval_contract(example)


def test_excluded_train_prompts_include_gold_shadow_and_recipe_lock() -> None:
    excluded = excluded_train_prompts()
    assert {_normalized(text) for text in gold_user_prompts()}.issubset(excluded)
    assert {_normalized(text) for text in recipe_lock_prompts()}.issubset(excluded)
    shadow = build_shadow_examples(seed=20260906, count=100)
    for row in shadow:
        user = next(message for message in row["messages"] if message.get("role") == "user")
        assert _normalized(str(user["content"])) in excluded


def test_score_quality_gold_passes_identical_tool_call_prediction() -> None:
    example = next(row for row in build_gold_examples() if expected_tool_calls(row))
    calls = expected_tool_calls(example)
    actual = [{"role": "assistant", "content": "", "tool_calls": calls}]
    scored = score_quality_gold(example, actual)
    assert scored["pass"]
