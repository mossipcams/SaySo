"""Tests for Home-LLM pile loading and pile-based generation."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.home_llm_piles import SAMPLE_FACTORS, SMALL_FACTORS, generate_pile_examples
from generators.piles import DatasetPiles


def test_piles_load_english_fixtures() -> None:
    piles = DatasetPiles.load()
    assert len(piles.pile_of_specific_actions) > 500
    assert len(piles.stacks_of_device_names["light"]) > 50
    assert len(piles.pile_of_system_prompts) >= 3


def test_small_generation_produces_thousands() -> None:
    examples = list(generate_pile_examples(seed=7, factors=SMALL_FACTORS))
    assert len(examples) >= 3000


def test_sample_generation_is_smaller_than_small() -> None:
    small = len(list(generate_pile_examples(seed=1, factors=SMALL_FACTORS)))
    sample = len(list(generate_pile_examples(seed=1, factors=SAMPLE_FACTORS)))
    assert sample < small
    assert sample >= 500


def test_diversity_device_names_and_templates() -> None:
    examples = list(generate_pile_examples(seed=99, factors=SAMPLE_FACTORS))
    device_names: set[str] = set()
    templates: set[str] = set()
    for example in examples:
        templates.add(example["metadata"]["template_family"])
        for message in example["messages"]:
            for call in message.get("tool_calls") or []:
                args = json.loads(call["function"]["arguments"])
                name = args.get("name")
                if isinstance(name, str):
                    device_names.add(name)
    assert len(device_names) > 50
    assert len(templates) >= 5


def test_no_forbidden_labels_or_keys() -> None:
    allowed_tools = {
        "GetDateTime",
        "GetLiveContext",
        "HassCancelAllTimers",
        "HassFanSetSpeed",
        "HassLightSet",
        "HassTurnOff",
        "HassTurnOn",
    }
    for example in generate_pile_examples(seed=11, factors=SAMPLE_FACTORS):
        tool_names = {tool["function"]["name"] for tool in example["tools"]}
        assert tool_names == allowed_tools
        for tool in example["tools"]:
            assert tool["type"] == "function"
        blob = json.dumps(example)
        assert "<tool_call>" not in blob
        for message in example["messages"]:
            for call in message.get("tool_calls") or []:
                args = json.loads(call["function"]["arguments"])
                assert "entity_id" not in args
                assert "target_device" not in args
                assert "service" not in args


def test_non_v1_pile_rows_are_dropped_with_reasons() -> None:
    from generators.piles import GenerationStats

    stats = GenerationStats()
    list(generate_pile_examples(seed=5, factors=SAMPLE_FACTORS, stats=stats))
    assert stats.dropped
    assert stats.dropped.get("non_v1_domain:climate", 0) > 0
    assert stats.dropped.get("non_v1_domain:vacuum", 0) > 0


def test_refusal_and_failure_families_present() -> None:
    families = Counter(
        ex["metadata"]["template_family"]
        for ex in generate_pile_examples(seed=3, factors=SAMPLE_FACTORS)
    )
    assert families["pile_refusal"] > 0
    assert families["pile_failure"] > 0
    assert families["pile_status"] > 0


def test_small_factors_covers_all_v1_tool_calls() -> None:
    required = {
        "GetDateTime",
        "GetLiveContext",
        "HassCancelAllTimers",
        "HassFanSetSpeed",
        "HassLightSet",
        "HassTurnOff",
        "HassTurnOn",
    }
    seen: set[str] = set()
    for example in generate_pile_examples(seed=7, factors=SMALL_FACTORS):
        for message in example["messages"]:
            for call in message.get("tool_calls") or []:
                seen.add(call["function"]["name"])
    assert required.issubset(seen)
