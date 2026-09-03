"""Tests for SaySo example generator."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.home_llm_piles import SAMPLE_FACTORS, generate_pile_examples  # noqa: E402


def _examples(count: int, seed: int = 42):
    produced = 0
    for example in generate_pile_examples(seed=seed, factors=SAMPLE_FACTORS):
        yield example
        produced += 1
        if produced >= count:
            break


def test_generator_emits_valid_jsonl_shape() -> None:
    example = next(_examples(1, seed=7))
    assert "messages" in example
    assert "tools" in example
    assert "metadata" in example
    meta = example["metadata"]
    assert "template_family" in meta
    assert "phrasing_family" in meta
    assert "seed" in meta
    roles = [m["role"] for m in example["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_generator_mix_categories() -> None:
    examples = list(generate_pile_examples(seed=123, factors=SAMPLE_FACTORS))
    templates = Counter(ex["metadata"]["template_family"] for ex in examples)
    assert len(templates) >= 5


def test_no_entity_id_in_tool_arguments() -> None:
    for example in _examples(50, seed=99):
        for message in example["messages"]:
            for tc in message.get("tool_calls") or []:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    parsed = json.loads(args)
                else:
                    parsed = args
                assert "entity_id" not in parsed


def test_multi_example_adapts() -> None:
    from adapters.home_llm_v2 import convert_entry

    for example in generate_pile_examples(seed=55, factors=SAMPLE_FACTORS):
        if example["metadata"]["template_family"] == "pile_templated_multi":
            assert convert_entry(example, seed=1) is not None
            return
    raise AssertionError("no pile_templated_multi example in sample")


def test_generated_examples_adapt_without_schema_validation_failures() -> None:
    from adapters.home_llm_v2 import convert_entry
    from adapters.schema import RejectionStats

    stats = RejectionStats()
    converted = 0
    for example in _examples(200, seed=42):
        if convert_entry(example, seed=42, stats=stats) is not None:
            converted += 1
    assert stats.counts.get("schema_validation_failed", 0) == 0
    assert converted > 0


def test_generated_tools_are_v1_subset_with_function_envelope() -> None:
    allowed = {
        "GetDateTime",
        "GetLiveContext",
        "HassCancelAllTimers",
        "HassFanSetSpeed",
        "HassLightSet",
        "HassTurnOff",
        "HassTurnOn",
    }
    for example in _examples(40, seed=11):
        names = {tool["function"]["name"] for tool in example["tools"]}
        assert names == allowed
        for tool in example["tools"]:
            assert tool["type"] == "function"
