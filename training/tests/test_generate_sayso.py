"""Tests for SaySo example generator."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_sayso_examples import MIX, generate_examples  # noqa: E402


def test_generator_emits_valid_jsonl_shape() -> None:
    example = next(generate_examples(1, seed=7))
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
    examples = list(generate_examples(200, seed=123))
    assert len(examples) == 200
    templates = Counter(ex["metadata"]["template_family"] for ex in examples)
    assert len(templates) >= 5


def test_no_entity_id_in_tool_arguments() -> None:
    for example in generate_examples(50, seed=99):
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

    for example in generate_examples(30, seed=55):
        if example["metadata"]["template_family"] == "multi_off_on":
            assert convert_entry(example, seed=1) is not None
            return
    raise AssertionError("no multi_off_on example in sample")


def test_mix_weights_documented() -> None:
    total = sum(weight for _cat, weight in MIX)
    assert total == 100
