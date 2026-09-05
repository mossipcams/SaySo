"""Contract checks for the balanced synthetic test-data composer."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_balanced_test_data import DEFAULT_COUNT, build_balanced_test_set


def _user_text(example: dict) -> str:
    content = next(
        message["content"]
        for message in example["messages"]
        if message["role"] == "user"
    )
    if isinstance(content, str):
        return content
    return content[0]["text"]


def test_balanced_test_set_has_exact_mix_and_canonical_shape() -> None:
    assert DEFAULT_COUNT == 2_500

    first = build_balanced_test_set(count=100, seed=1042)
    second = build_balanced_test_set(count=100, seed=1042)

    assert first == second
    assert Counter(
        example["metadata"]["evaluation_category"] for example in first
    ) == {
        "normal_household": 50,
        "paraphrase_conversational": 20,
        "multi_action_exclusion": 10,
        "ambiguity_context": 10,
        "stt_corrupted": 5,
        "refusal_unsupported_clarification": 5,
    }
    assert len({_user_text(example) for example in first}) == len(first)
    rendered_prompts = "\n".join(_user_text(example).casefold() for example in first)
    assert "you to is " not in rendered_prompts
    assert "you to are " not in rendered_prompts
    assert "can you let's " not in rendered_prompts

    for example in first:
        assert example["metadata"]["held_out"] is True
        assert "evals/cases/" not in json.dumps(example)
        for message in example["messages"]:
            for call in message.get("tool_calls") or []:
                arguments = call["function"]["arguments"]
                assert isinstance(arguments, str)
                assert isinstance(json.loads(arguments), dict)
