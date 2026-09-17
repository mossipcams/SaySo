"""Contract checks for the balanced synthetic test-data composer."""

from __future__ import annotations

import json
import sys
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
    assert len({_user_text(example) for example in first}) == len(first)

    for example in first:
        assert example["metadata"]["held_out"] is True
        assert example["metadata"]["evaluation_category"]
        assert example["metadata"]["target_area_source"] in {
            "satellite_fallback", "explicit_area", "named_target", "ambiguous", "missing_area",
        }
        assert "evals/cases/" not in json.dumps(example)
        for message in example["messages"]:
            for call in message.get("tool_calls") or []:
                arguments = call["function"]["arguments"]
                assert isinstance(arguments, str)
                assert isinstance(json.loads(arguments), dict)
