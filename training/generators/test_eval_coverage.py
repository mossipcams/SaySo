"""Coverage checks for the runtime scenarios represented in training data."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.home_llm_piles import SAMPLE_FACTORS, generate_pile_examples


def _examples() -> list[dict]:
    return list(generate_pile_examples(seed=42, factors=SAMPLE_FACTORS))


def test_sample_data_has_ambiguous_requests_that_ask_for_clarification() -> None:
    examples = [
        example
        for example in _examples()
        if example["metadata"]["template_family"] == "pile_ambiguity"
    ]

    assert len(examples) >= 3
    for example in examples:
        assert not any(message.get("tool_calls") for message in example["messages"])
        final_text = example["messages"][-1]["content"][0]["text"].lower()
        assert any(word in final_text for word in ("which", "what", "specify"))


def test_sample_data_has_partial_multi_action_failures() -> None:
    examples = [
        example
        for example in _examples()
        if example["metadata"]["template_family"] == "pile_partial_failure"
    ]

    assert len(examples) >= 3
    for example in examples:
        calls = [
            call
            for message in example["messages"]
            for call in message.get("tool_calls") or []
        ]
        assert len(calls) >= 2
        tool_blob = json.dumps(example["messages"])
        assert '"result": "Failed"' in tool_blob
        assert example["messages"][-1]["role"] == "assistant"
        assert example["messages"][-1]["content"][0]["text"]


def test_sample_data_expands_sparse_context_tool_families() -> None:
    families = Counter(
        example["metadata"]["template_family"] for example in _examples()
    )

    assert families["pile_synth_datetime"] >= 6
    assert families["pile_synth_cancel_all_timers"] >= 6
