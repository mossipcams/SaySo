"""Tests for the raw-completion v3 scorer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_v3_rawparse import to_tool_calls  # noqa: E402
from evals.v3_quality import build_gold_examples, expected_tool_calls, score_quality_gold  # noqa: E402


def test_apostrophe_names_survive_the_parser() -> None:
    calls, error = to_tool_calls("HassTurnOn(name='O'Malley's Lamp', domain=['light'])")
    assert error is None
    assert calls[0]["function"]["name"] == "HassTurnOn"
    assert "O'Malley's Lamp" in calls[0]["function"]["arguments"]


def test_prose_scores_as_no_call_not_as_an_error() -> None:
    calls, error = to_tool_calls("I can't do that.")
    assert calls == []
    assert error is not None


def test_a_correct_completion_scores_pass_for_every_gold_row() -> None:
    """Replay each gold label as Python-call text and score it back to a pass."""
    for example in build_gold_examples():
        import json

        rendered = ", ".join(
            "{}({})".format(
                call["function"]["name"],
                ", ".join(
                    f"{key}={value!r}"
                    for key, value in json.loads(call["function"]["arguments"]).items()
                ),
            )
            for call in expected_tool_calls(example)
        )
        calls, _ = to_tool_calls(rendered)
        scored = score_quality_gold(example, [{"role": "assistant", "content": rendered, "tool_calls": calls}])
        assert scored["pass"], (example["metadata"]["candidate_id"], rendered, scored)
