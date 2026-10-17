from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_preflight import _row, _run  # noqa: E402


def test_withheld_tool_refusal_is_not_a_conflict(tmp_path: Path) -> None:
    # Same request: an action when the tool is offered, a refusal when it is withheld.
    refusal = _row(positive=False)
    refusal["metadata"]["unavailable_tools"] = ["HassMediaPause"]
    report, errors = _run(tmp_path, [_row(), refusal], limits={"max_conflict_rate": 0.1})
    assert report["conflict_rate"] == 0
    assert not any("conflicting example rate" in error for error in errors)


def test_same_tools_different_labels_still_conflict(tmp_path: Path) -> None:
    report, _ = _run(tmp_path, [_row(), _row(positive=False)], limits={"max_conflict_rate": 0.1})
    assert report["conflict_rate"] > 0


def test_follow_up_turn_is_not_a_conflict_with_its_clarify_row(tmp_path: Path) -> None:
    clarify = _row(positive=False)
    follow_up = _row()
    follow_up["messages"][1:1] = [
        {"role": "assistant", "content": "Which TV?"},
        {"role": "user", "content": "the living room one"},
    ]
    report, _ = _run(tmp_path, [clarify, follow_up], limits={"max_conflict_rate": 0.1})
    assert report["conflict_rate"] == 0
