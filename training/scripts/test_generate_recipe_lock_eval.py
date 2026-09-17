"""Tests for recipe-lock eval generation and canonical TRL rendering."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evals.recipe_lock import build_quality_eval_examples  # noqa: E402
from generators.labels import render_for_trl  # noqa: E402
from generate_recipe_lock_eval import main as generate_main  # noqa: E402


def test_render_for_trl_parses_arguments_and_flattens_content() -> None:
    example = build_quality_eval_examples()[0]
    rendered = render_for_trl(example)
    assistant = next(message for message in rendered["messages"] if message.get("tool_calls"))
    args = assistant["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict)
    assert isinstance(rendered["messages"][0]["content"], str)
    assert isinstance(rendered["messages"][1]["content"], str)


def test_generate_cli_writes_eval_to_tmp_path(tmp_path: Path, monkeypatch) -> None:
    eval_out = tmp_path / "quality.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_recipe_lock_eval.py",
            "--eval-out",
            str(eval_out),
        ],
    )
    assert generate_main() == 0
    eval_rows = [json.loads(line) for line in eval_out.read_text().splitlines() if line.strip()]
    assert len(eval_rows) == 38
