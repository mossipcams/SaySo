"""Tests for recipe-lock eval generation and deterministic 10k train output."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_dataset import (  # noqa: E402
    CATEGORY_WEIGHTS,
    build_deterministic_train_examples,
    render_for_trl,
)
from evals.recipe_lock import build_quality_eval_examples, quality_eval_user_prompts  # noqa: E402
from generate_recipe_lock_eval import main as generate_main  # noqa: E402


def test_build_deterministic_train_is_repeatable_and_balanced() -> None:
    excluded = quality_eval_user_prompts()
    first = build_deterministic_train_examples(100, seed=20260904, excluded_utterances=excluded)
    second = build_deterministic_train_examples(100, seed=20260904, excluded_utterances=excluded)
    assert first == second
    assert len(first) == 100
    categories = Counter(row["metadata"]["category"] for row in first)
    assert categories == CATEGORY_WEIGHTS


def test_deterministic_train_excludes_quality_eval_user_prompts() -> None:
    excluded = quality_eval_user_prompts()
    rows = build_deterministic_train_examples(200, seed=20260904, excluded_utterances=excluded)
    train_prompts = {
        next(message["content"] for message in row["messages"] if message.get("role") == "user")
        for row in rows
    }
    assert not train_prompts.intersection(excluded)


def test_render_for_trl_parses_arguments_and_flattens_content() -> None:
    example = build_quality_eval_examples()[0]
    rendered = render_for_trl(example)
    assistant = next(message for message in rendered["messages"] if message.get("tool_calls"))
    args = assistant["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict)
    assert isinstance(rendered["messages"][0]["content"], str)
    assert isinstance(rendered["messages"][1]["content"], str)


def test_generate_cli_writes_eval_and_train_to_tmp_path(tmp_path: Path, monkeypatch) -> None:
    eval_out = tmp_path / "quality.jsonl"
    train_out = tmp_path / "train.jsonl"
    render_out = tmp_path / "train_render.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_recipe_lock_eval.py",
            "--eval-out",
            str(eval_out),
            "--train-out",
            str(train_out),
            "--train-render-out",
            str(render_out),
            "--count",
            "100",
        ],
    )
    assert generate_main() == 0
    eval_rows = [json.loads(line) for line in eval_out.read_text().splitlines() if line.strip()]
    train_rows = [json.loads(line) for line in train_out.read_text().splitlines() if line.strip()]
    render_rows = [json.loads(line) for line in render_out.read_text().splitlines() if line.strip()]
    assert len(eval_rows) == 38
    assert len(train_rows) == 100
    assert len(render_rows) == 100
    eval_prompts = {
        next(message["content"] for message in row["messages"] if message.get("role") == "user")
        for row in eval_rows
    }
    train_prompts = {
        next(message["content"] for message in row["messages"] if message.get("role") == "user")
        for row in train_rows
    }
    assert not train_prompts.intersection(eval_prompts)


def test_first_training_trl_config_points_at_base_and_rendered_train() -> None:
    text = (ROOT / "configs" / "lfm25-230m-synthetic-v2-trl.yml").read_text()
    assert "model_name_or_path: /srv/models/LFM2.5-230M-Base" in text
    assert "output_dir: /srv/training-runs/SaySo-LFM2.5-230M-Base-First" in text
    assert "sayso_train_first_10000_render.jsonl" in text
    assert "assistant_only_loss: true" in text
    assert "lora_r: 32" in text
