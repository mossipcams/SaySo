"""Tests for v3 quality eval generation CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evals.recipe_lock import quality_eval_user_prompts  # noqa: E402
from evals.v3_quality import _normalized, gold_user_prompts  # noqa: E402
from generate_v3_quality_eval import main as generate_main  # noqa: E402


def test_generate_cli_writes_gold_and_shadow_to_tmp_path(tmp_path: Path, monkeypatch) -> None:
    gold_out = tmp_path / "gold.jsonl"
    shadow_out = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_v3_quality_eval.py",
            "--gold-out",
            str(gold_out),
            "--shadow-out",
            str(shadow_out),
            "--shadow-count",
            "100",
        ],
    )
    assert generate_main() == 0
    gold_rows = [json.loads(line) for line in gold_out.read_text().splitlines() if line.strip()]
    shadow_rows = [json.loads(line) for line in shadow_out.read_text().splitlines() if line.strip()]
    assert len(gold_rows) >= 25
    assert len(shadow_rows) == 100
    gold_prompts = {
        _normalized(next(message["content"] for message in row["messages"] if message.get("role") == "user"))
        for row in gold_rows
    }
    shadow_prompts = {
        _normalized(next(message["content"] for message in row["messages"] if message.get("role") == "user"))
        for row in shadow_rows
    }
    recipe_prompts = {_normalized(text) for text in quality_eval_user_prompts()}
    assert not gold_prompts.intersection(recipe_prompts)
    assert not shadow_prompts.intersection(recipe_prompts)
    assert not gold_prompts.intersection(shadow_prompts)


def test_v3_40k_trl_config_points_at_base_rendered_train() -> None:
    text = (ROOT / "configs" / "lfm25-230m-synthetic-v3-40k-trl.yml").read_text()
    assert "model_name_or_path: /srv/models/LFM2.5-230M-Base" in text
    assert "output_dir: /srv/training-runs/SaySo-LFM2.5-230M-v3-40k" in text
    assert "sayso_train_v3_40k_render.jsonl" in text
    assert "num_train_epochs: 2" in text
    assert "assistant_only_loss: true" in text
    assert "lora_r: 32" in text
    assert "use_rslora: true" in text
