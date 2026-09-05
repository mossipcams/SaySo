"""Validate LFM2.5-230M Axolotl configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from adapters.lfm import LFM_BASE_MODEL, validate_lfm_config_text

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
FIXTURES = ROOT / "fixtures"


def _load(name: str) -> dict:
    with (CONFIGS / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_lfm_smoke_config_points_at_lfm_model_and_fixture() -> None:
    cfg = _load("lfm25-230m-smoke.yml")
    assert cfg["base_model"] == LFM_BASE_MODEL == "LiquidAI/LFM2.5-230M"
    assert cfg["chat_template"] == "tokenizer"
    dataset_path = cfg["datasets"][0]["path"]
    assert "sayso_lfm_smoke.jsonl" in dataset_path
    assert cfg["bf16"] is False
    assert cfg["flash_attention"] is False
    assert cfg.get("fp16") is True
    assert cfg["datasets"][0].get("message_field_training") == "train_on_turn"


def test_lfm_prod_config_gtx1070_safe() -> None:
    cfg = _load("lfm25-230m-prod.yml")
    assert cfg["base_model"] == LFM_BASE_MODEL
    assert cfg["bf16"] is False
    assert cfg["flash_attention"] is False
    assert cfg.get("fp16") is True
    assert "sayso_train" in cfg["datasets"][0]["path"]


def test_lfm_prod_config_has_eval_dataset() -> None:
    cfg = _load("lfm25-230m-prod.yml")
    test_sets = cfg.get("test_datasets") or []
    assert any("sayso_val" in entry.get("path", "") for entry in test_sets)


def test_lfm_configs_do_not_embed_chatml_tool_call_labels() -> None:
    for name in ("lfm25-230m.yml", "lfm25-230m-smoke.yml", "lfm25-230m-prod.yml"):
        text = (CONFIGS / name).read_text(encoding="utf-8")
        validate_lfm_config_text(text)


def test_train_defaults_to_lfm_backend() -> None:
    train_py = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert 'default="lfm"' in train_py
    assert "lfm25-230m" in train_py


def test_lfm_smoke_fixture_exists() -> None:
    assert (FIXTURES / "sayso_lfm_smoke.jsonl").exists()
