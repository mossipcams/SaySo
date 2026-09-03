"""Validate Axolotl smoke/production configs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
FIXTURES = ROOT / "fixtures"


def _load(name: str) -> dict:
    with (CONFIGS / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _fixture_has_train_on_turn(path: Path) -> bool:
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for message in record.get("messages", []):
            if message.get("role") == "assistant":
                assert message.get("train_on_turn") is True
            elif message.get("role") in {"system", "user", "tool"}:
                assert message.get("train_on_turn") is False
        return True
    return False


def test_smoke_config_uses_fixture_dataset() -> None:
    cfg = _load("functiongemma-270m-smoke.yml")
    dataset_path = cfg["datasets"][0]["path"]
    assert "fixtures" in dataset_path
    assert cfg["bf16"] is False
    assert cfg["flash_attention"] is False
    assert cfg.get("fp16") is True
    assert cfg["seed"] == 42
    assert cfg["datasets"][0].get("message_field_training") == "train_on_turn"
    assert cfg["datasets"][0].get("roles_to_train") == []


def test_smoke_fixture_assistant_only_training() -> None:
    smoke_path = FIXTURES / "sayso_axolotl_smoke.jsonl"
    assert smoke_path.exists(), "run adapt_dataset to create smoke fixture"
    assert _fixture_has_train_on_turn(smoke_path)


def test_prod_config_gtx1070_safe() -> None:
    cfg = _load("functiongemma-270m-prod.yml")
    assert cfg["bf16"] is False
    assert cfg["flash_attention"] is False
    assert cfg.get("fp16") is True
    assert "sayso_train" in cfg["datasets"][0]["path"]
    assert cfg["datasets"][0].get("message_field_training") == "train_on_turn"


def test_prod_config_has_eval_dataset() -> None:
    cfg = _load("functiongemma-270m-prod.yml")
    test_sets = cfg.get("test_datasets") or []
    assert any("sayso_val" in entry.get("path", "") for entry in test_sets)


def test_base_model_is_functiongemma_270m() -> None:
    cfg = _load("functiongemma-270m.yml")
    assert cfg["base_model"] == "google/functiongemma-270m-it"
    assert cfg["chat_template"] == "jinja"
