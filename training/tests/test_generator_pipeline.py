"""Tests for generation pipeline."""

from __future__ import annotations

from generators.config import GeneratorConfig
from generators.pipeline import run_generation


def test_smoke_generation_small_n() -> None:
    config = GeneratorConfig(count=100, seed=99, paraphrase_enabled=False)
    result = run_generation(config)
    rows = result["rows"]
    assert len(rows) == 100
    assert result["stats"]["accepted"] == 100
    for row in rows:
        assert row["tools"]
        assert row["messages"][0]["role"] == "system"
        assert any(m["role"] == "user" for m in row["messages"])


def test_generation_is_deterministic() -> None:
    cfg = GeneratorConfig(count=50, seed=123)
    first = run_generation(cfg)["rows"]
    second = run_generation(cfg)["rows"]
    assert first == second
