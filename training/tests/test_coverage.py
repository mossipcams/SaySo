"""Coverage distribution tests."""

from __future__ import annotations

from collections import Counter

from generators.config import GeneratorConfig
from generators.pipeline import run_generation


def test_accepted_rows_span_capabilities() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = GeneratorConfig.from_yaml(root / "configs/generation/smoke.yaml", repo_root=root.parent)
    cfg.count = 120
    cfg.seed = 77
    cfg.area_distribution_path = None
    result = run_generation(cfg)
    caps = Counter(r["metadata"]["capability"] for r in result["rows"])
    assert len(caps) >= 5
