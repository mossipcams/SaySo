"""v5 recipe feasibility against grounding and real-home ceilings."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from generators.config import GeneratorConfig
from generators.grounding import training_variants
from generators.planning import verify_feasible

V5_CONFIG = ROOT / "configs" / "generation" / "full_sft_v5.yaml"


def test_v5_recipe_is_feasible() -> None:
    cfg = GeneratorConfig.from_yaml(V5_CONFIG, repo_root=REPO)
    verify_feasible(
        cfg.count,
        cfg.allocations,
        near_duplicate_limit=cfg.near_duplicate_limit,
        area_required={},
        grounding_variants=len(training_variants()),
        grounding_rate=cfg.grounding_rate,
        real_home_path=cfg.real_home_path,
        real_home_rate=cfg.real_home_rate,
    )
    assert abs(sum(cfg.allocations.values()) - 1.0) < 0.02
    assert cfg.grounding_rate == 0.08
    assert cfg.stt_log_rate == 0.10


def test_v5_recipe_grounding_does_not_starve_families() -> None:
    """40k v5 deadlocked: grounding stole slots until only two families moved."""
    from dataclasses import replace

    from generators.grounding import GROUNDING_CARRIER_FAMILIES
    from generators.pipeline import run_generation

    cfg = replace(GeneratorConfig.from_yaml(V5_CONFIG, repo_root=REPO), count=1500)
    result = run_generation(cfg)
    rows = result["rows"]
    assert len(rows) == 1500
    achieved = result["stats"]["quota"]["achieved_family"]
    assert len([f for f, n in achieved.items() if n]) > 10, achieved
    grounded = [r["metadata"] for r in rows if r["metadata"].get("grounding_family")]
    assert len(grounded) >= 0.05 * len(rows), len(grounded)
    assert {m.get("family") for m in grounded} <= GROUNDING_CARRIER_FAMILIES
