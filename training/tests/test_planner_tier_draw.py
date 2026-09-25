"""Regression: the ordinary capability draw must never hit an action-less tier.

Tier 3 capabilities (lawn_mowers, todo_lists, buttons) have no tool-backed
action operations, so drawing tier 3 in the ordinary family previously crashed
``_pick_capability_operation`` with ``IndexError: Cannot choose from an empty
sequence`` (empty ``tier_caps``). The fix restricts the tier draw to tiers
that actually contain an actionable capability.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.capability_registry import CAPABILITIES, TIER_PROPORTIONS
from generators.planning import _action_ops, _pick_capability_operation


def test_tier3_capabilities_have_no_action_ops():
    """Document the registry fact the fix relies on."""
    tier3 = [cap for cap in CAPABILITIES.values() if cap.tier == 3]
    assert tier3, "expected tier-3 capabilities in the registry"
    for cap in tier3:
        assert not _action_ops(cap), f"tier-3 capability {cap.name!r} gained action ops"


def test_ordinary_pick_never_raises_and_always_actionable():
    # Pre-fix, a 0.05-weight tier-3 draw hit within the first ~20 draws
    # almost always; 50 seeds x 200 draws = 10k draws covers it with margin.
    for seed in range(50):
        rng = random.Random(seed)
        for _ in range(200):
            cap_name, operation, tier = _pick_capability_operation("ordinary", rng)
            assert cap_name in CAPABILITIES
            assert tier in TIER_PROPORTIONS
            assert _action_ops(CAPABILITIES[cap_name]), (
                f"ordinary draw picked action-less capability {cap_name!r} (tier {tier})"
            )
            assert operation != "query_state"
            assert operation in _action_ops(CAPABILITIES[cap_name])


def test_ordinary_draw_tier_distribution_skips_empty_tiers():
    rng = random.Random(20260923)
    seen: dict[int, int] = {}
    for _ in range(20_000):
        _, _, tier = _pick_capability_operation("ordinary", rng)
        seen[tier] = seen.get(tier, 0) + 1
    # Tiers 1:2 weighting must follow TIER_PROPORTIONS (0.80:0.15), and a
    # tier with no actionable capability must never be drawn.
    total = sum(seen.values())
    p1 = seen.get(1, 0) / total
    assert 0.78 <= p1 <= 0.88, f"tier-1 share {p1:.3f} deviates from 0.842"
    assert seen.get(3, 0) == 0, "tier 3 (no action ops) was drawn by ordinary"
