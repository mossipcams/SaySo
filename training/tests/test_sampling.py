"""Tests for weighted sampling."""

from __future__ import annotations

from collections import Counter

from generators.capability_registry import TIER1_CAPABILITY_WEIGHTS, TIER_PROPORTIONS
from generators.config import GeneratorConfig
from generators.pipeline import run_generation
from generators.sampling import (
    QuotaTracker,
    build_quota_plan,
    build_quota_targets,
    compute_tier_quotas,
)


def test_quota_plan_length() -> None:
    plan = build_quota_plan(400, seed=1)
    assert len(plan) == 400


def test_tier_quotas_sum_to_count() -> None:
    quotas = compute_tier_quotas(1000)
    assert sum(quotas.values()) == 1000


def test_plan_includes_multiple_tiers() -> None:
    plan = build_quota_plan(500, seed=2)
    tiers = Counter(slot["tier"] for slot in plan)
    assert len(tiers) >= 2


def test_quota_targets_match_count() -> None:
    targets = build_quota_targets(400)
    assert sum(targets["tier"].values()) == 400
    assert sum(targets["operation"].values()) == 400


def test_tier1_covers_weight_is_six() -> None:
    assert TIER1_CAPABILITY_WEIGHTS["covers"] == 6


def test_generation_hits_tier_proportions_within_tolerance() -> None:
    count = 400
    tolerance = 0.05
    result = run_generation(GeneratorConfig(count=count, seed=4040, paraphrase_enabled=False))
    tiers = Counter(row["metadata"]["tier"] for row in result["rows"])
    for tier, proportion in TIER_PROPORTIONS.items():
        expected = count * proportion
        actual = tiers[tier]
        assert abs(actual / count - proportion) <= tolerance, (
            f"tier {tier}: expected ~{proportion:.0%}, got {actual / count:.1%} ({actual}/{count})"
        )


def test_quota_tracker_reports_requested_and_achieved() -> None:
    result = run_generation(GeneratorConfig(count=200, seed=9090, paraphrase_enabled=False))
    quota = result["stats"]["quota"]
    assert quota["requested"]["tier"]
    assert quota["achieved"]["tier"]
    assert not any(quota["shortfall"].values())
