"""Tests for weighted sampling."""

from __future__ import annotations

from collections import Counter

from pathlib import Path

from generators.capability_registry import TIER1_CAPABILITY_WEIGHTS, TIER_PROPORTIONS
from generators.config import GeneratorConfig
from generators.pipeline import run_generation
from generators.coverage import classify_row, row_calls
from generators.gold import gold_matches_family
from generators.tools import namespaced_tool_name
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


def test_generation_hits_family_allocations() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = GeneratorConfig.from_yaml(root / "configs/generation/smoke.yaml", repo_root=root.parent)
    cfg.count = 120
    cfg.seed = 4040
    cfg.area_distribution_path = None
    result = run_generation(cfg)
    summary = result["stats"]["quota"]
    for family, need in summary["requested"].items():
        assert summary["achieved_family"][family] >= need, family


def test_generation_family_outcomes_match_slots() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = GeneratorConfig.from_yaml(root / "configs/generation/smoke.yaml", repo_root=root.parent)
    cfg.count = 120
    cfg.seed = 4040
    cfg.area_distribution_path = None
    result = run_generation(cfg)
    for row in result["rows"]:
        family = row["metadata"].get("family")
        if not family or family == "area":
            continue
        facets = classify_row(row)
        calls = row_calls(row)
        if facets["outcome"] == "action":
            expected = {"kind": "action", "calls": calls, "unavailable_tools": []}
        elif facets["outcome"] == "status":
            expected = {"kind": "status", "calls": calls, "unavailable_tools": []}
        else:
            expected = {
                "kind": "no_action",
                "response": row["metadata"].get("no_action_reason"),
                "calls": [],
                "unavailable_tools": row["metadata"].get("unavailable_tools") or [],
            }
        assert gold_matches_family(expected, family), (
            family,
            facets["outcome"],
            row["metadata"].get("no_action_reason"),
        )


def test_generation_unavailable_rows_withhold_namespaced_tool() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = GeneratorConfig.from_yaml(root / "configs/generation/smoke.yaml", repo_root=root.parent)
    cfg.count = 120
    cfg.seed = 4040
    cfg.area_distribution_path = None
    rows = run_generation(cfg)["rows"]
    unavailable = [row for row in rows if row["metadata"].get("family") == "unavailable"]

    assert unavailable
    for row in unavailable:
        withheld = row["metadata"]["unavailable_tools"]
        offered = {tool["function"]["name"] for tool in row["tools"]}
        assert withheld
        assert all(namespaced_tool_name(tool) not in offered for tool in withheld)


def test_quota_tracker_reports_requested_and_achieved() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = GeneratorConfig.from_yaml(root / "configs/generation/smoke.yaml", repo_root=root.parent)
    cfg.count = 120
    cfg.seed = 9090
    cfg.area_distribution_path = None
    result = run_generation(cfg)
    quota = result["stats"]["quota"]
    assert quota["requested"]
    assert quota["achieved_family"]
    assert quota["total"] == cfg.count
