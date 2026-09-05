"""Weighted quota sampling for tiers, capabilities, operations, and home sizes."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from generators.capability_registry import (
    CAPABILITIES,
    HOME_SIZE_WEIGHTS,
    MIN_OPERATION_COVERAGE,
    MIN_OPERATION_FRACTION,
    TIER1_CAPABILITY_WEIGHTS,
    TIER2_CAPABILITIES,
    TIER3_CAPABILITIES,
    TIER_PROPORTIONS,
    CapabilitySpec,
    SupportLevel,
)


def weighted_choice(weights: dict[str, int], rng: random.Random) -> str:
    keys = list(weights.keys())
    values = [weights[k] for k in keys]
    return rng.choices(keys, weights=values, k=1)[0]


def sample_home_size(weights: dict[int, int], rng: random.Random) -> int:
    sizes = list(weights.keys())
    values = [weights[s] for s in sizes]
    return rng.choices(sizes, weights=values, k=1)[0]


def compute_tier_quotas(count: int, proportions: dict[int, float] | None = None) -> dict[int, int]:
    props = proportions or TIER_PROPORTIONS
    quotas: dict[int, int] = {}
    remaining = count
    tiers = sorted(props.keys())
    for tier in tiers[:-1]:
        q = int(count * props[tier])
        quotas[tier] = q
        remaining -= q
    quotas[tiers[-1]] = remaining
    return quotas


def compute_capability_quotas(tier_quota: int, tier: int) -> dict[str, int]:
    if tier == 1:
        weights = TIER1_CAPABILITY_WEIGHTS
    elif tier == 2:
        weights = {name: 1 for name in TIER2_CAPABILITIES}
    else:
        weights = {name: 1 for name in TIER3_CAPABILITIES}
    total_weight = sum(weights.values())
    quotas: dict[str, int] = {}
    remaining = tier_quota
    keys = list(weights.keys())
    for key in keys[:-1]:
        q = max(1, tier_quota * weights[key] // total_weight)
        quotas[key] = q
        remaining -= q
    quotas[keys[-1]] = max(0, remaining)
    return quotas


def quota_operation_names(cap: CapabilitySpec) -> list[str]:
    """Operations eligible for accepted-row quota allocation."""
    names: list[str] = []
    for op in cap.operations:
        if op.support in {SupportLevel.SUPPORTED, SupportLevel.PARTIAL}:
            names.append(op.name)
        elif op.name == "query_state" and op.tool_name:
            names.append(op.name)
    return names or [cap.operations[0].name]


def compute_operation_quotas(cap_quota: int, cap: CapabilitySpec) -> dict[str, int]:
    """Allocate per-operation targets with a meaningful minimum share."""
    op_names = quota_operation_names(cap)
    if cap_quota <= 0:
        return {name: 0 for name in op_names}
    min_per = max(MIN_OPERATION_COVERAGE, int(cap_quota * MIN_OPERATION_FRACTION))
    min_per = min(min_per, cap_quota // max(len(op_names), 1))
    quotas = {name: min_per for name in op_names}
    total = sum(quotas.values())
    if total > cap_quota:
        factor = cap_quota / total
        quotas = {k: max(1, int(v * factor)) for k, v in quotas.items()}
        while sum(quotas.values()) > cap_quota:
            richest = max(quotas, key=quotas.get)
            quotas[richest] -= 1
        while sum(quotas.values()) < cap_quota:
            poorest = min(quotas, key=quotas.get)
            quotas[poorest] += 1
        return quotas
    remainder = cap_quota - total
    per_extra, leftover = divmod(remainder, len(op_names))
    for name in op_names:
        quotas[name] += per_extra
    for name in op_names[:leftover]:
        quotas[name] += 1
    return quotas


def build_quota_targets(
    count: int,
    proportions: dict[int, float] | None = None,
) -> dict[str, dict[Any, int]]:
    """Return requested accepted-row targets keyed by tier, capability, and operation."""
    tier_targets = compute_tier_quotas(count, proportions)
    cap_targets: dict[tuple[int, str], int] = {}
    op_targets: dict[tuple[int, str, str], int] = {}
    for tier, tier_q in sorted(tier_targets.items()):
        cap_quotas = compute_capability_quotas(tier_q, tier)
        for cap_name, cap_q in cap_quotas.items():
            cap = CAPABILITIES[cap_name]
            cap_targets[(tier, cap_name)] = cap_q
            op_quotas = compute_operation_quotas(cap_q, cap)
            for op_name, op_q in op_quotas.items():
                op_targets[(tier, cap_name, op_name)] = op_q
    return {
        "tier": tier_targets,
        "capability": cap_targets,
        "operation": op_targets,
    }


def build_quota_plan(count: int, seed: int, proportions: dict[int, float] | None = None) -> list[dict[str, Any]]:
    """Return one generation slot per requested accepted row (no padding/cloning)."""
    targets = build_quota_targets(count, proportions)
    rng = random.Random(seed)
    slots: list[dict[str, Any]] = []
    index = 0
    for (tier, cap_name, op_name), op_q in sorted(targets["operation"].items()):
        for _ in range(op_q):
            slots.append(
                {
                    "index": index,
                    "tier": tier,
                    "capability": cap_name,
                    "operation": op_name,
                    "home_size": sample_home_size(HOME_SIZE_WEIGHTS, rng),
                }
            )
            index += 1
    if len(slots) != count:
        raise ValueError(f"quota plan length {len(slots)} != requested count {count}")
    rng.shuffle(slots)
    return slots


class QuotaTracker:
    """Track accepted-row progress against requested tier/capability/operation quotas."""

    def __init__(self, count: int, seed: int, proportions: dict[int, float] | None = None) -> None:
        self.count = count
        self.targets = build_quota_targets(count, proportions)
        self.rng = random.Random(seed)
        self.accepted_tier: Counter[int] = Counter()
        self.accepted_cap: Counter[tuple[int, str]] = Counter()
        self.accepted_op: Counter[tuple[int, str, str]] = Counter()

    def accepted_total(self) -> int:
        return sum(self.accepted_tier.values())

    def is_complete(self) -> bool:
        return self.accepted_total() >= self.count

    def next_slot(self) -> dict[str, Any]:
        """Pick the operation bucket with the largest remaining shortfall."""
        candidates: list[tuple[int, tuple[int, str, str]]] = []
        for key, target in self.targets["operation"].items():
            shortfall = target - self.accepted_op[key]
            if shortfall > 0:
                candidates.append((shortfall, key))
        if not candidates:
            raise RuntimeError("quota tracker has no remaining shortfall buckets")
        max_shortfall = max(s for s, _ in candidates)
        top = [key for shortfall, key in candidates if shortfall == max_shortfall]
        tier, cap_name, op_name = self.rng.choice(top)
        return {
            "index": self.accepted_total(),
            "tier": tier,
            "capability": cap_name,
            "operation": op_name,
            "home_size": sample_home_size(HOME_SIZE_WEIGHTS, self.rng),
        }

    def record_accept(self, row: dict[str, Any]) -> None:
        meta = row.get("metadata") or {}
        tier = meta.get("tier")
        cap = meta.get("capability")
        op = meta.get("operation")
        if tier is None or cap is None or op is None:
            raise ValueError("accepted row missing tier/capability/operation metadata")
        self.accepted_tier[tier] += 1
        self.accepted_cap[(tier, cap)] += 1
        self.accepted_op[(tier, cap, op)] += 1

    def shortfall(self) -> dict[str, dict[str, int]]:
        gaps: dict[str, dict[str, int]] = {"tier": {}, "capability": {}, "operation": {}}
        for tier, target in self.targets["tier"].items():
            got = self.accepted_tier.get(tier, 0)
            if got < target:
                gaps["tier"][str(tier)] = target - got
        for key, target in self.targets["capability"].items():
            got = self.accepted_cap.get(key, 0)
            if got < target:
                gaps["capability"][str(key)] = target - got
        for key, target in self.targets["operation"].items():
            got = self.accepted_op.get(key, 0)
            if got < target:
                gaps["operation"][str(key)] = target - got
        return gaps

    def verify_complete(self) -> None:
        gaps = self.shortfall()
        if any(gaps.values()):
            raise RuntimeError(f"accepted-row quota shortfall: {gaps}")

    def summary(self) -> dict[str, Any]:
        return {
            "requested": {
                "tier": {str(k): v for k, v in self.targets["tier"].items()},
                "capability": {str(k): v for k, v in self.targets["capability"].items()},
                "operation": {str(k): v for k, v in self.targets["operation"].items()},
            },
            "achieved": {
                "tier": {str(k): v for k, v in sorted(self.accepted_tier.items())},
                "capability": {str(k): v for k, v in sorted(self.accepted_cap.items())},
                "operation": {str(k): v for k, v in sorted(self.accepted_op.items())},
            },
            "shortfall": self.shortfall(),
        }


def quota_shortfall(plan: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> dict[str, Any]:
    planned = Counter((s["tier"], s["capability"], s["operation"]) for s in plan)
    actual = Counter(
        (
            r.get("metadata", {}).get("tier"),
            r.get("metadata", {}).get("capability"),
            r.get("metadata", {}).get("operation"),
        )
        for r in accepted
    )
    gaps: dict[str, int] = {}
    for key, target in planned.items():
        got = actual.get(key, 0)
        if got < target:
            gaps[str(key)] = target - got
    return gaps
