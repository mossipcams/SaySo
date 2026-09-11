"""Weighted quota sampling for tiers, capabilities, operations, and home sizes.

Quotas are accounted on what a row *teaches*, not on the metadata it carries. A
refusal tagged ``media_players``/``turn_on`` used to fill that operation's quota,
so an operation could reach its target without a single row that calls the tool.
Positive supervision and refusals now have separate budgets in every bucket.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from generators.capability_registry import (
    CAPABILITIES,
    DEFAULT_NEGATIVE_RATE,
    HOME_SIZE_WEIGHTS,
    MIN_OPERATION_COVERAGE,
    MIN_OPERATION_FRACTION,
    TIER1_CAPABILITY_WEIGHTS,
    TIER2_CAPABILITIES,
    TIER3_CAPABILITIES,
    TIER_PROPORTIONS,
    CapabilitySpec,
    SupportLevel,
    operation_spec,
)
from generators.coverage import expected_tool

# Rows per unavailable operation, as a share of its capability's quota. These
# buckets exist only to teach the refusal, so they stay small but never empty.
REFUSAL_OPERATION_SHARE: float = 0.05


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
    """Operations eligible for positive accepted-row quota allocation."""
    names: list[str] = []
    for op in cap.operations:
        if op.support in {SupportLevel.SUPPORTED, SupportLevel.PARTIAL}:
            names.append(op.name)
        elif op.name == "query_state" and op.tool_name:
            names.append(op.name)
    return names or [cap.operations[0].name]


def refusal_operation_names(cap: CapabilitySpec) -> list[str]:
    """Operations whose only correct supervision is a refusal."""
    supported = set(quota_operation_names(cap))
    return [op.name for op in cap.operations
            if op.support is SupportLevel.UNAVAILABLE and op.name not in supported]


def _split_evenly(cap_quota: int, op_names: list[str]) -> dict[str, int]:
    """Allocate per-operation targets with a meaningful minimum share."""
    if cap_quota <= 0 or not op_names:
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


def compute_operation_quotas(cap_quota: int, cap: CapabilitySpec) -> dict[str, int]:
    """Split a capability quota across its supported and its refusal-only operations."""
    supported = quota_operation_names(cap)
    refusals = refusal_operation_names(cap)
    if cap_quota <= 0:
        return {name: 0 for name in (*supported, *refusals)}
    quotas: dict[str, int] = {name: 0 for name in refusals}
    budget = cap_quota
    if refusals and cap_quota >= len(supported) + len(refusals):
        per = max(1, int(cap_quota * REFUSAL_OPERATION_SHARE))
        per = min(per, max(1, (cap_quota - len(supported)) // len(refusals)))
        quotas = {name: per for name in refusals}
        budget -= sum(quotas.values())
    quotas.update(_split_evenly(budget, supported))
    return quotas


def _positive_floor(capability: str, operation: str, target: int, negative_rate: float) -> int:
    """Rows in this bucket that must carry real positive supervision."""
    op = operation_spec(capability, operation)
    if target <= 0 or op is None or op.support is SupportLevel.UNAVAILABLE:
        return 0
    return max(1, int(round(target * (1.0 - negative_rate))))


def build_quota_targets(
    count: int,
    proportions: dict[int, float] | None = None,
    *,
    negative_rate: float = DEFAULT_NEGATIVE_RATE,
) -> dict[str, dict[Any, int]]:
    """Requested accepted-row targets keyed by tier, capability, and operation.

    ``operation`` is the total rows for a bucket and still sums to ``count``;
    ``positive`` and ``negative`` are the floors within each bucket.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if not 0.0 <= negative_rate < 1.0:
        raise ValueError(f"negative_rate must be in [0, 1), got {negative_rate}")
    props = proportions or TIER_PROPORTIONS
    if abs(sum(props.values()) - 1.0) > 0.01 or any(value < 0 for value in props.values()):
        raise ValueError(f"tier proportions must be non-negative and sum to 1: {props}")

    tier_targets = compute_tier_quotas(count, props)
    cap_targets: dict[tuple[int, str], int] = {}
    op_targets: dict[tuple[int, str, str], int] = {}
    positive: dict[tuple[int, str, str], int] = {}
    negative: dict[tuple[int, str, str], int] = {}
    for tier, tier_q in sorted(tier_targets.items()):
        cap_quotas = compute_capability_quotas(tier_q, tier)
        for cap_name, cap_q in cap_quotas.items():
            cap = CAPABILITIES[cap_name]
            cap_targets[(tier, cap_name)] = cap_q
            for op_name, op_q in compute_operation_quotas(cap_q, cap).items():
                key = (tier, cap_name, op_name)
                op_targets[key] = op_q
                positive[key] = _positive_floor(cap_name, op_name, op_q, negative_rate)
                negative[key] = op_q if positive[key] == 0 else 0
    targets = {
        "tier": tier_targets,
        "capability": cap_targets,
        "operation": op_targets,
        "positive": positive,
        "negative": negative,
    }
    _verify_feasible(count, targets)
    return targets


def _verify_feasible(count: int, targets: dict[str, dict[Any, int]]) -> None:
    """Fail before the generation loop, not after exhausting max_attempts."""
    total = sum(targets["operation"].values())
    if total != count:
        raise ValueError(f"operation quotas sum to {total}, expected {count}")
    unbuildable = []
    for (tier, cap_name, op_name), target in targets["operation"].items():
        if target <= 0 or targets["positive"][(tier, cap_name, op_name)] <= 0:
            continue
        if expected_tool(cap_name, op_name) is None:
            unbuildable.append(f"{cap_name}/{op_name}")
    if unbuildable:
        raise ValueError(
            "quota requires positive rows for operations with no tool mapping: "
            + ", ".join(sorted(unbuildable))
        )


def uncovered_operations(targets: dict[str, dict[Any, int]]) -> list[str]:
    """Supported operations this run is too small to reach. Recorded, not silent."""
    return sorted(
        f"{cap_name}/{op_name}"
        for (tier, cap_name, op_name), target in targets["operation"].items()
        if target <= 0
        and (operation_spec(cap_name, op_name) or CAPABILITIES[cap_name].operations[0]).support
        is not SupportLevel.UNAVAILABLE
    )


def minimum_count_for_full_coverage(proportions: dict[int, float] | None = None) -> int:
    """Smallest count at which every supported operation gets at least one row."""
    props = proportions or TIER_PROPORTIONS
    needed = 1
    for tier, proportion in props.items():
        if proportion <= 0:
            continue
        if tier == 1:
            weights = TIER1_CAPABILITY_WEIGHTS
        elif tier == 2:
            weights = {name: 1 for name in TIER2_CAPABILITIES}
        else:
            weights = {name: 1 for name in TIER3_CAPABILITIES}
        total_weight = sum(weights.values())
        for cap_name, weight in weights.items():
            cap = CAPABILITIES[cap_name]
            ops = len(quota_operation_names(cap)) + len(refusal_operation_names(cap))
            share = proportion * weight / total_weight
            needed = max(needed, int(-(-ops // share)) if share else needed)
    return needed


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
    """Track accepted-row progress against tier/capability/operation quotas.

    Every bucket carries three counters: total rows, rows with real positive
    supervision, and rows that refuse, clarify or report an absence. A row is
    accepted only when it advances one of them (:meth:`wants`).
    """

    def __init__(
        self,
        count: int,
        seed: int,
        proportions: dict[int, float] | None = None,
        *,
        negative_rate: float = DEFAULT_NEGATIVE_RATE,
    ) -> None:
        self.count = count
        self.negative_rate = negative_rate
        self.targets = build_quota_targets(count, proportions, negative_rate=negative_rate)
        self.rng = random.Random(seed)
        self.accepted_tier: Counter[int] = Counter()
        self.accepted_cap: Counter[tuple[int, str]] = Counter()
        self.accepted_op: Counter[tuple[int, str, str]] = Counter()
        self.accepted_positive: Counter[tuple[int, str, str]] = Counter()
        self.accepted_negative: Counter[tuple[int, str, str]] = Counter()
        self.accepted_outcome: Counter[str] = Counter()
        self.accepted_targeting: Counter[str] = Counter()
        self.accepted_tool: Counter[str] = Counter()

    def accepted_total(self) -> int:
        return sum(self.accepted_tier.values())

    def is_complete(self) -> bool:
        return self.accepted_total() >= self.count

    def _key(self, meta: dict[str, Any]) -> tuple[int, str, str]:
        tier, cap, op = meta.get("tier"), meta.get("capability"), meta.get("operation")
        if tier is None or cap is None or op is None:
            raise ValueError("accepted row missing tier/capability/operation metadata")
        return (tier, cap, op)

    def _negative_allowance(self, key: tuple[int, str, str]) -> int:
        """Refusals a bucket may hold: its floor, or the slack left by its positive floor."""
        target = self.targets["operation"].get(key, 0)
        floor = self.targets["negative"].get(key, 0)
        return max(floor, target - self.targets["positive"].get(key, 0))

    def wants(self, facets: dict[str, Any]) -> str | None:
        """Rejection reason, or None when this row advances a quota."""
        key = self._key(facets)
        if key not in self.targets["operation"]:
            return "quota_bucket_unknown"
        target = self.targets["operation"][key]
        if self.accepted_op[key] >= target:
            return "quota_bucket_full"
        if facets.get("positive"):
            return None
        if self.accepted_negative[key] >= self._negative_allowance(key):
            return "quota_negative_full"
        return None

    def record_accept(self, row: dict[str, Any], facets: dict[str, Any] | None = None) -> None:
        from generators.coverage import classify_row

        if facets is None:
            facets = classify_row(row)
        tier, cap, op = self._key(facets)
        self.accepted_tier[tier] += 1
        self.accepted_cap[(tier, cap)] += 1
        self.accepted_op[(tier, cap, op)] += 1
        if facets.get("positive"):
            self.accepted_positive[(tier, cap, op)] += 1
        else:
            self.accepted_negative[(tier, cap, op)] += 1
        self.accepted_outcome[facets["outcome"]] += 1
        self.accepted_targeting[facets["targeting"]] += 1
        for tool in facets["tools"]:
            self.accepted_tool[tool] += 1

    def shortfall(self) -> dict[str, dict[str, int]]:
        gaps: dict[str, dict[str, int]] = {
            "tier": {}, "capability": {}, "operation": {}, "positive": {}, "negative": {},
        }
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
        for key, target in self.targets["positive"].items():
            got = self.accepted_positive.get(key, 0)
            if got < target:
                gaps["positive"][str(key)] = target - got
        for key, target in self.targets["negative"].items():
            got = self.accepted_negative.get(key, 0)
            if got < target:
                gaps["negative"][str(key)] = target - got
        return gaps

    def verify_complete(self) -> None:
        gaps = self.shortfall()
        if any(gaps.values()):
            raise RuntimeError(f"accepted-row quota shortfall: {gaps}")

    def next_slot(self) -> dict[str, Any]:
        """Pick the bucket with the largest remaining need, positives first."""
        best: list[tuple[int, str, str]] = []
        best_gap = 0
        for key, target in self.targets["operation"].items():
            positive_gap = self.targets["positive"][key] - self.accepted_positive[key]
            negative_gap = self.targets["negative"][key] - self.accepted_negative[key]
            total_gap = target - self.accepted_op[key]
            gap = max(positive_gap, negative_gap, min(total_gap, 1) if total_gap > 0 else 0)
            if gap <= 0:
                continue
            if gap > best_gap:
                best_gap, best = gap, [key]
            elif gap == best_gap:
                best.append(key)
        if not best:
            raise RuntimeError("quota tracker has no remaining shortfall buckets")
        tier, cap_name, op_name = self.rng.choice(sorted(best))
        return {
            "index": self.accepted_total(),
            "tier": tier,
            "capability": cap_name,
            "operation": op_name,
            "home_size": sample_home_size(HOME_SIZE_WEIGHTS, self.rng),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "negative_rate": self.negative_rate,
            "requested": {
                "tier": {str(k): v for k, v in self.targets["tier"].items()},
                "capability": {str(k): v for k, v in self.targets["capability"].items()},
                "operation": {str(k): v for k, v in self.targets["operation"].items()},
                "positive": {str(k): v for k, v in self.targets["positive"].items() if v},
                "negative": {str(k): v for k, v in self.targets["negative"].items() if v},
            },
            "achieved": {
                "tier": {str(k): v for k, v in sorted(self.accepted_tier.items())},
                "capability": {str(k): v for k, v in sorted(self.accepted_cap.items())},
                "operation": {str(k): v for k, v in sorted(self.accepted_op.items())},
                "positive": {str(k): v for k, v in sorted(self.accepted_positive.items())},
                "negative": {str(k): v for k, v in sorted(self.accepted_negative.items())},
                "outcome": dict(sorted(self.accepted_outcome.items())),
                "targeting": dict(sorted(self.accepted_targeting.items())),
                "tool": dict(sorted(self.accepted_tool.items())),
            },
            "uncovered_operations": uncovered_operations(self.targets),
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
