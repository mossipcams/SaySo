"""Recipe feasibility, family allocations, and generation slot planning."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from generators.capability_registry import CAPABILITIES, HOME_SIZE_WEIGHTS, TIER_PROPORTIONS
from generators.sampling import sample_home_size

PRIMARY_FAMILIES = (
    "ordinary",
    "follow_up",
    "status",
    "settings",
    "clarify",
    "correction",
    "junk",
    "multi_action",
    "exclusion",
    "unavailable",
    "unsupported",
    "absence",
    "aliases",
)

# Families that teach catalog gaps, absence refusals, or non-command transcripts,
# not real-home names. The real-home balancer must never steer these slots onto
# ``real_home_train``.
REAL_HOME_EXCLUDED_FAMILIES = frozenset({
    "unavailable",
    "unsupported",
    "junk",
    "datetime",
    "absence",
})

FAMILY_ROBUSTNESS: dict[str, str] = {
    "datetime": "datetime",
    "ordinary": "ordinary",
    "follow_up": "ambiguity",
    "status": "ordinary",
    "settings": "ordinary",
    "multi_action": "multi_action",
    "clarify": "ambiguity",
    "correction": "ordinary",
    "exclusion": "exclusion",
    "unavailable": "unavailable",
    "absence": "ambiguity",
    "unsupported": "unsupported",
    "aliases": "alias_distractor",
    "junk": "ambiguity",
}

SETTINGS_OPERATIONS = frozenset({
    "set_brightness",
    "set_color",
    "set_color_temperature",
    "set_speed",
    "set_temperature",
    "set_volume",
    "set_position",
})


@dataclass
class GenerationSlot:
    """One accepted-row target in the main generation loop."""

    index: int
    family: str
    robustness: str
    capability: str
    operation: str
    tier: int
    home_size: int
    area_scenario: str | None = None
    contrast_group: str | None = None


@dataclass
class AllocationPlan:
    """Requested vs planned row counts per family."""

    count: int
    requested: dict[str, int]
    slots: list[GenerationSlot] = field(default_factory=list)
    area_required: dict[str, int] = field(default_factory=dict)

    def achieved_template(self) -> dict[str, int]:
        return {family: 0 for family in self.requested}


def _distribute_shares(count: int, shares: dict[str, float]) -> dict[str, int]:
    if abs(sum(shares.values()) - 1.0) > 0.02:
        raise ValueError(f"allocation shares must sum to ~1.0, got {sum(shares.values()):.4f}")
    raw = {name: int(round(count * share)) for name, share in shares.items()}
    delta = count - sum(raw.values())
    order = sorted(shares, key=lambda k: shares[k], reverse=True)
    idx = 0
    while delta > 0:
        raw[order[idx % len(order)]] += 1
        delta -= 1
        idx += 1
    while delta < 0:
        richest = max(raw, key=lambda k: raw[k])
        if raw[richest] <= 0:
            raise ValueError("cannot reconcile allocation counts to requested total")
        raw[richest] -= 1
        delta += 1
    if sum(raw.values()) != count:
        raise ValueError(f"allocation counts sum to {sum(raw.values())}, expected {count}")
    return raw


def datetime_slot(index: int, *, home_size: int = 16) -> GenerationSlot:
    """One GetDateTime supervision slot (no entity graph)."""
    return GenerationSlot(
        index=index,
        family="datetime",
        robustness="datetime",
        capability="datetime",
        operation="query_time",
        tier=0,
        home_size=home_size,
    )


def _pick_capability_operation(family: str, rng: random.Random) -> tuple[str, str, int]:
    """Pick a (capability, operation, tier) suitable for ``family``."""
    if family == "datetime":
        return "datetime", "query_time", 0
    if family == "status":
        candidates = [
            (cap_name, "query_state", CAPABILITIES[cap_name].tier)
            for cap_name, cap in CAPABILITIES.items()
            if any(op.name == "query_state" and op.tool_name for op in cap.operations)
        ]
        cap_name, operation, tier = rng.choice(candidates)
        return cap_name, operation, tier
    if family == "settings":
        candidates = []
        for cap_name, cap in CAPABILITIES.items():
            for op in cap.operations:
                if op.name in SETTINGS_OPERATIONS and op.tool_name:
                    candidates.append((cap_name, op.name, cap.tier))
        cap_name, operation, tier = rng.choice(candidates)
        return cap_name, operation, tier
    if family in {"multi_action", "exclusion"}:
        cap_name = rng.choice(["lights", "switches", "fans", "covers"])
        operation = rng.choice(["turn_on", "turn_off", "open", "close"])
        return cap_name, operation, CAPABILITIES[cap_name].tier
    if family == "clarify":
        cap_name = rng.choice(["lights", "fans", "switches", "media_players"])
        operation = rng.choice(["turn_on", "turn_off", "query_state"])
        return cap_name, operation, CAPABILITIES[cap_name].tier
    if family == "follow_up":
        cap_name = rng.choice(["lights", "fans", "switches", "media_players"])
        operation = rng.choice(["turn_on", "turn_off"])
        return cap_name, operation, CAPABILITIES[cap_name].tier
    if family == "correction":
        cap_name = rng.choice(["lights", "fans", "switches", "media_players"])
        operation = rng.choice(["turn_on", "turn_off"])
        return cap_name, operation, CAPABILITIES[cap_name].tier
    if family == "unavailable":
        # Any operation whose tool Home Assistant can withhold, not just the
        # thermostat. Pinned to climate/set_temperature this family had one
        # semantic identity to draw from, so a 40k allocation spent 95% of its
        # attempts colliding on duplicate_semantic_id and still fell short --
        # and the corpus only ever taught "the thermostat tool is missing",
        # never the same refusal for a light, a vacuum, or a media player.
        cap_name, operation = rng.choice(_WITHHOLDABLE_OPERATIONS)
        return cap_name, operation, CAPABILITIES[cap_name].tier
    if family == "absence":
        return rng.choice(["lights", "fans", "media_players"]), "turn_on", 1
    if family == "unsupported":
        # Same widening as ``unavailable``: a device that cannot do the thing is
        # not only ever a media player.
        cap_name, operation = rng.choice(_UNDERPOWERED_OPERATIONS)
        return cap_name, operation, CAPABILITIES[cap_name].tier
    if family == "aliases":
        return rng.choice(["lights", "fans", "switches"]), "turn_on", 1
    if family == "junk":
        # The graph is irrelevant -- a junk transcript names no device. Borrow the
        # clarify shape so the row still renders against a real home and catalog.
        cap_name = rng.choice(["lights", "fans", "switches", "media_players"])
        return cap_name, "turn_on", CAPABILITIES[cap_name].tier
    # ordinary
    # Only draw tiers that actually contain an actionable capability. A
    # tier whose capabilities are all settings/query-only (tier 3 today)
    # has no action ops, so an ordinary slot could never be satisfied —
    # and the pre-fix code crashed on it (rng.choice over an empty list).
    # Keep TIER_PROPORTIONS relative weighting among eligible tiers.
    eligible_tiers = sorted(
        tier
        for tier in TIER_PROPORTIONS
        if any(cap.tier == tier and _action_ops(cap) for cap in CAPABILITIES.values())
    )
    tier = rng.choices(
        eligible_tiers,
        weights=[TIER_PROPORTIONS[t] for t in eligible_tiers],
        k=1,
    )[0]
    # Only capabilities with an action to take: a slot whose only op is query_state
    # (status gold) can never satisfy ordinary/aliases, and gets re-picked forever.
    tier_caps = [name for name, cap in CAPABILITIES.items() if cap.tier == tier and _action_ops(cap)]
    cap_name = rng.choice(tier_caps)
    return cap_name, rng.choice(_action_ops(CAPABILITIES[cap_name])), tier


def _action_ops(cap: Any) -> list[str]:
    """Tool-backed operations whose gold is an action: no settings, no state queries."""
    return [
        op.name
        for op in cap.operations
        if op.tool_name and op.name not in SETTINGS_OPERATIONS and op.name != "query_state"
    ]


def _eligible_operations() -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """(withholdable, underpowered) capability/operation pairs for refusal families.

    ``withholdable`` is anything Home Assistant supplies a tool for, so the
    recipe can remove that tool. ``underpowered`` additionally needs the device
    template to carry a feature the operation does not require, so an entity
    that genuinely lacks the capability can be built.
    """
    from generators.capability_registry import SupportLevel, required_features
    from generators.homes import _ENTITY_TEMPLATES

    withholdable: list[tuple[str, str]] = []
    underpowered: list[tuple[str, str]] = []
    for cap_name, cap in CAPABILITIES.items():
        template = _ENTITY_TEMPLATES.get(cap_name)
        for op in cap.operations:
            if not op.tool_name or op.support != SupportLevel.SUPPORTED:
                continue
            withholdable.append((cap_name, op.name))
            if not template:
                continue
            needed = set(required_features(cap_name, op.name))
            if needed and any(feature not in needed for feature in template[2]):
                underpowered.append((cap_name, op.name))
    return tuple(withholdable), tuple(underpowered)


_WITHHOLDABLE_OPERATIONS, _UNDERPOWERED_OPERATIONS = _eligible_operations()


def _area_required_counts(distribution_path: Path | None, count: int) -> dict[str, int]:
    if distribution_path is None or not distribution_path.is_file():
        return {}
    from generators.scenarios.area import assert_required_counts_feasible, load_distribution

    plan = load_distribution(distribution_path)
    return assert_required_counts_feasible(plan, count)


def _refusal_identity_ceiling(
    real_home_path: Path,
    *,
    near_duplicate_limit: int,
) -> dict[str, int]:
    """Max rows per refusal family if every attempt used the real home fixture."""
    from generators.capability_registry import entities_supporting, operation_spec
    from generators.real_home import load_real_home

    home = load_real_home(real_home_path, split="train")
    entities = home["entities"]
    areas = {entity["area"] for entity in entities}

    unavailable_keys: set[tuple[str, str, str, str | None]] = set()
    for cap_name, op_name in _WITHHOLDABLE_OPERATIONS:
        cap_entities = [entity for entity in entities if entity["capability"] == cap_name]
        op = operation_spec(cap_name, op_name)
        tool = op.tool_name if op else None
        for entity in entities_supporting(cap_entities, cap_name, op_name):
            unavailable_keys.add((cap_name, op_name, entity["entity_id"], tool))

    unsupported_keys: set[tuple[str, str, str]] = set()
    for cap_name, op_name in _UNDERPOWERED_OPERATIONS:
        for area in areas:
            unsupported_keys.add((cap_name, op_name, area))

    return {
        "unavailable": len(unavailable_keys) * near_duplicate_limit,
        "unsupported": len(unsupported_keys) * near_duplicate_limit,
    }


def _eligible_real_home_slots(
    family_counts: dict[str, int],
    area_required: dict[str, int],
) -> int:
    """Rows whose family may be drawn from the real home when balancing the mix."""
    eligible = sum(
        count
        for family, count in family_counts.items()
        if family not in REAL_HOME_EXCLUDED_FAMILIES
    )
    eligible += sum(area_required.values())
    return eligible


def verify_feasible(
    count: int,
    allocations: dict[str, float],
    *,
    near_duplicate_limit: int,
    area_required: dict[str, int],
    grounding_variants: int,
    grounding_rate: float,
    real_home_path: Path | None = None,
    real_home_rate: float = 0.0,
    refusal_on_synthetic_only: bool = True,
) -> None:
    """Fail before generation when quotas cannot be met."""
    family_counts = _distribute_shares(count, allocations)
    if area_required:
        area_total = sum(area_required.values())
        if area_total > count:
            raise ValueError(
                f"area scenario minimums ({area_total}) exceed corpus size ({count})"
            )
    if grounding_rate > 0:
        max_grounding = grounding_variants * near_duplicate_limit
        requested = int(round(count * grounding_rate))
        if requested > max_grounding:
            raise ValueError(
                f"grounding_rate {grounding_rate} requests {requested} rows but "
                f"catalogue allows at most {max_grounding}"
            )
    if real_home_path and real_home_rate > 0:
        real_target = int(round(count * real_home_rate))
        eligible = _eligible_real_home_slots(family_counts, area_required)
        if real_target > eligible:
            raise ValueError(
                f"real_home rate {real_home_rate} requests {real_target} rows but only "
                f"{eligible} slots may use the real home once "
                f"{sorted(REAL_HOME_EXCLUDED_FAMILIES)} stay synthetic"
            )
        ceilings = _refusal_identity_ceiling(
            real_home_path,
            near_duplicate_limit=near_duplicate_limit,
        )
        for family in ("unavailable", "unsupported"):
            need = family_counts.get(family, 0)
            ceiling = ceilings[family]
            if need > ceiling and not refusal_on_synthetic_only:
                raise ValueError(
                    f"{family} allocation requests {need} rows but the real home "
                    f"supports at most {ceiling} distinct refusal identities at "
                    f"near_duplicate_limit={near_duplicate_limit}; keep the family "
                    f"synthetic or lower its share"
                )


def build_plan(
    count: int,
    seed: int,
    allocations: dict[str, float],
    *,
    area_distribution_path: Path | None = None,
    near_duplicate_limit: int = 8,
    grounding_variants: int = 0,
    grounding_rate: float = 0.0,
    datetime_required: int = 0,
    real_home_path: Path | None = None,
    real_home_rate: float = 0.0,
) -> AllocationPlan:
    """Build shuffled generation slots from a versioned recipe."""
    area_required = _area_required_counts(area_distribution_path, count)
    area_total = sum(area_required.values())
    primary_count = count - area_total - datetime_required
    if primary_count < 1:
        raise ValueError(
            f"area minimums ({area_total}) and datetime slots ({datetime_required}) "
            f"leave no room for primary families"
        )
    verify_feasible(
        count,
        allocations,
        near_duplicate_limit=near_duplicate_limit,
        area_required=area_required,
        grounding_variants=grounding_variants,
        grounding_rate=grounding_rate,
        real_home_path=real_home_path,
        real_home_rate=real_home_rate,
    )
    requested = _distribute_shares(primary_count, allocations)
    if datetime_required:
        requested = {**requested, "datetime": datetime_required}
    rng = random.Random(seed)
    slots: list[GenerationSlot] = []
    index = 0
    for family, family_count in requested.items():
        robustness = FAMILY_ROBUSTNESS[family]
        for _ in range(family_count):
            if family == "datetime":
                slots.append(datetime_slot(index))
                index += 1
                continue
            cap, operation, tier = _pick_capability_operation(family, rng)
            home_size = sample_home_size(HOME_SIZE_WEIGHTS, rng)
            if family in {"exclusion", "multi_action"}:
                home_size = max(home_size, 64)
                cap, operation = "lights", "turn_off" if family == "exclusion" else "turn_on"
            slots.append(
                GenerationSlot(
                    index=index,
                    family=family,
                    robustness=robustness,
                    capability=cap,
                    operation=operation if family != "status" else "query_state",
                    tier=tier,
                    home_size=home_size,
                    contrast_group=f"{family}_contrast" if family in {"status", "clarify", "follow_up"} else None,
                )
            )
            index += 1
    for scenario, need in area_required.items():
        for offset in range(need):
            slots.append(
                GenerationSlot(
                    index=index + offset,
                    family="area",
                    robustness="area",
                    capability="lights",
                    operation="turn_off",
                    tier=1,
                    home_size=16,
                    area_scenario=scenario,
                )
            )
        index += need
    rng.shuffle(slots)
    for i, slot in enumerate(slots):
        slot.index = i
    if len(slots) != count:
        raise ValueError(f"planned {len(slots)} slots, expected {count}")
    return AllocationPlan(count=count, requested=requested, slots=slots, area_required=area_required)


class FamilyTracker:
    """Track accepted rows against family and area allocations."""

    def __init__(self, plan: AllocationPlan) -> None:
        self.plan = plan
        self.accepted_family: dict[str, int] = {k: 0 for k in plan.requested}
        self.accepted_area: dict[str, int] = {k: 0 for k in plan.area_required}
        self._total = 0

    def total(self) -> int:
        return self._total

    def is_complete(self) -> bool:
        return self._total >= self.plan.count

    def family_deficit(self, family: str) -> int:
        if family == "area":
            return sum(
                max(0, need - self.accepted_area.get(name, 0))
                for name, need in self.plan.area_required.items()
            )
        target = self.plan.requested.get(family, 0)
        return max(0, target - self.accepted_family.get(family, 0))

    def record(self, family: str, area_scenario: str | None = None) -> None:
        if family == "area" and area_scenario:
            self.accepted_area[area_scenario] = self.accepted_area.get(area_scenario, 0) + 1
        else:
            self.accepted_family[family] = self.accepted_family.get(family, 0) + 1
        self._total += 1

    def verify_complete(self) -> None:
        shortfalls = {
            family: (self.accepted_family[family], need)
            for family, need in self.plan.requested.items()
            if self.accepted_family[family] < need
        }
        area_short = {
            name: (self.accepted_area.get(name, 0), need)
            for name, need in self.plan.area_required.items()
            if self.accepted_area.get(name, 0) < need
        }
        if shortfalls or area_short:
            raise RuntimeError(
                f"allocation shortfall: families={shortfalls} area={area_short}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "requested": dict(self.plan.requested),
            "achieved_family": dict(self.accepted_family),
            "requested_area": dict(self.plan.area_required),
            "achieved_area": dict(self.accepted_area),
            "total": self._total,
        }


def discrimination_available_share(config: Any) -> float:
    """Share of accepted rows that can realistically be description-based rows."""
    from generators.scenarios.discrimination import DISCRIMINATION_DELIVERY_CEILING

    share = DISCRIMINATION_DELIVERY_CEILING
    if config.real_home_path and config.real_home_rate:
        share *= max(0.0, 1.0 - config.real_home_rate)
    return share


def discrimination_capable_slot(quota: Any, rng: random.Random) -> dict[str, Any] | None:
    """Take a quota slot that can host a description-based row."""
    from generators.scenarios.discrimination import DISCRIMINATION_CAPABILITIES

    keys = sorted(
        key
        for key in quota.targets["operation"]
        if key[1] in DISCRIMINATION_CAPABILITIES and quota._key_gap(key) > 0
    )
    if not keys:
        return None
    rng.shuffle(keys)
    return quota._slot_from(keys)
