"""Build one candidate training row from a generation slot."""

from __future__ import annotations

import copy
import random
from typing import Any

from generators.capability_registry import CAPABILITIES
from generators.config import GeneratorConfig
from generators.deduplication import DuplicateTracker, pair_hash
from generators.grounding import pick_variant
from generators.gold import gold_matches_family
from generators.labels import render_example, scenario_to_spec
from generators.real_home import load_real_home
from generators.scenarios import build_scenario, pick_robustness, pick_targeting
from generators.scenarios.discrimination import (
    call_carries_value,
    pick_discriminating_description,
)
from generators.scenarios.unavailable import unique_no_action_hint
from generators.stt_noise import apply_stt_noise
from generators.utterances import (
    apply_generic_wording,
    expand_utterance,
    finalize_training_utterance,
    request_seed_from_spec,
)
from generators.validation import check_quality_eval_overlap, validate_row

_DATETIME_UTTERANCES = (
    "what time is it",
    "tell me the time",
    "what's the time right now",
    "what is the current time",
    "what time is it now",
)


def _datetime_utterance(rng: random.Random, index: int) -> str:
    return rng.choice(_DATETIME_UTTERANCES)


def generate_row(
    slot: dict[str, Any],
    config: GeneratorConfig,
    rng: random.Random,
    *,
    excluded: set[str],
    dup_tracker: DuplicateTracker,
    attempt: int = 0,
    stt_remaining: int = 0,
    rows_remaining: int = 1,
    real_home_targets: Any | None = None,
    real_home_entity_cap: int = 0,
    real_home_names: frozenset[str] = frozenset(),
    real_home_usage: Any | None = None,
    quota: Any | None = None,
    grounding_required: bool = False,
    discrimination_required: bool = False,
    real_home_selected: bool | None = None,
    grounding_variants: list[dict[str, Any]] | None = None,
    grounding_rate: float = 0.0,
) -> tuple[dict[str, Any] | None, str | None]:
    area_scenario = slot.get("area_scenario")
    if area_scenario:
        from generators.scenarios.area import build_area_spec

        spec = build_area_spec(area_scenario, slot["index"], config.seed)
        spec["full_tool_catalog"] = True
        if check_quality_eval_overlap(spec["utterance"]):
            return None, "quality_eval_overlap"
        spec["utterance"] = finalize_training_utterance(spec["utterance"], rng)
        if spec["utterance"].casefold() in excluded:
            return None, "excluded_prompt"
        if check_quality_eval_overlap(spec["utterance"]):
            return None, "quality_eval_overlap"
        reason = validate_row(spec, token_budget=config.token_budget)
        if reason:
            return None, reason
        reject = dup_tracker.would_reject(spec)
        if reject:
            return None, reject
        spec["candidate_id"] = (
            f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
            f"_{dup_tracker.occurrences(spec)}"
        )
        try:
            row = render_example(spec)
        except ValueError as exc:
            return None, str(exc)
        if spec.get("clarify_question"):
            row["messages"][-1]["content"] = spec["clarify_question"]
        row["metadata"]["area_scenario"] = area_scenario
        row["metadata"]["family"] = "area"
        row["metadata"]["real_home"] = False
        row["metadata"]["grounding_family"] = None
        row["metadata"]["discrimination"] = False
        dup_tracker.record(spec)
        return row, None

    capability = slot["capability"]
    operation = slot["operation"]
    family = slot.get("family")
    home_size = slot["home_size"]
    home = None
    real_home_row = False
    grounding_family = None
    variant = None
    if capability == "datetime":
        robustness = "datetime"
        targeting = "context"
        inject_missing = True
        scenario_index = slot["index"] + attempt * 10000
    else:
        cap = CAPABILITIES[capability]
        robustness = slot.get("robustness") or pick_robustness(rng, 0.85)
        if (
            not slot.get("family")
            and attempt % 125 == 0
            and robustness == "ordinary"
            and capability in {"lights", "switches", "fans", "covers"}
            and operation in {"turn_on", "turn_off", "open", "close"}
        ):
            robustness = "exclusion"
        targeting = pick_targeting(cap, rng, robustness)
        if robustness == "large_home":
            home_size = max(home_size, 64)
        inject_missing = family not in {"absence", "unsupported"}
        select_real_home = (
            rng.random() < config.real_home_rate
            if real_home_selected is None else real_home_selected
        )
        variant = (
            pick_variant(capability, operation, slot["index"] + attempt, variants=grounding_variants)
            if grounding_variants else None
        )
        if variant is not None:
            pass
        elif config.real_home_path and select_real_home:
            home = load_real_home(config.real_home_path, split="train")
            real_home_row = True
        elif grounding_rate and grounding_required:
            variant = pick_variant(capability, operation, slot["index"] + attempt)
        elif grounding_rate and rng.random() < grounding_rate:
            variant = pick_variant(capability, operation, slot["index"] + attempt)
        if variant is not None:
            home = copy.deepcopy(variant["home"])
            targeting = variant["targeting"]
            robustness = variant["robustness"]
            grounding_family = variant["family"]
            if family not in {"absence", "unsupported"}:
                inject_missing = variant.get("inject_missing", True)
        scenario_index = slot["index"] + attempt * 10000
        if variant is not None and variant.get("rng_index") is not None:
            scenario_index = variant["rng_index"]

    scenario = build_scenario(
        index=scenario_index,
        seed=config.seed ^ (attempt << 16),
        capability=capability,
        operation=operation,
        home_size=home_size,
        targeting=targeting,
        robustness=robustness,
        split=config.split,
        attempt=attempt,
        home=home,
        inject_missing=inject_missing,
        target_usage=real_home_usage if real_home_row else None,
        family=family,
    )
    if grounding_family:
        scenario["phrasing_seed"] = variant["phrasing_seed"]

    spec = scenario_to_spec(scenario)
    if family:
        spec["family"] = family
    spec["full_tool_catalog"] = True
    expected = spec.get("expected") or {}
    if family and not gold_matches_family(expected, family):
        return None, "family_mismatch"
    if expected.get("kind") == "no_action":
        spec["request_hint"] = unique_no_action_hint(spec, rng)
    elif robustness == "ambiguity":
        apply_generic_wording(spec)
    if family == "datetime":
        spec["utterance"] = _datetime_utterance(rng, slot["index"])
        spec["linguistics"] = [{"source": "sayso_fallback", "intent": "GetDateTime"}]
    else:
        spec["utterance"] = expand_utterance(spec)

    discrimination = False
    if (
        discrimination_required
        and variant is None
        and expected.get("kind") == "action"
        and scenario.get("targeting") == "individual"
        and len(expected.get("calls") or []) == 1
        and not call_carries_value((expected.get("calls") or [{}])[0])
    ):
        seed_text = request_seed_from_spec(spec)
        description = pick_discriminating_description(scenario, rng)
        if description:
            verb = "turn on"
            for candidate in ("turn on", "turn off", "open", "close", "lock", "unlock", "set", "run"):
                if seed_text.lower().startswith(candidate):
                    verb = candidate
                    break
            spec["utterance"] = f"{verb} {description}"
            discrimination = True
            spec["discrimination"] = True

    if grounding_required and not grounding_family:
        return None, "grounding_variant_miss"

    if check_quality_eval_overlap(spec["utterance"]):
        return None, "quality_eval_overlap"

    if (
        stt_remaining > 0
        and rows_remaining > 0
        and rng.random() < stt_remaining / rows_remaining
    ):
        corrupted, kind = apply_stt_noise(
            spec["utterance"],
            rng,
            target_names=spec.get("target_names"),
            force_transform=True,
        )
        if kind:
            trial = dict(spec)
            trial["utterance"] = corrupted
            trial["stt_corruption"] = kind
            if validate_row(trial, token_budget=config.token_budget) is None:
                spec = trial

    spec["utterance"] = finalize_training_utterance(spec["utterance"], rng)
    if spec["utterance"].casefold() in excluded:
        return None, "excluded_prompt"
    if check_quality_eval_overlap(spec["utterance"]):
        return None, "quality_eval_overlap"

    reason = validate_row(spec, token_budget=config.token_budget)
    if reason:
        return None, reason

    reject = dup_tracker.would_reject(spec)
    if reject:
        return None, reject

    spec["candidate_id"] = (
        f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
        f"_{dup_tracker.occurrences(spec)}"
    )

    targets = spec.get("target_names") or []
    if real_home_row and real_home_entity_cap and real_home_targets is not None:
        if any(real_home_targets[name] >= real_home_entity_cap for name in targets):
            return None, "real_home_entity_cap"

    try:
        row = render_example(spec)
    except ValueError as exc:
        return None, str(exc)

    row["metadata"]["real_home"] = real_home_row
    row["metadata"]["grounding_family"] = grounding_family
    row["metadata"]["discrimination"] = discrimination
    if family:
        row["metadata"]["family"] = family
    if quota is not None and family != "datetime":
        from generators.coverage import classify_row

        reason = quota.wants(classify_row(row))
        if reason:
            return None, reason
    dup_tracker.record(spec)
    if real_home_row and real_home_targets is not None:
        real = [name for name in targets if not real_home_names or name in real_home_names]
        real_home_targets.update(real)
        if real_home_usage is not None:
            real_home_usage.update(real)
    return row, None
