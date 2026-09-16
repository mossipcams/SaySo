"""Main deterministic generation pipeline."""

from __future__ import annotations

import copy
import json
import random
import zlib
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from generators.capability_registry import (
    CAPABILITIES,
    registry_summary,
)
from generators.config import GeneratorConfig
from generators.coverage import classify_row
from generators.duplicates import DuplicateTracker, pair_hash
from generators.grounding import (
    pick_variant,
    required_training_variants,
    training_variants,
)
from generators.labels import render_example, scenario_to_spec
from generators.gold import target_names_from_expected
from generators.homes import make_entity, _ENTITY_TEMPLATES
from generators.tools import build_call_for_operation
from generators.paraphrase import load_paraphraser
from generators.real_home import derive_entity_cap, load_real_home
from generators.sampling import QuotaTracker, build_quota_plan
from generators.scenarios import build_scenario, pick_robustness, pick_targeting
from generators.stats import empty_stats, finalize_stats, record_accept, record_reject
from generators.stt_noise import apply_stt_noise
from generators.utterances import (
    _BRAND_TOKENS as _DESCRIPTION_BRANDS,
    _STOP_TOKENS as _DESCRIPTION_STOPWORDS,
    _plural,
    apply_generic_wording,
    describe_target,
    expand_utterance,
    request_seed_from_spec,
    vary_training_utterance,
)
from generators.validate import validate_row


# Operations Home Assistant supplies no tool for. They have no call to render, so
# the request they refuse has to be written out; the refusal itself still comes
# from the registry (SupportLevel.UNAVAILABLE), not from this table.
_UNAVAILABLE_REQUESTS: dict[tuple[str, str], str] = {
    ("lawn_mowers", "control"): "start mowing the lawn with {name}",
    ("todo_lists", "control"): "add milk to {name}",
    ("buttons", "control"): "press {name}",
    ("covers", "set_position"): "set {name} to 40 percent open",
    ("scripts", "query_state"): "what is the status of {name}",
}


# Requiring every grounding contrast family needs slack, not parity. A family
# lands only when a slot of its capability/operation comes up *and* that bucket
# still has room for its outcome; the refusal families (`moved`, `incapable`)
# additionally compete for a bucket's small negative allowance. At parity a
# 400-row run fails closed on a gate it was never large enough to meet.
GROUNDING_FAMILY_SLACK = 2


def _enforce_rate_gate(
    section: dict[str, Any],
    name: str,
    config: GeneratorConfig,
    accepted: int,
    available_share: float = 1.0,
) -> None:
    """Fail the build when a requested row share was not actually achieved.

    v1 requested a 3% grounding share and shipped 0.22% with no error, because
    ``achieved_rate`` was recorded but never asserted. A gate here is the only
    thing that turns a silent shortfall into a loud one.

    The gate is size-aware. Below ``_RATE_GATE_MIN_ROWS`` a percentage is not
    meaningfully assertable, so the rate is recorded but not asserted.

    ``available_share`` is the fraction of slots that could host this kind of row
    at all. Grounding variants exist for only a few (capability, operation)
    pairs, and real-home mixing or other prefilters can claim slots first, so the
    effective ceiling is often below the requested rate. Comparing against an
    unreachable request would fail every honest run; comparing against the
    reachable ceiling is what makes the gate informative.
    """
    requested = section["requested_rate"]
    if requested <= 0 or accepted < _RATE_GATE_MIN_ROWS:
        return
    ceiling = min(requested, available_share)
    floor = ceiling * config.min_rate_achieved_fraction
    if section["achieved_rate"] + 1e-9 < floor:
        raise RuntimeError(
            f"{name} rate shortfall: requested {requested:.4f}, "
            f"reachable ceiling {ceiling:.4f}, achieved "
            f"{section['achieved_rate']:.4f} ({section['rows']}/{accepted} rows); "
            f"floor is {floor:.4f} "
            f"(min_rate_achieved_fraction={config.min_rate_achieved_fraction})"
        )


# Below this many accepted rows a percentage rate is not meaningfully assertable.
_RATE_GATE_MIN_ROWS = 1000


def _call_carries_value(call: dict[str, Any]) -> bool:
    """True when the call needs an argument a description cannot convey."""
    arguments = call.get("arguments") or {}
    value_keys = {
        "brightness", "color", "temperature", "percentage", "volume",
        "position", "mode", "fan_speed", "duration", "seconds", "hours",
        "minutes", "query", "media_id", "search_query",
    }
    return bool(value_keys & set(arguments))


def _grounding_capacity(config: GeneratorConfig) -> int:
    """Most grounding rows this catalogue can ever produce, at any corpus size.

    Every grounding variant is one fixed scenario, and ``DuplicateTracker``
    accepts a scenario at most ``near_duplicate_limit`` times. So the catalogue,
    not the corpus size, sets the absolute ceiling -- which is why v1 shipped 136
    grounding rows against a 1120-row request while a 2k smoke run of the same
    code looked healthy. More capacity means more sites, see
    ``grounding.site_variants``.
    """
    return len(training_variants()) * config.near_duplicate_limit


def _grounding_available_share(config: GeneratorConfig) -> float:
    """Share of accepted rows that can realistically be grounding rows.

    Two hard limits, whichever binds first: grounding variants exist for only
    four (capability, operation) pairs (~7.3% of quota slots, less whatever
    real-home mixing claims), and the catalogue itself caps the row count
    absolutely (``_grounding_capacity``).
    """
    catalogue_share = _grounding_capacity(config) / max(config.count, 1)
    return min(_slot_ceiling(config, _grounding_pairs()), catalogue_share)


def _discrimination_available_share(config: GeneratorConfig) -> float:
    """Share of accepted rows that can realistically be description-based rows.

    Slot supply is not the limiter: 35.7% of the quota plan is lights, covers or
    switches. The eligibility chain is. A description row needs individual
    targeting, a single call, no value argument, and at least one same-domain
    same-area sibling to discriminate against -- and 54% of scenarios have no
    such sibling (measured over 1,200). See ``_DISCRIMINATION_DELIVERY_CEILING``.
    """
    share = _DISCRIMINATION_DELIVERY_CEILING
    if config.real_home_path and config.real_home_rate:
        share *= max(0.0, 1.0 - config.real_home_rate)
    return share


def _grounding_pairs() -> set[tuple[str, str]]:
    return {
        (variant["capability"], variant["operation"])
        for variant in required_training_variants()
    }


def _slot_ceiling(config: GeneratorConfig, pairs: set[tuple[str, str]]) -> float:
    """Share of the quota plan landing on ``pairs``, net of real-home mixing."""
    try:
        plan = build_quota_plan(config.count, config.seed, config.tier_proportions)
    except Exception:  # noqa: BLE001 - ceiling is advisory; never fail a build here
        return 1.0
    matching = sum(
        1 for slot in plan if (slot["capability"], slot["operation"]) in pairs
    )
    share = matching / max(len(plan), 1)
    # Real-home rows are drawn first and take their share of every pair.
    if config.real_home_path and config.real_home_rate:
        share *= max(0.0, 1.0 - config.real_home_rate)
    return min(1.0, share)


# Measured delivery ceiling for description-based rows (2026-09-13). Saturating
# the request does not move it: 0.35 requested at n=4,000 delivered 0.0198, and
# 0.02 requested at n=40,000 delivered 0.0146 and 0.0153 on two seeds' worth of
# RNG drift. The previous model (slot share x 0.5 = ~45%) made the gate compare
# against the raw request, so an honest build failed roughly half the time.
#
# Do not "fix" a shortfall by lowering the requested rate: the request drives the
# forcing loop (``discrimination_target``), so asking for less delivers less. The
# way up is to relax the eligibility chain -- forcing ``home_size >= 48`` on these
# rows alone measured 0.0198 -> 0.034.
_DISCRIMINATION_DELIVERY_CEILING = 0.02


# Capabilities that can yield a description-based row. A description singles out
# one entity from several of the same domain in one area, so it is only
# producible where homes actually place siblings: lights, covers and switches.
# Measured yield for the rest is zero (a home rarely has two thermostats or two
# locks in one room), so requesting discrimination rows there wastes slots.
_DISCRIMINATION_CAPABILITIES: frozenset[str] = frozenset(
    {"lights", "covers", "switches"}
)


def _discrimination_capable_slot(quota: Any, rng: random.Random) -> dict[str, Any] | None:
    """Take a quota slot that can host a description-based row."""
    keys = sorted(
        key
        for key in quota.targets["operation"]
        if key[1] in _DISCRIMINATION_CAPABILITIES and quota._key_gap(key) > 0
    )
    if not keys:
        return None
    rng.shuffle(keys)
    return quota._slot_from(keys)


def _grounding_capable_slot(quota: Any, rng: random.Random) -> dict[str, Any] | None:
    """Take a quota slot whose (capability, operation) has a grounding variant.

    ``pick_variant`` only returns a variant for the pairs named by
    ``required_training_variants``; every other pair yields None. Asking for a
    slot in one of those pairs is the difference between a 3% grounding share
    being attainable and being a 0.2% accident.
    """
    pairs = sorted(
        {(variant["capability"], variant["operation"]) for variant in required_training_variants()}
    )
    rng.shuffle(pairs)
    for capability, operation in pairs:
        slot = quota.take_slot(capability, operation)
        if slot is not None:
            return slot
    return None


def _unique_no_action_hint(spec: dict[str, Any], rng: random.Random) -> str:
    """Describe the actual blocked request, never an unrelated random action."""
    requested = spec["expected"].get("requested")
    if requested:
        return request_seed_from_spec({
            **spec, "expected": requested,
            "target_names": target_names_from_expected(requested),
        })
    capability = spec["capability"]
    operation = spec["operation"]
    area = spec["home"]["sayso_entity_area"]
    entity = None
    if capability != "timers":
        entity = make_entity(
            name=f"the {area} {'routine' if capability == 'scripts' else _ENTITY_TEMPLATES[capability][0].lower()}",
            capability=capability, area=area, floor="Main Floor", rng=rng,
        )
    template = _UNAVAILABLE_REQUESTS.get((capability, operation))
    if template:
        spec["linguistics"] = [{"source": "sayso_fallback", "intent": f"{capability}.{operation}"}]
        return template.format(name=entity["name"] if entity else "it")
    call = build_call_for_operation(entity, capability, operation, rng, area=area)
    return request_seed_from_spec({
        "expected": {"kind": "action", "calls": [call]},
        "target_names": [entity["name"]] if entity else [""],
        "phrasing_seed": spec.get("phrasing_seed"),
    })


def _load_excluded_prompts(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    excluded: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            for msg in row.get("messages", []):
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    excluded.add(msg["content"].casefold())
        except json.JSONDecodeError:
            continue
    return excluded


def _normalize_prompt(text: str) -> str:
    """Match excluded_train_prompts(): punctuation-insensitive, so "joe's" == "joe s"."""
    prompt = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    prompt = re.sub(r"^(?:(?:please|can you|could you|tell me) )+", "", prompt)
    return re.sub(r"(?: for me)+$", "", prompt)


@lru_cache(maxsize=1)
def _quality_eval_prompts() -> frozenset[str]:
    """Normalized gold, shadow, and recipe-lock prompts, none of which may be trained on."""
    try:
        from evals.v3_quality import excluded_train_prompts

        prompts = {_normalize_prompt(prompt) for prompt in excluded_train_prompts()}
    except ImportError:
        try:
            from evals.recipe_lock import locked_specs

            prompts = {_normalize_prompt(spec["utterance"]) for spec in locked_specs()}
        except ImportError:
            prompts = set()

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "realistic_eval_20260908_v2.json"
    for case in json.loads(fixture.read_text(encoding="utf-8"))["cases"]:
        prompts.update(_normalize_prompt(message["content"]) for message in case["messages"]
                       if message["role"] == "user")
    return frozenset(prompts)


def _check_quality_eval_overlap(utterance: str) -> bool:
    """Reject contamination from golden, shadow, and recipe-lock eval utterances."""
    return _normalize_prompt(utterance) in _quality_eval_prompts()


def _entity_descriptors(entity: dict[str, Any], area: str | None) -> dict[str, str | None]:
    """Pull genuinely descriptive tokens out of an entity's own name.

    Returns ``modifier`` (a placement/kind token such as "Ceiling", "Pendant",
    "Wall", "Vanity") and ``brand`` (a manufacturer token such as "Nanoleaf",
    "Lutron"). Both come from the name, so a description built from them is
    factually true of that entity. Tokens that would leak the whole name or are
    just the domain noun are excluded.
    """
    name = str(entity.get("name") or "")
    area = area or str(entity.get("area") or "")
    tokens = [t for t in re.split(r"[\s\-]+", name) if t]
    # Drop the leading area word(s) so "Kitchen Counter Light" -> ["Counter"].
    area_tokens = [t.lower() for t in re.split(r"[\s\-]+", area) if t]
    while tokens and tokens[0].lower() in area_tokens:
        tokens.pop(0)
    # Trailing generic nouns ("light", "lights", "tv") carry no signal.
    tokens = [
        t for t in tokens
        if t.lower() not in _DESCRIPTION_STOPWORDS and not t.lower().endswith("'s")
    ]
    brand = next((t for t in tokens if t.lower() in _DESCRIPTION_BRANDS), None)
    modifier = next(
        (t for t in tokens if t.lower() not in _DESCRIPTION_BRANDS and len(t) > 2),
        None,
    )
    # "spotlights" is already the plural domain noun; using it as a modifier would
    # render "the spotlights lights".
    if modifier and modifier.lower() in {
        _plural(str(entity.get("domain") or "")).lower(),
        str(entity.get("domain") or "").lower(),
    }:
        modifier = None
    return {"modifier": modifier, "brand": brand}


def _sibling_entities(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Same-domain, same-area entities other than the target."""
    home = scenario.get("home") or {}
    target = scenario.get("target_entity") or {}
    domain = target.get("domain")
    area = target.get("area")
    if not domain or not area:
        return []
    return [
        entity
        for entity in home.get("entities", [])
        if entity.get("domain") == domain
        and entity.get("area") == area
        and entity.get("entity_id") != target.get("entity_id")
    ]


def _uniqueness_safe(
    text: str,
    target: dict[str, Any],
    siblings: list[dict[str, Any]],
) -> bool:
    """Reject a description that any sibling could also satisfy.

    Two ways a description can be non-unique:

    * it names the target itself (or an alias), which is not discrimination at
      all;
    * it names a *sibling* (or a sibling's alias), which points at the wrong
      device;
    * it is a bare area form ("the lights in the Kitchen") while siblings exist,
      which refers to all of them.
    """
    lowered = text.lower()
    own = {str(target.get("name") or "").lower()}
    own.update(str(a).lower() for a in (target.get("aliases") or ()))
    own.discard("")
    if any(name in lowered for name in own):
        return False
    for sibling in siblings:
        names = {str(sibling.get("name") or "").lower()}
        names.update(str(a).lower() for a in (sibling.get("aliases") or ()))
        names.discard("")
        if any(name in lowered for name in names):
            return False
    return True


def _pick_discriminating_description(
    scenario: dict[str, Any],
    rng: random.Random,
) -> str | None:
    """Render a description that can only refer to the target.

    A description-based row is only usable if exactly one device satisfies it.
    Uniqueness is enforced structurally:

    * there must be at least one same-domain same-area sibling, otherwise the row
      teaches nothing (it is a direct request with worse wording);
    * no rendered description may contain the target's or any sibling's name or
      alias;
    * if siblings exist, plain area forms ("the lights in the Kitchen") are
      refused, because they denote the whole group. Only qualifier forms are
      eligible.

    Returns None when no safe description can be rendered; callers must drop the
    row rather than emit an ambiguous one.
    """
    target = scenario.get("target_entity") or {}
    domain = target.get("domain")
    area = target.get("area")
    if not domain or not area or not target.get("name"):
        return None
    siblings = _sibling_entities(scenario)
    if not siblings:
        # Nothing to discriminate between; a description here is strictly easier
        # than the direct request the corpus already contains.
        return None
    for _ in range(24):
        descriptors = _entity_descriptors(target, area)
        text = describe_target(
            domain, area, rng,
            modifier=descriptors["modifier"],
            brand=descriptors["brand"],
        )
        if text and _uniqueness_safe(text, target, siblings):
            # The description must also not accidentally describe a sibling: if
            # the modifier token appears in a sibling's name the row is ambiguous.
            mod = descriptors["modifier"]
            if mod and any(
                mod.lower() in str(s.get("name") or "").lower() for s in siblings
            ):
                continue
            return text
    return None


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
    real_home_targets: Counter[str] | None = None,
    real_home_entity_cap: int = 0,
    real_home_names: frozenset[str] = frozenset(),
    real_home_usage: Counter[str] | None = None,
    quota: Any | None = None,
    grounding_required: bool = False,
    discrimination_required: bool = False,
    real_home_selected: bool | None = None,
    grounding_variants: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    capability = slot["capability"]
    operation = slot["operation"]
    cap = CAPABILITIES[capability]
    robustness = pick_robustness(rng, config.ordinary_rate)
    targeting = pick_targeting(cap, rng, robustness)
    home_size = slot["home_size"]
    if robustness == "large_home":
        home_size = max(home_size, 64)

    # ponytail: deep-copied per row because build_scenario mutates the home.
    # Cheap next to utterance expansion; cache the split lists if it ever isn't.
    home = None
    real_home_row = False
    grounding_family = None
    inject_missing = True
    select_real_home = (
        rng.random() < config.real_home_rate
        if real_home_selected is None else real_home_selected
    )
    # Rotate even while forcing the missing families: two of them can share one
    # capability/operation pool, and a fixed index would keep retrying the first
    # until it lands, leaving the second unreachable.
    variant = (
        pick_variant(capability, operation, slot["index"] + attempt, variants=grounding_variants)
        if grounding_variants else None
    )
    if variant is not None:
        # A still-missing contrast family outranks the real-home draw: real rows
        # are a rate that self-corrects across the run, a missing family is a gate.
        pass
    elif config.real_home_path and select_real_home:
        # A real-home row that has already been selected outranks an owed
        # grounding row. Real-home mixing needs specific (capability, operation)
        # pairs -- media_players/turn_on among them -- and those are also
        # grounding-capable, so preempting here would starve real-home mixing of
        # the very operations it is meant to cover.
        home = load_real_home(config.real_home_path, split="train")
        real_home_row = True
    elif config.grounding_rate and grounding_required:
        # Owed grounding row, and no real-home row was drawn for this slot.
        variant = pick_variant(capability, operation, slot["index"] + attempt)
    elif config.grounding_rate and rng.random() < config.grounding_rate:
        variant = pick_variant(capability, operation, slot["index"] + attempt)
    if variant is not None:
        home = copy.deepcopy(variant["home"])
        targeting = variant["targeting"]
        robustness = variant["robustness"]
        grounding_family = variant["family"]
        inject_missing = variant.get("inject_missing", True)

    # A grounding variant may pin the scenario index, because ``pick_target``
    # rotates on it: the alias family lists the aliased entity first and a
    # distractor second, so an unpinned index made ~42% of alias rows target the
    # distractor while still being labelled and counted as alias grounding rows.
    # ``grounding.build_spec`` already honours this, and the eval goes through
    # that path -- ignoring it here made training and eval disagree about what a
    # variant means.
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
        # Only the real home reuses its entities across rows, so only it needs the
        # least-used tie-break; synthetic homes are fresh each row.
        target_usage=real_home_usage if real_home_row else None,
    )
    if grounding_family:
        scenario["phrasing_seed"] = variant["phrasing_seed"]

    spec = scenario_to_spec(scenario)
    # Which tool contract this row renders in. Seeded from the row's own identity
    # rather than drawn from the shared run rng: taking two draws from the shared
    # stream would shift every later row's home and distractors, so turning these
    # rates on -- or off -- would silently change an unrelated corpus. crc32, not
    # builtin hash(), because str hashing is randomized per process.
    contract_rng = random.Random(
        zlib.crc32(
            f"{spec.get('semantic_id') or spec['candidate_id']}"
            f":{slot['index']}:{attempt}:contract".encode()
        )
    )
    spec["namespaced_tools"] = contract_rng.random() < config.namespaced_tool_rate
    spec["full_tool_catalog"] = contract_rng.random() < config.full_catalog_rate
    expected = spec.get("expected") or {}
    if expected.get("kind") == "no_action":
        spec["request_hint"] = _unique_no_action_hint(spec, rng)
    elif robustness == "ambiguity":
        apply_generic_wording(spec)
    spec["utterance"] = expand_utterance(spec)

    # Entity-discrimination row: replace the name-based request with a
    # description that only the target satisfies. Applied after the normal
    # utterance so every upstream gate (overlap, STT, validation) still sees a
    # well-formed request; a description that cannot be made unique is dropped
    # rather than shipped ambiguous.
    discrimination = False
    if (
        discrimination_required
        and variant is None
        and expected.get("kind") == "action"
        and scenario.get("targeting") == "individual"
        # Only single-call, value-free requests can be reworded as a description.
        # A request that carries a setting ("set X to 40 percent") or several
        # calls cannot be replaced by "turn on the <description>" without losing
        # the argument, and validate_row would reject it anyway.
        and len(expected.get("calls") or []) == 1
        and not _call_carries_value((expected.get("calls") or [{}])[0])
    ):
        seed_text = request_seed_from_spec(spec)
        description = _pick_discriminating_description(scenario, rng)
        if description:
            # Keep the leading verb so the request still encodes the operation.
            verb = "turn on"
            for candidate in ("turn on", "turn off", "open", "close", "lock", "unlock", "set", "run"):
                if seed_text.lower().startswith(candidate):
                    verb = candidate
                    break
            spec["utterance"] = f"{verb} {description}"
            discrimination = True
            # validate_row reads this: a discrimination row refers to the target
            # by description, so the name-presence check must not apply.
            spec["discrimination"] = True

    # Check before style/noise transforms too: variants of held-out requests stay held out.
    if _check_quality_eval_overlap(spec["utterance"]):
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

    # Apply the same casing distribution to every label, including refusals.
    utterance = vary_training_utterance(spec["utterance"], rng)
    spec["utterance"] = (
        utterance.lower() if rng.random() < 0.5
        else utterance[:1].upper() + utterance[1:]
    )
    if spec["utterance"].casefold() in excluded:
        return None, "excluded_prompt"
    if _check_quality_eval_overlap(spec["utterance"]):
        return None, "quality_eval_overlap"

    reason = validate_row(spec, token_budget=config.token_budget)
    if reason:
        return None, reason

    reject = dup_tracker.would_reject(spec)
    if reject:
        return None, reject

    # One semantic scenario may appear more than once, so the row id is the
    # scenario, the utterance+home pair, and how many rows already share that
    # pair. Deterministic because generation is sequential. semantic_id keeps
    # identifying the scenario itself.
    spec["candidate_id"] = (
        f"{spec['semantic_id']}_{pair_hash(spec['utterance'], spec['home'])[:8]}"
        f"_{dup_tracker.occurrences(spec)}"
    )

    # A rejected real-home row is retried, and the retry re-rolls the real/synthetic
    # draw, so capping converts surplus rows for one entity into synthetic rows
    # rather than shrinking the corpus.
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
    # Ask the quota before recording the row anywhere: a bucket that is already
    # full must not consume the duplicate tracker's budget for the next attempt.
    if quota is not None:
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


def run_generation(config: GeneratorConfig) -> dict[str, Any]:
    """Generate accepted training rows up to config.count."""
    rng = random.Random(config.seed)
    grounding_target = int(round(config.count * config.grounding_rate))
    capacity = _grounding_capacity(config)
    if grounding_target > capacity:
        # Before the build, not after: this is the defect that shipped twice. The
        # rate gate at the end would only report the shortfall, and the run costs
        # minutes to reach it.
        raise ValueError(
            f"grounding_rate {config.grounding_rate} on {config.count} rows asks for "
            f"{grounding_target} grounding rows, but the catalogue can produce at most "
            f"{capacity} ({len(training_variants())} scenarios x near_duplicate_limit "
            f"{config.near_duplicate_limit}). Add sites in generators.grounding._sites "
            "or lower the rate."
        )
    quota = QuotaTracker(
        config.count, config.seed, config.tier_proportions, negative_rate=config.negative_rate
    )

    excluded = _load_excluded_prompts(config.exclude_prompts_path)
    dup_tracker = DuplicateTracker(near_limit=config.near_duplicate_limit)
    real_home_targets: Counter[str] = Counter()
    # Per operation, so "the TV has had three turn_on rows" cannot crowd out its
    # first pause row. The flat counter above still backs the entity cap.
    real_home_usage: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    real_home_names: frozenset[str] = frozenset()
    real_home_entity_cap = config.real_home_entity_cap
    if config.real_home_path:
        real_entities = load_real_home(config.real_home_path, split="train")["entities"]
        # build_scenario injects an entity for a capability the real home lacks;
        # that name is synthetic, so it must not inflate the real-entity counts.
        real_home_names = frozenset(entity["name"] for entity in real_entities)
        if not real_home_entity_cap:
            real_home_entity_cap = derive_entity_cap(
                config.count, config.real_home_rate, len(real_entities)
            )
    stats = empty_stats()
    accepted: list[dict[str, Any]] = []
    semantic_ids: set[str] = set()
    real_home_rows = 0
    grounding_rows: Counter[str] = Counter()
    required_grounding = required_training_variants()
    grounding_missing = (
        {variant["family"]: variant for variant in required_grounding}
        if config.count * config.grounding_rate
        >= GROUNDING_FAMILY_SLACK * len(required_grounding)
        else {}
    )
    # v1 shipped a 14x grounding shortfall because forcing stopped the instant
    # one row landed (`not grounding_rows`) and the rest reverted to a 3% random
    # draw. Track the rate against rows accepted so far so demand is maintained
    # across the whole run. (``grounding_target`` is set above, with the
    # catalogue-capacity pre-flight that uses it.)
    discrimination_target = int(round(config.count * config.discrimination_rate))
    discrimination_rows = 0
    real_home_target = int(round(config.count * config.real_home_rate))
    balance_real_home = bool(
        config.real_home_path and config.real_home_rate and config.real_home_entity_cap == 0
    )
    attempts = 0
    max_attempts = config.max_attempts()

    load_paraphraser(config.paraphrase_enabled)
    stt_target = int(round(config.count * config.stt_noise_rate))

    while not quota.is_complete() and attempts < max_attempts:
        slot = quota.next_slot()
        rows_remaining = max(1, config.count - quota.accepted_total())
        real_home_selected = None
        if balance_real_home:
            real_home_remaining = max(0, real_home_target - real_home_rows)
            real_home_selected = rng.random() < real_home_remaining / rows_remaining

        # Grounding variants exist for only a handful of (capability, operation)
        # pairs. When the run is behind its grounding target, take a slot from a
        # pair that can actually host a variant; otherwise the demand is
        # unsatisfiable and the requested rate silently collapses.
        grounding_deficit = grounding_target - sum(grounding_rows.values())
        discrimination_deficit = discrimination_target - discrimination_rows
        want_grounding = (
            config.grounding_rate > 0
            and grounding_deficit > 0
            and rng.random() < min(1.0, grounding_deficit / rows_remaining)
        )
        want_discrimination = (
            config.discrimination_rate > 0
            and discrimination_deficit > 0
            and not want_grounding
            # Demand-driven, like grounding: once a few rows land the deficit
            # shrinks and a proportional draw stops firing, so the requested
            # share is never delivered. Force while meaningfully behind instead.
            and rng.random() < max(0.5, min(1.0, 2.0 * discrimination_deficit / rows_remaining))
        )
        if want_grounding:
            forced = _grounding_capable_slot(quota, rng)
            if forced is not None:
                slot = forced
        elif want_discrimination:
            forced = _discrimination_capable_slot(quota, rng)
            if forced is not None:
                slot = forced
        row, reason = generate_row(
            slot,
            config,
            rng,
            excluded=excluded,
            dup_tracker=dup_tracker,
            attempt=attempts,
            stt_remaining=max(0, stt_target - stats["stt_corrupted"]),
            rows_remaining=rows_remaining,
            real_home_targets=real_home_targets,
            real_home_entity_cap=real_home_entity_cap,
            real_home_names=real_home_names,
            real_home_usage=real_home_usage[(slot["capability"], slot["operation"])],
            quota=quota,
            grounding_required=want_grounding,
            discrimination_required=want_discrimination,
            real_home_selected=real_home_selected,
            grounding_variants=list(grounding_missing.values()) if grounding_missing else None,
        )
        attempts += 1
        if row is None:
            record_reject(stats, reason or "unknown")
            continue
        sem = row.get("metadata", {}).get("semantic_id")
        if sem:
            semantic_ids.add(sem)
        quota.record_accept(row)
        accepted.append(row)
        record_accept(stats, row)
        real_home_rows += bool(row["metadata"].get("real_home"))
        if row["metadata"].get("grounding_family"):
            family = row["metadata"]["grounding_family"]
            grounding_rows[family] += 1
            grounding_missing.pop(family, None)
        if row["metadata"].get("discrimination"):
            discrimination_rows += 1

    if not quota.is_complete():
        raise RuntimeError(
            f"failed to meet accepted-row quota: accepted {quota.accepted_total()}/{config.count} "
            f"after {attempts} attempts; shortfall={quota.shortfall()}; "
            f"missing_grounding={sorted(grounding_missing)}; "
            f"rejections={dict(stats['rejection_reasons'])}"
        )

    quota.verify_complete()
    if grounding_missing:
        raise RuntimeError(
            f"missing required grounding families: {sorted(grounding_missing)}"
        )
    # Balance within each label so small refusal samples cannot acquire a case shortcut by chance.
    for has_calls in (False, True):
        users = [next(message for message in row["messages"] if message["role"] == "user")
                 for row in accepted if any(message.get("tool_calls") for message in row["messages"]) == has_calls]
        rng.shuffle(users)
        for index, user in enumerate(users):
            text = user["content"]
            user["content"] = text.lower() if index % 2 == 0 else text[:1].upper() + text[1:]
    report = finalize_stats(stats, semantic_ids, quota_summary=quota.summary())
    report["requested_stt_rate"] = config.stt_noise_rate
    report["achieved_stt_rate"] = round(stats["stt_corrupted"] / max(stats["accepted"], 1), 4)
    report["requested"] = config.count
    report["attempts"] = attempts
    report["config"] = config.to_dict()
    report["grounding"] = {
        "requested_rate": config.grounding_rate,
        "rows": sum(grounding_rows.values()),
        "achieved_rate": round(sum(grounding_rows.values()) / max(len(accepted), 1), 4),
        "by_family": dict(sorted(grounding_rows.items())),
    }
    report["discrimination"] = {
        "requested_rate": config.discrimination_rate,
        "rows": discrimination_rows,
        "achieved_rate": round(discrimination_rows / max(len(accepted), 1), 4),
    }
    # Grounding can only land on slots whose (capability, operation) has a
    # variant, and real-home mixing claims some of those same slots. Pass the
    # reachable ceiling so the gate measures delivery of what was achievable
    # rather than failing honest runs against an unreachable request.
    _enforce_rate_gate(
        report["grounding"],
        "grounding",
        config,
        len(accepted),
        available_share=_grounding_available_share(config),
    )
    _enforce_rate_gate(
        report["discrimination"],
        "discrimination",
        config,
        len(accepted),
        available_share=_discrimination_available_share(config),
    )
    # Counted on accepted rows after validation, deduplication and the quota, not
    # assumed from the requested rate. The first build of this pair delivered 11.5%
    # against a 35% request because coverage compared raw tool names and rejected
    # every namespaced row; nothing failed, it just shipped short.
    for key, requested in (
        ("namespaced_tools", config.namespaced_tool_rate),
        ("full_tool_catalog", config.full_catalog_rate),
    ):
        rows = sum(1 for row in accepted if row["metadata"].get(key))
        report[key] = {
            "requested_rate": requested,
            "rows": rows,
            "achieved_rate": round(rows / max(len(accepted), 1), 4),
        }
        _enforce_rate_gate(report[key], key, config, len(accepted))
    if config.real_home_path:
        report["real_home"] = {
            "path": str(config.real_home_path),
            "requested_rate": config.real_home_rate,
            "synthetic_only": config.synthetic_only,
            # One row is one row: a multi-target row names several entities but is
            # still a single real-home example.
            "rows": real_home_rows,
            "achieved_rate": round(real_home_rows / max(len(accepted), 1), 4),
            "entity_cap": real_home_entity_cap,
            "entities_used": len(real_home_targets),
            "target_counts": dict(sorted(real_home_targets.items())),
            "most_common": real_home_targets.most_common(5),
        }
    report["registry"] = registry_summary()
    from generators.audit import audit_rows

    report["quality_audit"] = audit_rows(
        accepted,
        expected_count=config.count,
        required_operations={
            key for key, target in quota.targets["positive"].items() if target > 0
        },
        min_positive_per_operation=config.min_positive_per_operation,
        min_positive_per_tool=config.min_positive_per_tool,
        max_absence_rate=config.max_absence_rate,
    )
    return {"rows": accepted, "stats": report}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {k: v for k, v in report.items() if k != "rows"}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
