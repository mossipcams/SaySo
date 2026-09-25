"""Paired scenarios where the request is fixed and the entity graph moves.

The failure this addresses is a model that answers "turn on the living room media
player" from the wording alone. Each family below holds the request constant and
changes one thing about the home -- the target's name, its id, its aliases, its
area, its domain, or the actions it supports -- so the correct answer changes with
it. Nothing here hard-codes an expected call: every variant is handed to
``build_scenario``/``gold_from_scenario``, which derive the label from the graph
and the pinned tool contract. Wording is applied afterwards, as for any other row.
"""

from __future__ import annotations

import random
import re
from functools import lru_cache
from typing import Any

from generators.homes import make_entity


def entity(
    name: str,
    capability: str,
    area: str,
    *,
    floor: str = "Main Floor",
    aliases: list[str] | None = None,
    features: tuple[str, ...] | None = None,
    entity_id: str | None = None,
    device_class: str | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    made = make_entity(
        name=name,
        capability=capability,
        area=area,
        floor=floor,
        rng=random.Random(0),
        aliases=aliases or [name],
        features=features,
        state=state or "off",
        device_class=device_class,
    )
    if entity_id:
        made["entity_id"] = entity_id
    return made


def home(home_id: str, entities: list[dict[str, Any]], *, sayso_entity_area: str) -> dict[str, Any]:
    """A home dict in the shape ``generate_home`` returns."""
    floors = {}
    for item in entities:
        floors.setdefault(item["area"], item.get("floor") or "Main Floor")
    floors.setdefault(sayso_entity_area, "Main Floor")
    return {
        "home_id": home_id,
        "size": len(entities),
        "sayso_entity_area": sayso_entity_area,
        "entities": entities,
        "active_timers": [],
        "areas": list(floors),
        "area_floors": floors,
        "owners": (),
        "exposure_source": "synthetic",
    }


def variant(
    *,
    prefix: str,
    label: str,
    capability: str,
    operation: str,
    targeting: str,
    robustness: str,
    entities: list[dict[str, Any]],
    area: str,
    inject_missing: bool = True,
    utterance: str | None = None,
    rng_index: int | None = None,
    request_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One member of a family. ``phrasing_seed`` is the family, not the member, so
    every member renders the same request and only the label moves.
    """
    return {
        "family": f"{prefix}_{label}",
        "phrasing_seed": prefix,
        "capability": capability,
        "operation": operation,
        "targeting": targeting,
        "robustness": robustness,
        "inject_missing": inject_missing,
        "utterance": utterance,
        "rng_index": rng_index,
        "request_intent": request_intent,
        "home": home(f"{prefix}_{label}", entities, sayso_entity_area=area),
    }


# Distractors that must not steal the target: other domains in the same room, and
# the same domain in another room.
def _distractors(area: str, elsewhere: str) -> list[dict[str, Any]]:
    return [
        entity("Floor Lamp", "lights", area),
        entity("Corner Lamp", "lights", area),
        entity("Console Outlet", "switches", area),
        entity(f"{elsewhere} Speaker", "media_players", elsewhere,
               features=("play", "pause", "volume", "volume_step", "mute", "next", "previous")),
    ]


def media_presence_family(
    *,
    prefix: str,
    target_name: str,
    renamed: str,
    area: str,
    elsewhere: str,
    entity_id: str,
    aliases: list[str] | None = None,
) -> list[dict[str, Any]]:
    """One request ("turn on the <area> media player"), four entity graphs.

    present -> the one eligible player; renamed -> the same device under a new
    name; moved -> no eligible player in the area at all; distractors -> the same
    answer as `present` with lights, switches and another player added.
    """
    features = ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous")

    def player(name: str, in_area: str) -> dict[str, Any]:
        return entity(name, "media_players", in_area, aliases=aliases or [name],
                      features=features, entity_id=entity_id, device_class="tv")

    common = dict(
        prefix=prefix, capability="media_players", operation="turn_on",
        targeting="area", robustness="ambiguity", area=area,
        utterance=f"Turn on the {area.casefold()} media player",
    )
    return [
        variant(label="present", entities=[player(target_name, area)], **common),
        variant(label="renamed", entities=[player(renamed, area)], **common),
        variant(
            label="moved",
            entities=[player(target_name, elsewhere), entity("Reading Lamp", "lights", area)],
            inject_missing=False,
            **common,
        ),
        variant(
            label="distractors",
            entities=[player(target_name, area), *_distractors(area, elsewhere)],
            **common,
        ),
    ]


_SCREEN_FEATURES = ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous")


def _screen(name: str, area: str, *, aliases: list[str] | None = None,
            entity_id: str | None = None) -> dict[str, Any]:
    return entity(name, "media_players", area, aliases=aliases or [name],
                  features=_SCREEN_FEATURES, entity_id=entity_id, device_class="tv")


def _speaker(name: str, area: str) -> dict[str, Any]:
    """A same-domain, same-area distractor that cannot be the requested screen."""
    return entity(name, "media_players", area, device_class="speaker",
                  features=("on", "off", "play", "pause", "volume", "volume_step", "mute"))


def named_device_family(
    *,
    prefix: str,
    area: str,
    elsewhere: str,
    device: str,
    entity_id: str,
    operation: str = "turn_on",
    utterance: str,
    from_elsewhere: bool = False,
) -> list[dict[str, Any]]:
    """One fixed request against a device whose *name* and *area* are separate.

    This is the shape issue #52 failed on: `media_player.living_room_tv` is named
    "TV" and lives in "Living Room", so neither "turn on the TV" nor "turn on the
    living room TV" contains the entity's full canonical name plus its room. The
    request is held fixed and only the home moves:

    ``bare``        the device named exactly ``device`` in ``area``;
    ``room_named``  the same device renamed to "<area> <device>", so the canonical
                    name now contains the room -- the answer must not change;
    ``aliased``     canonical name is something else entirely and ``device`` is
                    only an alias;
    ``speaker``     a same-area, same-domain speaker distractor whose *name*
                    carries the room ("<area> Media Player"). This is the exact
                    trap the deployed model fell into, answering
                    ``name="Living Room Media Player"`` for "turn off the living
                    room TV";
    ``absent``      no screen anywhere, but other media players are present, so
                    absence is real and is not "no media players at all";
    ``ambiguous``   two eligible screens in the room, so one short question is
                    the only correct answer;
    ``irrelevant``  ``bare`` plus unrelated lights and a switch elsewhere; the
                    expected action must be identical to ``bare``.

    ``from_elsewhere`` puts the satellite in another room, so the room named in
    the request has to override the satellite's default area.
    """
    satellite_area = elsewhere if from_elsewhere else area
    # The structured intent behind the fixed wording. ``area`` is stated whenever
    # the request names a room, so an explicit room overrides the satellite's.
    names_room = area.casefold() in utterance.casefold()
    common = dict(
        prefix=prefix, capability="media_players", operation=operation,
        targeting="individual", robustness="ambiguity", area=satellite_area,
        utterance=utterance, rng_index=0,
        request_intent={"name": device, "area": area if names_room else None},
    )
    return [
        variant(label="bare", entities=[_screen(device, area)], **common),
        variant(label="room_named", entities=[_screen(f"{area} {device}", area)], **common),
        variant(
            label="aliased",
            entities=[_screen("Screen One", area, aliases=["Screen One", device],
                              entity_id=entity_id)],
            **common,
        ),
        variant(
            label="speaker",
            entities=[_screen(device, area, entity_id=entity_id),
                      _speaker(f"{area} Media Player", area)],
            **common,
        ),
        variant(
            label="absent",
            entities=[_speaker(f"{area} Media Player", area),
                      _screen(device, elsewhere),
                      entity("Reading Lamp", "lights", area)],
            inject_missing=False,
            **common,
        ),
        variant(
            label="ambiguous",
            entities=[_screen(f"{area} {device}", area), _screen(f"Second {device}", area)],
            **common,
        ),
        variant(
            label="irrelevant",
            entities=[_screen(device, area), entity("Floor Lamp", "lights", area),
                      entity("Hall Switch", "switches", elsewhere)],
            **common,
        ),
    ]


def ambiguity_family(*, prefix: str, area: str, names: tuple[str, str]) -> list[dict[str, Any]]:
    """One eligible light -> act on it; two -> genuine ambiguity, ask which."""
    first, second = names
    common = dict(
        prefix=prefix, capability="lights", operation="turn_on",
        targeting="area", robustness="ambiguity", area=area,
    )
    return [
        variant(label="single", entities=[entity(first, "lights", area)], **common),
        variant(
            label="pair",
            entities=[entity(first, "lights", area), entity(second, "lights", area)],
            **common,
        ),
    ]


def domain_family(*, prefix: str, area: str, fan_name: str, light_name: str) -> list[dict[str, Any]]:
    """The same room with a fan, and without one: the domain decides the answer."""
    common = dict(
        prefix=prefix, capability="fans", operation="turn_on",
        targeting="area", robustness="ambiguity", area=area,
    )
    return [
        variant(
            label="present",
            entities=[entity(fan_name, "fans", area), entity(light_name, "lights", area)],
            **common,
        ),
        variant(
            label="absent",
            entities=[entity(light_name, "lights", area)],
            inject_missing=False,
            **common,
        ),
    ]


def supported_action_family(*, prefix: str, area: str, name: str) -> list[dict[str, Any]]:
    """Same device, same request: one build supports volume, the other does not."""
    common = dict(
        prefix=prefix, capability="media_players", operation="volume_set",
        targeting="area", robustness="ambiguity", area=area, rng_index=0,
    )
    return [
        variant(
            label="capable",
            entities=[entity(name, "media_players", area, device_class="speaker",
                             features=("on", "off", "play", "pause", "volume", "volume_step", "mute"))],
            **common,
        ),
        variant(
            label="incapable",
            entities=[entity(name, "media_players", area, device_class="speaker",
                             features=("on", "off", "play", "pause"))],
            inject_missing=False,
            **common,
        ),
    ]


def alias_family(*, prefix: str, area: str, name: str, alias: str) -> list[dict[str, Any]]:
    """The alias, not the canonical name, is what the request will use.

    ``rng_index=0`` pins the target to the aliased entity. Without it
    ``pick_target`` rotates on the caller's row index, so roughly half of these
    rows targeted the *distractor* ("Ceiling Light") and taught nothing about
    aliases while still being counted and labelled as alias grounding rows.
    """
    return [
        variant(
            prefix=prefix, label="alias", capability="lights", operation="turn_on",
            targeting="individual", robustness="alias_distractor", area=area,
            rng_index=0,
            entities=[entity(name, "lights", area, aliases=[name, alias]),
                      entity("Ceiling Light", "lights", area)],
        ),
    ]


def canonical_variants() -> list[dict[str, Any]]:
    """The hand-written contrast set: one site, names/ids/areas held out of eval."""
    variants: list[dict[str, Any]] = []
    variants += media_presence_family(
        prefix="ground_media_a",
        target_name="Den Projector",
        renamed="Beamer",
        area="Den",
        elsewhere="Attic",
        entity_id="media_player.den_projector",
    )
    variants += media_presence_family(
        prefix="ground_media_b",
        target_name="Sunroom Screen",
        renamed="Panorama",
        area="Sunroom",
        elsewhere="Workshop",
        entity_id="media_player.sunroom_screen",
        aliases=["Sunroom Screen", "the screen"],
    )
    variants += ambiguity_family(
        prefix="ground_light", area="Craft Room", names=("Bench Lamp", "Shelf Lamp")
    )
    variants += domain_family(
        prefix="ground_domain", area="Pantry", fan_name="Pantry Extractor", light_name="Pantry Strip"
    )
    variants += supported_action_family(
        prefix="ground_features", area="Landing", name="Landing Speaker"
    )
    variants += alias_family(
        prefix="ground_alias", area="Music Room", name="Practice Lamp", alias="rehearsal light"
    )
    # Issue #52. Canonical, so `required_training_variants` forces every contrast
    # to land: at 2.8% grounding the site rotation delivered 12 rows at n=4000 and
    # the `speaker` and `irrelevant` contrasts -- the two that carry the actual
    # defect -- landed zero times. A sampling rate is not coverage.
    variants += named_device_family(
        prefix="ground_named",
        area="Loft",
        elsewhere="Cellar",
        device="Television",
        # turn_off, not turn_on: a required grounding family outranks the
        # real-home draw for its (capability, operation) pair, and real-home
        # mixing needs media_players/turn_on. `media_presence_family` already
        # claims turn_on, so a second required turn_on family starved it.
        entity_id="media_player.loft_television",
        operation="turn_off",
        utterance="Turn off the television",
    )
    return variants


# Areas the canonical set and the eval set already occupy. A site reusing one
# would either collide with a held-out eval prompt (the area is what the media
# and ambiguity families say out loud) or duplicate a canonical scenario.
_RESERVED_AREAS: frozenset[str] = frozenset({
    # canonical (above)
    "Den", "Attic", "Sunroom", "Workshop", "Craft Room", "Pantry", "Landing", "Music Room",
    "Loft", "Cellar",
    # evals/cases/regressions.jsonl grounding variants
    "Living Room", "Bedroom", "Office", "Basement", "Kitchen", "Bathroom", "Porch",
})

# Device words per site, rotated so neighbouring sites do not read alike. Drawn
# from the same vocabulary the synthetic homes use (homes._ROLES), so a grounding
# row looks like the rest of the corpus.
#
# Bare "TV" is held out the way `_RESERVED_AREAS` holds out Living Room. The
# named-device families say the device word on its own ("Turn on the tv"), which
# carries no area, so a site drawing "TV" reproduces the frozen issue #52 eval
# prompt verbatim. "OLED TV" and "Streaming TV" keep the token in training
# without colliding.
_RESERVED_DEVICE_WORDS = frozenset({"TV"})
_SITE_SCREENS = tuple(
    word for word in ("TV", "Television", "OLED TV", "Streaming TV", "Display", "Projector")
    if word not in _RESERVED_DEVICE_WORDS
)
_SITE_RENAMES = ("Beamer", "Panorama", "Big Picture", "The Set", "Screen One", "Viewer")
_SITE_LAMPS = ("Bench Lamp", "Shelf Lamp", "Desk Lamp", "Corner Lamp", "Floor Lamp",
               "Table Lamp", "Wall Light", "Pendant Light", "Ceiling Lights", "Strip Light")
_SITE_FANS = ("Extractor", "Ceiling Fan", "Tower Fan", "Desk Fan")
_SITE_ALIASES = ("accent light", "task light", "side light", "spare light", "nook light")


def _slug(text: str) -> str:
    return "_".join(part for part in re.split(r"[^a-z0-9]+", text.casefold()) if part)


def site_variants(index: int, area: str, elsewhere: str) -> list[dict[str, Any]]:
    """The whole contrast set instantiated at one more area, with fresh names.

    Every variant is one fixed scenario, and ``DuplicateTracker`` caps rows per
    scenario at ``near_duplicate_limit`` (8). A single-site catalogue therefore
    has a hard ceiling of ~15*8 grounding rows *no matter how large the corpus
    is* -- which is exactly why the 40k v1 corpus shipped 136 grounding rows
    against a 1120-row request, and why 2k looked healthy at 2.4%. Sites are the
    capacity knob: ceiling = variants * near_duplicate_limit.
    """
    tag = f"s{index:02d}"
    screen = _SITE_SCREENS[index % len(_SITE_SCREENS)]
    lamp_a = _SITE_LAMPS[index % len(_SITE_LAMPS)]
    lamp_b = _SITE_LAMPS[(index + 3) % len(_SITE_LAMPS)]
    lamp_c = _SITE_LAMPS[(index + 6) % len(_SITE_LAMPS)]
    variants: list[dict[str, Any]] = []
    variants += media_presence_family(
        prefix=f"ground_media_{tag}",
        target_name=f"{area} {screen}",
        renamed=_SITE_RENAMES[index % len(_SITE_RENAMES)],
        area=area,
        elsewhere=elsewhere,
        entity_id=f"media_player.{_slug(area)}_{_slug(screen)}_{tag}",
    )
    variants += ambiguity_family(
        prefix=f"ground_light_{tag}", area=area, names=(f"{area} {lamp_a}", f"{area} {lamp_b}")
    )
    variants += domain_family(
        prefix=f"ground_domain_{tag}",
        area=area,
        fan_name=f"{area} {_SITE_FANS[index % len(_SITE_FANS)]}",
        light_name=f"{area} {lamp_c}",
    )
    variants += supported_action_family(
        prefix=f"ground_features_{tag}", area=area, name=f"{area} Speaker"
    )
    variants += alias_family(
        prefix=f"ground_alias_{tag}",
        area=area,
        name=f"{area} {lamp_b}",
        alias=_SITE_ALIASES[index % len(_SITE_ALIASES)],
    )
    # Issue #52: name and area stored separately. Three wordings of one request --
    # bare device noun, room-qualified, and room-as-prepositional-phrase -- plus
    # both polarities and a satellite standing in another room.
    # One wording per site, rotated. All three at every site would triple the
    # grounding catalogue's share of (media_players, turn_on) slots and starve the
    # real-home TV rows that need the same pair -- caught by
    # test_mixing_produces_positive_tv_rows_for_its_supported_operations.
    lowered = area.casefold()
    wording, operation, away = (
        ("bare", "turn_on", False),
        ("room", "turn_off", False),
        # The satellite sits in `elsewhere`, so "in the <area>" must beat its default.
        ("away", "turn_on", True),
    )[index % 3]
    utterance = {
        "bare": f"Turn on the {screen.casefold()}",
        "room": f"Turn off the {lowered} {screen.casefold()}",
        "away": f"Turn on the {screen.casefold()} in the {lowered}",
    }[wording]
    variants += named_device_family(
        prefix=f"ground_named_{wording}_{tag}", area=area, elsewhere=elsewhere,
        device=screen,
        entity_id=f"media_player.{_slug(area)}_{_slug(screen)}_{wording}_{tag}",
        operation=operation, utterance=utterance, from_elsewhere=away,
    )
    return variants


def _sites() -> list[tuple[str, str]]:
    """(area, elsewhere) pairs, one per extra instantiation of the contrast set."""
    from generators.homes import _AREAS

    areas = [area for area in _AREAS if area not in _RESERVED_AREAS]
    return [(area, areas[(i + 1) % len(areas)]) for i, area in enumerate(areas)]


@lru_cache(maxsize=1)
def _catalogue() -> tuple[dict[str, Any], ...]:
    variants = canonical_variants()
    for index, (area, elsewhere) in enumerate(_sites()):
        variants += site_variants(index, area, elsewhere)
    return tuple(variants)


def training_variants() -> list[dict[str, Any]]:
    """Grounding variants safe to train on: names, ids and areas held out of eval.

    Cached and shared: every consumer deep-copies the home before building a
    scenario, so handing out the same dicts is safe and saves rebuilding a few
    hundred homes on every slot.
    """
    return list(_catalogue())


def required_training_variants() -> list[dict[str, Any]]:
    """One complete contrast set; every other site is extra diversity.

    Kept to the canonical site so the "every contrast family must land" gate
    stays reachable on small runs. The extra sites carry no gate: they exist to
    lift the per-scenario duplicate ceiling, not to add contrast categories.
    """
    canonical = {variant["family"] for variant in canonical_variants()}
    # The named family's refusal contrasts (`absent`, `ambiguous`) are not
    # required. Every refusal family competes for a bucket's small negative
    # allowance, and requiring two more exhausted it: a 1500-row mixed run failed
    # closed on `quota_negative_full` after 30k attempts. They stay in the
    # catalogue and land through the site rotation; the positive contrasts --
    # `speaker` and `irrelevant`, which carry the actual issue #52 defect -- are
    # the ones the gate must force.
    return [
        variant for variant in training_variants()
        if variant["family"] in canonical
        and not variant["family"].startswith("ground_media_b_")
        and variant["family"] not in {"ground_named_absent", "ground_named_ambiguous"}
    ]


def build_spec(variant: dict[str, Any], *, seed: int = 20260910, index: int = 0) -> dict[str, Any]:
    """Scenario -> spec for one variant, with the label derived from the graph.

    Deliberately the same order the pipeline uses: build the entity graph, let
    ``gold_from_scenario`` decide the answer, and only then choose wording.
    """
    import copy

    from generators.labels import scenario_to_spec
    from generators.scenarios.unavailable import unique_no_action_hint
    from generators.scenarios import build_scenario
    from generators.utterances import apply_generic_wording, expand_utterance

    scenario = build_scenario(
        index=variant.get("rng_index") if variant.get("rng_index") is not None else index,
        seed=seed,
        capability=variant["capability"],
        operation=variant["operation"],
        home_size=len(variant["home"]["entities"]),
        targeting=variant["targeting"],
        robustness=variant["robustness"],
        home=copy.deepcopy(variant["home"]),
        inject_missing=variant.get("inject_missing", True),
        request_intent=variant.get("request_intent"),
    )
    scenario["phrasing_seed"] = variant["phrasing_seed"]
    spec = scenario_to_spec(scenario)
    spec["grounding_family"] = variant["family"]
    if spec["expected"].get("kind") == "no_action":
        requested = _requested_names(spec)
        if requested:
            spec["spoken_targets"] = apply_generic_wording(
                {**spec, "target_names": requested}
            )["spoken_targets"]
        spec["request_hint"] = unique_no_action_hint(spec, random.Random(seed))
    elif variant["robustness"] == "ambiguity":
        apply_generic_wording(spec)
    spec["utterance"] = variant.get("utterance") or expand_utterance(spec)
    return spec


def _requested_names(spec: dict[str, Any]) -> list[str]:
    from generators.gold import target_names_from_expected

    return target_names_from_expected(spec["expected"].get("requested") or {})


def build_row(variant: dict[str, Any], *, seed: int = 20260910, index: int = 0) -> dict[str, Any]:
    """A rendered row in the production prompt/context format and tool schema."""
    from generators.labels import render_example

    spec = build_spec(variant, seed=seed, index=index)
    row = render_example(spec)
    row["metadata"]["grounding_family"] = variant["family"]
    return row


def pick_variant(
    capability: str, operation: str, index: int, *, variants: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """Deterministically choose a grounding variant for a quota slot, or None."""
    pool = [
        variant
        for variant in (variants if variants is not None else training_variants())
        if variant["capability"] == capability and variant["operation"] == operation
    ]
    if not pool:
        return None
    return pool[index % len(pool)]


# Requiring every grounding contrast family needs slack, not parity.
GROUNDING_FAMILY_SLACK = 2

# Recipe families that can host a grounding row unchanged: the variant's gold
# alone decides the family (action -> ordinary, area_unavailable -> absence, ...).
# follow_up/correction/junk rewrite the gold after the fact; aliases/settings
# would lose their own contract; multi_action/area never match one variant.
GROUNDING_CARRIER_FAMILIES = frozenset({"ordinary", "absence", "unsupported", "clarify"})


@lru_cache(maxsize=None)
def carrier_variants(family: str) -> tuple[dict[str, Any], ...]:
    """Training variants whose graph-derived gold satisfies ``family``."""
    if family not in GROUNDING_CARRIER_FAMILIES:
        return ()
    from generators.gold import gold_matches_family

    return tuple(
        variant
        for variant in training_variants()
        if gold_matches_family(build_spec(variant)["expected"], family)
    )


def pick_carrier_variant(
    slot: dict[str, Any], rng: random.Random, missing: Any = ()
) -> dict[str, Any] | None:
    """Variant for a carrier slot: missing families first, then the slot's own pair.

    A slot the plan marked for grounding draws from the whole family pool: the
    overlay replaces its operation anyway, and pinning it to its own pair kept
    re-drawing the few same-pair variants past the duplicate limit.
    """
    pool = list(carrier_variants(slot.get("family") or ""))
    pool = [v for v in pool if v["family"] in missing] or pool
    if slot.get("grounding"):
        return rng.choice(pool) if pool else None
    same_pair = [
        v for v in pool
        if (v["capability"], v["operation"]) == (slot.get("capability"), slot.get("operation"))
    ]
    return rng.choice(same_pair or pool) if pool else None


@lru_cache(maxsize=1)
def grounding_pairs() -> frozenset[tuple[str, str]]:
    return frozenset(
        (variant["capability"], variant["operation"])
        for variant in required_training_variants()
    )


def slot_can_carry_grounding(slot: dict[str, Any] | None) -> bool:
    """True when a recipe slot can overlay a grounding variant without a family steal."""
    if not slot or slot.get("area_scenario"):
        return False
    return bool(carrier_variants(slot.get("family") or ""))


def grounding_capacity(config: Any) -> int:
    """Most grounding rows the catalogue can ever produce at any corpus size."""
    return len(training_variants()) * config.near_duplicate_limit


def slot_ceiling(config: Any, pairs: set[tuple[str, str]]) -> float:
    """Share of the quota plan landing on ``pairs``, net of real-home mixing."""
    from generators.sampling import build_quota_plan

    plan = build_quota_plan(config.count, config.seed, config.tier_proportions)
    matching = sum(
        1 for slot in plan if (slot["capability"], slot["operation"]) in pairs
    )
    share = matching / max(len(plan), 1)
    if config.real_home_path and config.real_home_rate:
        share *= max(0.0, 1.0 - config.real_home_rate)
    return min(1.0, share)


def grounding_available_share(config: Any) -> float:
    """Share of accepted rows that can realistically be grounding rows."""
    catalogue_share = grounding_capacity(config) / max(config.count, 1)
    return min(slot_ceiling(config, grounding_pairs()), catalogue_share)


def grounding_capable_slot(quota: Any, rng: random.Random) -> dict[str, Any] | None:
    """Take a quota slot whose (capability, operation) has a grounding variant."""
    pairs = sorted(
        {(variant["capability"], variant["operation"]) for variant in required_training_variants()}
    )
    rng.shuffle(pairs)
    for capability, operation in pairs:
        slot = quota.take_slot(capability, operation)
        if slot is not None:
            return slot
    return None
