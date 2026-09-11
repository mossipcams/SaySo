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
    """The alias, not the canonical name, is what the request will use."""
    return [
        variant(
            prefix=prefix, label="alias", capability="lights", operation="turn_on",
            targeting="individual", robustness="alias_distractor", area=area,
            entities=[entity(name, "lights", area, aliases=[name, alias]),
                      entity("Ceiling Light", "lights", area)],
        ),
    ]


def training_variants() -> list[dict[str, Any]]:
    """Grounding variants safe to train on: names, ids and areas held out of eval."""
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
    return variants


def required_training_variants() -> list[dict[str, Any]]:
    """One complete contrast set; the second media family remains extra diversity."""
    return [
        variant for variant in training_variants()
        if not variant["family"].startswith("ground_media_b_")
    ]


def build_spec(variant: dict[str, Any], *, seed: int = 20260910, index: int = 0) -> dict[str, Any]:
    """Scenario -> spec for one variant, with the label derived from the graph.

    Deliberately the same order the pipeline uses: build the entity graph, let
    ``gold_from_scenario`` decide the answer, and only then choose wording.
    """
    import copy

    from generators.labels import scenario_to_spec
    from generators.pipeline import _unique_no_action_hint
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
        spec["request_hint"] = _unique_no_action_hint(spec, random.Random(seed))
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
