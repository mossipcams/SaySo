"""Area-context contract and fail-closed routing checks."""

from __future__ import annotations

import random

from generators.context import render_area_context, resolve_area_context
from generators.config import GeneratorConfig
from generators.gold import gold_from_scenario
from generators.grounding import area_context_variants, build_spec
from generators.labels import render_example
from generators.pipeline import run_generation


def test_satellite_fallback_and_explicit_area_are_distinct() -> None:
    entities = [{"name": "Bedroom Light", "aliases": [], "area": "Bedroom"}]
    fallback = resolve_area_context(
        "turn on the lights", satellite_area="Kitchen",
        areas=("Kitchen", "Bedroom"), entities=entities,
    )
    explicit = resolve_area_context(
        "turn on the lights in the Bedroom", satellite_area="Kitchen",
        areas=("Kitchen", "Bedroom"), entities=entities,
    )
    assert (fallback.target_area, fallback.target_area_source) == (
        "Kitchen", "satellite_fallback"
    )
    other_satellite = resolve_area_context(
        "turn on the lights", satellite_area="Bedroom",
        areas=("Kitchen", "Bedroom"), entities=entities,
    )
    assert other_satellite.target_area == "Bedroom"
    assert (explicit.target_area, explicit.target_area_source) == (
        "Bedroom", "explicit_area"
    )


def test_gold_uses_satellite_area_when_legacy_area_differs() -> None:
    home = {
        "satellite_area": "Kitchen",
        "sayso_entity_area": "Bedroom",
        "entities": [
            {"name": "Kitchen Light", "capability": "lights", "area": "Kitchen",
             "features": ("on",), "domain": "light"},
        ],
    }
    expected = gold_from_scenario(
        {"home": home, "capability": "lights", "operation": "turn_on",
         "targeting": "context", "robustness": "ordinary"},
        random.Random(1),
    )
    assert expected["calls"][0]["arguments"]["area"] == "Kitchen"


def test_named_target_survives_area_words_and_ambiguity_fails_closed() -> None:
    named = resolve_area_context(
        "turn on the Kitchen Light", satellite_area="Bedroom",
        areas=("Kitchen", "Bedroom"),
        entities=[{"name": "Kitchen Light", "aliases": [], "area": "Kitchen"}],
    )
    ambiguous = resolve_area_context(
        "turn on the TV", satellite_area="Kitchen", areas=("Kitchen",),
        entities=[
            {"name": "TV", "aliases": [], "area": "Bedroom"},
            {"name": "Kitchen TV", "aliases": ["TV"], "area": "Kitchen"},
        ],
    )
    assert named.target_area_source == "named_target"
    assert ambiguous.target_area_source == "ambiguous"
    assert ambiguous.target_area is None


def test_area_families_render_the_contract_and_expected_targeting() -> None:
    rows = {variant["family"]: render_example(build_spec(variant, seed=3))
            for variant in area_context_variants()}
    assert set(rows) == {
        "area_generic_satellite_fallback", "area_explicit_cross_room_bedroom",
        "area_named_other_room_tv", "area_cross_room_distractors_tv",
        "area_missing_satellite_clarify", "area_ambiguous_target_clarify",
    }
    assert rows["area_generic_satellite_fallback"]["metadata"].get(
        "target_area_source"
    ) == (
        "satellite_fallback"
    )
    assert rows["area_explicit_cross_room_bedroom"]["metadata"].get(
        "target_area_source"
    ) == (
        "explicit_area"
    )
    assert rows["area_missing_satellite_clarify"]["metadata"]["target_area_source"] == (
        "missing_area"
    )
    assert "satellite_area=Kitchen" in rows["area_named_other_room_tv"]["messages"][0][
        "content"
    ]
    context = resolve_area_context(
        "turn on the lights", satellite_area="Kitchen",
        areas=("Kitchen",), entities=()
    )
    assert render_area_context(context) == (
        "satellite_area=Kitchen\ntarget_area=Kitchen\n"
        "target_area_source=satellite_fallback"
    )


def test_production_generation_delivers_area_families_and_zero_rate_is_valid() -> None:
    result = run_generation(
        GeneratorConfig(
            count=1000, seed=20260916, grounding_rate=0,
            discrimination_rate=0, stt_noise_rate=0,
        )
    )
    assert set(result["stats"]["area_context"]["by_family"]) == {
        variant["family"] for variant in area_context_variants()
    }
    zero_rate = run_generation(
        GeneratorConfig(
            count=1000, seed=7, area_context_rate=0, grounding_rate=0,
            discrimination_rate=0, stt_noise_rate=0,
        )
    )
    assert zero_rate["stats"]["area_context"]["rows"] == 0
