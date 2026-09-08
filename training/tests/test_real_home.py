"""Splitting and mixing in a fetched real Home Assistant home."""

from __future__ import annotations

import json

import pytest

from generators.config import GeneratorConfig
from generators.real_home import (
    HOLDOUT_STRIDE,
    derive_entity_cap,
    holdout_entity_names,
    load_real_home,
    split_entities,
)
from generators.scenarios import build_scenario

FIXTURE = "training/fixtures/real_home.json"


def _entity(entity_id: str, capability: str, name: str) -> dict:
    return {
        "entity_id": entity_id,
        "name": name,
        "aliases": [name],
        "domain": entity_id.split(".")[0],
        "kind": entity_id.split(".")[0],
        "capability": capability,
        "device_class": None,
        "area": "Kitchen",
        "floor": "Main Floor",
        "state": "on",
        "capabilities": ["on", "off"],
    }


ENTITIES = [_entity(f"light.l{i}", "lights", f"Light {i}") for i in range(10)] + [
    _entity("climate.only", "climate", "Only Thermostat")
]


def test_splits_are_disjoint_and_cover_everything():
    train = split_entities(ENTITIES, "train")
    holdout = split_entities(ENTITIES, "holdout")
    train_ids = {e["entity_id"] for e in train}
    holdout_ids = {e["entity_id"] for e in holdout}
    assert not train_ids & holdout_ids
    assert train_ids | holdout_ids == {e["entity_id"] for e in ENTITIES}
    assert len(holdout) == len(ENTITIES) // HOLDOUT_STRIDE


def test_a_lone_entity_of_its_capability_stays_in_training():
    assert "climate.only" in {e["entity_id"] for e in split_entities(ENTITIES, "train")}
    assert "climate.only" not in {e["entity_id"] for e in split_entities(ENTITIES, "holdout")}


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="split must be one of"):
        split_entities(ENTITIES, "validation")


def test_loading_returns_an_independent_copy():
    first = load_real_home(FIXTURE)
    first["entities"].append(_entity("light.injected", "lights", "Injected"))
    second = load_real_home(FIXTURE)
    assert len(second["entities"]) == len(first["entities"]) - 1


def test_the_real_fixture_splits_without_losing_capabilities():
    train = load_real_home(FIXTURE, split="train")
    holdout = load_real_home(FIXTURE, split="holdout")
    assert holdout["entities"]
    assert train["home_id"].endswith("_train")
    assert train["size"] == len(train["entities"])
    assert train["sayso_entity_area"] in {e["area"] for e in train["entities"]}
    fixture = json.load(open(FIXTURE, encoding="utf-8"))
    assert {e["capability"] for e in fixture["entities"]} == {
        e["capability"] for e in train["entities"]
    }


def test_holdout_names_never_appear_in_the_training_split():
    train_names = {e["name"] for e in load_real_home(FIXTURE, split="train")["entities"]}
    assert not train_names & holdout_entity_names(FIXTURE)


def test_build_scenario_uses_the_home_it_is_given():
    home = load_real_home(FIXTURE, split="train")
    scenario = build_scenario(
        index=0,
        seed=1,
        capability="lights",
        operation="turn_on",
        home_size=16,
        home=home,
    )
    assert scenario["home"]["home_id"] == home["home_id"]
    assert scenario["target_entity"]["entity_id"] in {e["entity_id"] for e in home["entities"]}


def test_build_scenario_injects_capabilities_the_real_home_lacks():
    home = load_real_home(FIXTURE, split="train")
    assert not any(e["capability"] == "locks" for e in home["entities"])
    scenario = build_scenario(
        index=3,
        seed=1,
        capability="locks",
        operation="lock",
        home_size=16,
        home=home,
    )
    assert scenario["target_entity"]["capability"] == "locks"


def test_build_scenario_without_a_home_is_unchanged():
    kwargs = dict(index=5, seed=2, capability="lights", operation="turn_on", home_size=16)
    assert build_scenario(**kwargs) == build_scenario(**kwargs, home=None)


def test_real_home_settings_reach_the_manifest():
    config = GeneratorConfig(real_home_path=FIXTURE, real_home_rate=0.05)
    recorded = config.to_dict()
    assert recorded["real_home_path"] == FIXTURE
    assert recorded["real_home_rate"] == 0.05


def test_default_config_does_not_mix_in_a_real_home():
    config = GeneratorConfig()
    assert config.real_home_path is None
    assert config.real_home_rate == 0.0


def test_mixing_is_off_by_default_in_generated_rows():
    from generators.pipeline import run_generation

    rows = run_generation(GeneratorConfig(count=40, seed=3))["rows"]
    markers = [
        e["name"]
        for e in load_real_home(FIXTURE, split="train")["entities"]
        if len(e["name"]) > 14 and " " in e["name"]
    ]
    blob = json.dumps(rows)
    assert not any(marker in blob for marker in markers)


def test_derived_cap_scales_with_count_and_rate():
    assert derive_entity_cap(40_000, 0.10, 56) == 288
    assert derive_entity_cap(40_000, 0.05, 56) == 144
    assert derive_entity_cap(40_000, 0.0, 56) == 0
    assert derive_entity_cap(40_000, 0.10, 0) == 0


def test_an_explicit_cap_is_never_exceeded():
    from generators.pipeline import run_generation

    config = GeneratorConfig(
        count=600,
        seed=41,
        real_home_path=FIXTURE,
        real_home_rate=0.2,
        real_home_entity_cap=3,
    )
    report = run_generation(config)["stats"]
    counts = dict(report["real_home"]["most_common"])
    assert counts, "expected the real home to contribute rows"
    assert max(counts.values()) <= 3
    assert report["accepted"] == 600


def test_capped_rows_are_replaced_not_dropped():
    from generators.pipeline import run_generation

    report = run_generation(
        GeneratorConfig(
            count=600,
            seed=41,
            real_home_path=FIXTURE,
            real_home_rate=0.2,
            real_home_entity_cap=2,
        )
    )["stats"]
    # The cap rejects rows; the quota still completes because a retry re-rolls
    # the real/synthetic draw.
    assert report["rejection_reasons"].get("real_home_entity_cap", 0) > 0
    assert report["accepted"] == 600


def test_no_real_home_means_no_cap_bookkeeping():
    from generators.pipeline import run_generation

    report = run_generation(GeneratorConfig(count=40, seed=3))["stats"]
    assert "real_home" not in report
