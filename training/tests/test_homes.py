"""Tests for home generation."""

from __future__ import annotations

import collections
import random

from generators.homes import generate_home


def test_home_sizes_respected() -> None:
    rng = random.Random(42)
    for size in (8, 16, 32):
        home = generate_home(1, size, rng)
        assert len(home["entities"]) >= size


def test_home_has_sayso_area() -> None:
    home = generate_home(5, 16, random.Random(7))
    assert home["sayso_entity_area"]
    assert all(e["area"] for e in home["entities"])


def test_entity_ids_match_domains() -> None:
    home = generate_home(3, 16, random.Random(11))
    for entity in home["entities"]:
        assert entity["entity_id"].startswith(entity["domain"] + ".")


def test_entity_ids_are_unique_within_a_home() -> None:
    """Two names that slug alike would collide as per-script tool names."""
    for size in (16, 32, 64):
        home = generate_home(9, size, random.Random(size))
        ids = [entity["entity_id"] for entity in home["entities"]]
        assert len(ids) == len(set(ids))


def test_entity_names_balance_realistic_reuse_and_variety() -> None:
    """Ordinary households share names; variety must not require invented names."""
    names: list[str] = []
    rng = random.Random(20260906)
    for index, size in enumerate([16, 32, 64] * 100):
        names.extend(entity["name"].casefold() for entity in generate_home(index, size, rng)["entities"])
    counts = collections.Counter(names)
    assert len(counts) >= 1000, "fixture/room variety collapsed"
    assert max(counts.values()) / len(names) < 0.03, "one name dominates the corpus"
    familiar = [name for name, count in counts.items() if count >= 3
                and name.startswith(("kitchen ", "living room ", "bathroom ", "master bedroom "))]
    assert len(familiar) >= 20, "ordinary room/device names should recur across homes"


def test_generator_cannot_emit_an_eval_entity_name() -> None:
    """Real homes share room names, so training and eval areas overlap on
    purpose. The property that has to hold is narrower: no generated entity name
    may equal one the suites test on, or that eval row stops being held out."""
    from evals.v3_quality import build_shadow_specs, gold_specs

    eval_names = {
        entity["name"]
        for spec in gold_specs() + build_shadow_specs(seed=20260906, count=100)
        for entity in spec["home"]["entities"]
    }
    generated: set[str] = set()
    rng = random.Random(11)
    for index, size in enumerate([32, 64] * 120):
        generated.update(entity["name"] for entity in generate_home(index, size, rng)["entities"])
    assert not generated & eval_names, sorted(generated & eval_names)


def test_apostrophe_names_stay_represented_but_not_dominant() -> None:
    """Apostrophes are a known failure class, so they must appear — but a third
    of a home named after someone is not a home."""
    names: list[str] = []
    rng = random.Random(4)
    for index, size in enumerate([16, 32] * 60):
        names.extend(entity["name"] for entity in generate_home(index, size, rng)["entities"])
    share = sum("'" in name for name in names) / len(names)
    assert 0.05 < share < 0.25, share
