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


def test_entity_names_are_too_varied_to_memorize() -> None:
    """Run 008 memorized entity names: 2,124 of them filled 1.25M name slots, a
    median of 693 repeats each, so copying a name from context was never the
    cheapest rule to learn. Keep names near-unique across the corpus."""
    names: list[str] = []
    rng = random.Random(20260906)
    for index, size in enumerate([16, 32, 64] * 100):
        names.extend(entity["name"] for entity in generate_home(index, size, rng)["entities"])
    distinct = len(set(names))
    assert distinct / len(names) > 0.85, f"only {distinct} distinct names across {len(names)} slots"

    repeats = sorted(collections.Counter(names).values())
    assert repeats[len(repeats) // 2] == 1, "the median name repeats; it will be memorized"


def test_training_areas_never_collide_with_eval_areas() -> None:
    """The v3 suites hold their areas out so eval entity names are unseen in
    training. Share an area and the generator can emit an eval target verbatim —
    `Study Desk Fan` and `Dining Room Pendant Light` are gold targets."""
    from evals.v3_quality import _GOLD_AREAS, _SHADOW_AREAS
    from generators.homes import _AREAS

    shared = set(_AREAS) & (set(_GOLD_AREAS) | set(_SHADOW_AREAS))
    assert not shared, f"training and eval share areas: {sorted(shared)}"


def test_generator_cannot_emit_an_eval_entity_name() -> None:
    """The property the area split exists to protect, checked directly."""
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
