"""Tests for home generation."""

from __future__ import annotations

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
