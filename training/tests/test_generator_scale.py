"""Scale checks for pile-based generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.home_llm_piles import SMALL_FACTORS, estimate_example_count, generate_pile_examples


def test_small_run_exceeds_legacy_hardcoded_scale() -> None:
    examples = list(generate_pile_examples(seed=42, factors=SMALL_FACTORS))
    assert len(examples) >= 3000
    assert len(examples) > 200


def test_estimate_matches_emitted_order_of_magnitude() -> None:
    estimate = estimate_example_count(factors=SMALL_FACTORS)
    emitted = len(list(generate_pile_examples(seed=1, factors=SMALL_FACTORS)))
    assert emitted >= estimate * 0.8
