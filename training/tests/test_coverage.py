"""Coverage distribution tests."""

from __future__ import annotations

from collections import Counter

from generators.config import GeneratorConfig
from generators.pipeline import run_generation


def test_accepted_rows_span_capabilities() -> None:
    result = run_generation(GeneratorConfig(count=200, seed=77))
    caps = Counter(r["metadata"]["capability"] for r in result["rows"])
    assert len(caps) >= 5
