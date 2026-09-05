"""Provenance metadata helpers."""

from __future__ import annotations

from typing import Any


def provenance_for_scenario(scenario: dict[str, Any], *, paraphrase: bool = False) -> dict[str, Any]:
    return {
        "generator": "sayso_synthetic_v3",
        "semantic_id": scenario.get("semantic_id"),
        "seed": scenario.get("seed"),
        "capability": scenario.get("capability"),
        "operation": scenario.get("operation"),
        "paraphrased": paraphrase,
        "split": scenario.get("split"),
    }
