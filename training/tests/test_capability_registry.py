"""Tests for capability registry."""

from __future__ import annotations

from generators.capability_registry import (
    CAPABILITIES,
    TIER1_CAPABILITY_WEIGHTS,
    TIER_PROPORTIONS,
    SupportLevel,
    registry_summary,
    trainable_operations,
)


def test_tier_proportions_sum_to_one() -> None:
    assert abs(sum(TIER_PROPORTIONS.values()) - 1.0) < 0.01


def test_tier1_weights_present() -> None:
    assert set(TIER1_CAPABILITY_WEIGHTS) == {
        "lights",
        "media_players",
        "timers",
        "climate",
        "switches",
        "fans",
        "covers",
        "locks",
    }


def test_climate_marked_unavailable() -> None:
    cap = CAPABILITIES["climate"]
    assert cap.support == SupportLevel.UNAVAILABLE
    assert cap.blocker is not None


def test_media_partial_support() -> None:
    cap = CAPABILITIES["media_players"]
    ops = {op.name: op for op in cap.operations}
    assert ops["turn_on"].support == SupportLevel.PARTIAL
    assert ops["volume_set"].support == SupportLevel.UNAVAILABLE


def test_registry_summary_covers_all_capabilities() -> None:
    summary = registry_summary()
    assert set(summary) == set(CAPABILITIES)


def test_lights_have_trainable_operations() -> None:
    ops = trainable_operations(CAPABILITIES["lights"])
    names = {op.name for op in ops}
    assert {"turn_on", "turn_off", "set_brightness"}.issubset(names)
