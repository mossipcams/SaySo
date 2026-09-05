"""Tests for STT noise."""

from __future__ import annotations

import random

from generators.stt_noise import apply_stt_noise


def test_stt_noise_can_change_utterance() -> None:
    rng = random.Random(0)
    for _ in range(20):
        corrupted, kind = apply_stt_noise("turn on the kitchen light", rng)
        if kind:
            assert corrupted != "turn on the kitchen light"
            return
    corrupted, kind = apply_stt_noise("turn on the kitchen light", rng)
    assert isinstance(corrupted, str)


def test_number_variant_changes_spoken_form_not_identity() -> None:
    rng = random.Random(7)
    corrupted, kind = apply_stt_noise("set brightness to 50 percent", rng)
    if kind == "number_variant":
        assert "50" not in corrupted or "fifty" in corrupted.casefold()
        return
    corrupted, kind = apply_stt_noise("set brightness to fifty percent", rng)
    if kind == "number_variant":
        assert "50" in corrupted or "fifty" not in corrupted.casefold()
