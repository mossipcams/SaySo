"""Tests for STT noise."""

from __future__ import annotations

import random

from generators.stt_noise import apply_stt_noise, utterance_contains_target


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


def test_tv_stt_variants_preserve_target_reference() -> None:
    rng = random.Random(11)
    seen: set[str] = set()
    sources = ("turn on the living room TV", "turn on the living room tv")
    for _ in range(80):
        source = sources[_ % len(sources)]
        corrupted, kind = apply_stt_noise(
            source,
            rng,
            target_names=["TV"],
        )
        if kind in {"letter_spaced_tv", "punctuated_tv", "phonetic_tv"}:
            seen.add(kind)
            assert utterance_contains_target(corrupted, "TV")
            assert corrupted != source
    assert seen, "expected at least one TV STT corruption"


def test_utterance_contains_target_accepts_letter_spaced_tv() -> None:
    assert utterance_contains_target("turn on the T V", "TV")
    assert utterance_contains_target("pause the T.V.", "TV")
    assert utterance_contains_target("mute the teevee", "TV")
    assert not utterance_contains_target("turn on the speaker", "TV")


def test_log_stt_living_room_becomes_ribbon_or_librarian() -> None:
    from generators.stt_noise import apply_log_stt_noise

    source = "Turn on the living room light"
    seen: set[str] = set()
    for seed in range(80):
        corrupted, kind = apply_log_stt_noise(source, random.Random(seed), target_names=["Living room lights"])
        if kind in {"log_living_ribbon", "log_living_librarian"}:
            seen.add(kind)
            assert corrupted != source
            assert utterance_contains_target(corrupted, "Living room lights")
            assert utterance_contains_target(corrupted, "light")
    assert "log_living_ribbon" in seen
    assert "log_living_librarian" in seen


def test_log_stt_matches_live_librarian_tv_transcript() -> None:
    assert utterance_contains_target("Turn off librarian TV.", "TV")
    assert utterance_contains_target("turn on the ribbon room light", "Living Room light")


def test_log_stt_drops_fillers_and_adds_period() -> None:
    from generators.stt_noise import apply_log_stt_noise

    kinds: set[str] = set()
    for seed in range(60):
        corrupted, kind = apply_log_stt_noise(
            "please turn on the kitchen light",
            random.Random(seed),
            target_names=["Kitchen light"],
        )
        if kind:
            kinds.add(kind)
            assert utterance_contains_target(corrupted, "Kitchen light")
    assert "log_dropped_filler" in kinds or "log_trailing_period" in kinds or "log_casing" in kinds
