"""Tests for paraphrase module (disabled by default)."""

from __future__ import annotations

from generators.paraphrase import load_paraphraser


def test_paraphraser_disabled_by_default() -> None:
    assert load_paraphraser(False) is None
