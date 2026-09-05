"""Tests for duplicate detection."""

from __future__ import annotations

from generators.duplicates import DuplicateTracker, utterance_hash


def test_duplicate_tracker_rejects_repeated_utterance() -> None:
    tracker = DuplicateTracker(near_limit=2)
    spec = {"utterance": "turn on the light", "home": {"entities": []}, "semantic_id": "a"}
    assert tracker.would_reject(spec) is None
    tracker.record(spec)
    tracker.record(spec)
    assert tracker.would_reject(spec) == "exact_duplicate_utterance"


def test_utterance_hash_normalizes_case() -> None:
    assert utterance_hash("Turn On") == utterance_hash("turn on")
