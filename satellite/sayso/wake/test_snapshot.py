"""Colocated tests for wake corpus session-level snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from satellite.sayso.wake.corpus import import_spool_record, set_event_label
from satellite.sayso.wake.eval import write_synthetic_wav
from satellite.sayso.wake.livekit import WINDOW_SAMPLES
from satellite.sayso.wake.corpus import load_events
from satellite.sayso.wake.snapshot import (
    CorpusExample,
    assign_session_splits,
    assert_no_split_leak,
    derive_eval_examples,
    derive_snapshot_examples,
    derive_training_examples,
    ensure_session_splits,
    holdout_sessions,
    load_session_splits,
)


def _write_event(corpus: Path, event_id: str, session_id: str, *, label: str | None = None) -> None:
    spool = corpus / "spool"
    record_dir = spool / "records" / event_id
    record_dir.mkdir(parents=True)
    write_synthetic_wav(record_dir / "window.wav", np.full(WINDOW_SAMPLES, 1000, dtype="<i2"))
    meta = {
        "capture_id": event_id,
        "session_id": session_id,
        "score": 0.4,
        "sampling_reason": "near_threshold",
        "sample_start": 0,
        "sample_end": WINDOW_SAMPLES,
        "label": label,
        "hashes": {},
    }
    (record_dir / "record.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    import_spool_record(record_dir, corpus)
    if label:
        set_event_label(corpus, event_id, label)


def test_session_split_is_deterministic() -> None:
    first = assign_session_splits(["s1", "s2", "s3"], seed=7, holdout_session_ids=["s3"])
    second = assign_session_splits(["s1", "s2", "s3"], seed=7, holdout_session_ids=["s3"])
    assert first == second
    assert first["s3"] == "holdout"


def test_overlapping_windows_stay_in_one_session_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    session_id = "overlap_session"
    _write_event(corpus, "evt-a", session_id, label="positive")
    _write_event(corpus, "evt-b", session_id, label="negative")
    (corpus / "corpus_splits.json").write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 11,
                "holdout_session_ids": [],
                "assignments": {session_id: "train"},
            }
        ),
        encoding="utf-8",
    )
    splits = load_session_splits(corpus)
    examples = derive_training_examples(load_events(corpus), splits)
    assert {example.session_id for example in examples} == {session_id}


def test_holdout_sessions_excluded_from_training(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_event(corpus, "evt-h1", "holdout_a", label="positive")
    _write_event(corpus, "evt-h2", "holdout_a", label="negative")
    _write_event(corpus, "evt-t1", "train_a", label="positive")
    _write_event(corpus, "evt-t2", "train_a", label="negative")
    splits = ensure_session_splits(corpus, seed=3, holdout_session_ids=["holdout_a"])
    examples = derive_training_examples(load_events(corpus), splits)
    assert holdout_sessions(splits) == ["holdout_a"]
    assert all(example.session_id != "holdout_a" for example in examples)


def test_no_session_split_leak() -> None:
    train = [CorpusExample("a", "session_a", Path("a.wav"), "positive", "train")]
    eval_examples = [CorpusExample("b", "session_b", Path("b.wav"), "negative", "eval")]
    assert_no_split_leak(train, eval_examples)
    with pytest.raises(RuntimeError, match="split leak"):
        assert_no_split_leak(train, [CorpusExample("c", "session_a", Path("c.wav"), "negative", "eval")])


def test_unassigned_sessions_excluded_from_training(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_event(corpus, "evt-known", "train_a", label="positive")
    _write_event(corpus, "evt-unknown", "", label="negative")
    _write_event(corpus, "evt-missing", "missing_session", label="positive")
    splits = {"train_a": "train"}
    examples = derive_training_examples(load_events(corpus), splits)
    source_ids = {example.source_id for example in examples}
    assert source_ids == {"evt-known"}


def test_existing_splits_are_preserved(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_splits.json").write_text(
        json.dumps({"version": 1, "seed": 1, "holdout_session_ids": [], "assignments": {"s1": "holdout"}}),
        encoding="utf-8",
    )
    assert load_session_splits(corpus) == {"s1": "holdout"}
    assert ensure_session_splits(corpus, seed=99) == {"s1": "holdout"}


def test_ensure_session_splits_honors_new_holdout_on_existing_file(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_splits.json").write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 1,
                "holdout_session_ids": [],
                "assignments": {"s1": "train", "s2": "eval"},
            }
        ),
        encoding="utf-8",
    )
    splits = ensure_session_splits(corpus, seed=1, holdout_session_ids=["s2"])
    assert splits == {"s1": "train", "s2": "holdout"}
    payload = json.loads((corpus / "corpus_splits.json").read_text(encoding="utf-8"))
    assert payload["holdout_session_ids"] == ["s2"]


def test_eval_session_examples_appear_in_snapshot_not_train(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    _write_event(corpus, "evt-e1", "eval_session", label="positive")
    _write_event(corpus, "evt-e2", "eval_session", label="negative")
    _write_event(corpus, "evt-t1", "train_session", label="positive")
    splits = {"eval_session": "eval", "train_session": "train"}
    events = load_events(corpus)
    train_examples = derive_training_examples(events, splits)
    eval_examples = derive_eval_examples(events, splits)
    snapshot_examples = derive_snapshot_examples(events, splits)
    assert {example.source_id for example in train_examples} == {"evt-t1"}
    assert {example.source_id for example in eval_examples} == {"evt-e1", "evt-e2"}
    assert all(example.split == "eval" for example in eval_examples)
    assert {example.source_id for example in snapshot_examples} == {"evt-t1", "evt-e1", "evt-e2"}
