"""Colocated tests for wake corpus candidate events."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from satellite.sayso.wake.corpus import (
    TRAIN_LABELS,
    import_spool_record,
    load_events,
    set_event_label,
)
from satellite.sayso.wake.eval import write_synthetic_wav
from satellite.sayso.wake.livekit import SAMPLE_RATE, WINDOW_SAMPLES
from satellite.sayso.wake.mining import HardNegativeMiner
from satellite.sayso.wake.sessions import ingest_session


def _write_record(spool: Path, capture_id: str, *, session_id: str, score: float = 0.4) -> Path:
    record_dir = spool / "records" / capture_id
    record_dir.mkdir(parents=True)
    samples = np.full(WINDOW_SAMPLES, 1000, dtype="<i2")
    write_synthetic_wav(record_dir / "window.wav", samples)
    meta = {
        "capture_id": capture_id,
        "session_id": session_id,
        "score": score,
        "verifier_score": 0.12,
        "sampling_reason": "near_threshold",
        "sample_start": 0,
        "sample_end": WINDOW_SAMPLES,
        "label": None,
        "hashes": {},
    }
    (record_dir / "record.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return record_dir


def test_import_spool_record_adds_context(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    wav = tmp_path / "session.wav"
    samples = np.zeros(SAMPLE_RATE * 3, dtype="<i2")
    samples[SAMPLE_RATE : SAMPLE_RATE + WINDOW_SAMPLES] = 5000
    write_synthetic_wav(wav, samples)
    session = ingest_session(wav, corpus, session_id="ctx_session")
    record_dir = _write_record(tmp_path / "spool", "evt-1", session_id=session.session_id)
    event = import_spool_record(record_dir, corpus, session=session, verifier_score=0.33)
    assert event.session_id == session.session_id
    assert event.verifier_score == pytest.approx(0.33)
    assert (event.record_dir / "context.wav").is_file()
    assert event.label is None


def test_human_label_is_authoritative(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    record_dir = _write_record(tmp_path / "spool", "evt-2", session_id="s1")
    event = import_spool_record(record_dir, corpus)
    labeled = set_event_label(corpus, event.event_id, "negative")
    assert labeled.label == "negative"
    reloaded = load_events(corpus)[0]
    assert reloaded.label == "negative"
    meta = json.loads((reloaded.record_dir / "record.json").read_text())
    assert meta.get("labeled_utc")


def test_mined_records_start_unlabeled(tmp_path: Path) -> None:
    spool = tmp_path / "spool"
    window = np.full(WINDOW_SAMPLES, 1000, dtype="<i2")
    miner = HardNegativeMiner(
        spool,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    miner.offer(0.55, window, sample_index=31999)
    miner._flush_clusters(force=True)
    miner.stop(timeout=3.0)
    record_dir = next((spool / "records").iterdir())
    meta = json.loads((record_dir / "record.json").read_text())
    assert meta["label"] is None


def test_train_labels_exclude_unsure(tmp_path: Path) -> None:
    assert "unsure" not in TRAIN_LABELS


def test_verifier_score_populated_when_provider_has_verifier(tmp_path: Path) -> None:
    from types import SimpleNamespace

    corpus = tmp_path / "corpus"
    record_dir = _write_record(tmp_path / "spool", "evt-ver", session_id="s1")
    provider = SimpleNamespace(
        _verifier=SimpleNamespace(score=lambda window, model: 0.77),
        _model=object(),
    )
    event = import_spool_record(record_dir, corpus, provider=provider)
    assert event.verifier_score == pytest.approx(0.77)
    meta = json.loads((event.record_dir / "record.json").read_text())
    assert meta["verifier_score"] == pytest.approx(0.77)


def test_verifier_score_left_null_without_provider(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    record_dir = tmp_path / "spool" / "records" / "evt-no-ver"
    record_dir.mkdir(parents=True)
    write_synthetic_wav(record_dir / "window.wav", np.full(WINDOW_SAMPLES, 1000, dtype="<i2"))
    (record_dir / "record.json").write_text(
        json.dumps(
            {
                "capture_id": "evt-no-ver",
                "session_id": "s1",
                "score": 0.4,
                "sampling_reason": "near_threshold",
                "sample_start": 0,
                "sample_end": WINDOW_SAMPLES,
                "label": None,
                "hashes": {},
            }
        ),
        encoding="utf-8",
    )
    event = import_spool_record(record_dir, corpus)
    assert event.verifier_score is None
