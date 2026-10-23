from __future__ import annotations

import hashlib
import itertools
import json
import logging
import random
import shutil
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from satellite.sayso.wake.capture import WakeCaptureRing
from satellite.sayso.wake.mining import (
    SAMPLING_BELOW,
    SAMPLING_DETECTION,
    SAMPLING_NEAR,
    HardNegativeMiner,
    ingest_record,
    write_ack,
)


def _window(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(32000) * 1000).astype("<i2")


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")


def _publish(
    miner: HardNegativeMiner, score: float, window: np.ndarray, index: int
) -> str | None:
    capture_id = miner.offer(score, window, sample_index=index)
    miner._flush_clusters(force=True)
    miner.stop(timeout=3.0)
    miner.start()
    return capture_id


def test_mined_window_is_byte_exact(tmp_path: Path) -> None:
    window = _window(1)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    _publish(miner, 0.22, window, 31999)
    record = next((tmp_path / "records").iterdir())
    assert np.array_equal(_read_wav(record / "window.wav"), window)


def test_mined_raw_context_is_pre_gain_and_has_stream_metadata(
    tmp_path: Path, caplog
) -> None:
    raw_pre = np.arange(1600, dtype="<i2")
    window = _window(10)
    ring = WakeCaptureRing(16000 * 10)
    ring.append((raw_pre * 2).tobytes() + window.tobytes())
    raw_ring = WakeCaptureRing(16000 * 10)
    raw_ring.append(raw_pre.tobytes() + window.tobytes())
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        pre_context_ms=100,
        post_context_ms=0,
        below_sample_rate=0.0,
    )
    miner.bind_ring(ring)
    miner.bind_raw_ring(raw_ring)
    miner.start()
    try:
        with caplog.at_level(logging.WARNING, logger="satellite.sayso.wake.mining"):
            miner.snapshot_pre_trigger(33599)
            capture_id = miner.offer(0.8, window, sample_index=33599)
    finally:
        miner.stop(timeout=3.0)
        if miner._flush_timer is not None:
            miner._flush_timer.cancel()
            miner._flush_timer.join(timeout=3.0)
    assert capture_id
    assert not any("drifted" in rec.message for rec in caplog.records)
    record = tmp_path / "records" / capture_id
    raw_path = record / "raw_pre.wav"
    assert _read_wav(raw_path).tobytes() == raw_pre.tobytes()
    assert _read_wav(record / "pre.wav").tobytes() == (raw_pre * 2).tobytes()
    meta = json.loads((record / "record.json").read_text())
    assert meta["streams"]["raw"] == {
        "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "sample_rate": 16000,
        "channels": 1,
        "raw_missing": False,
    }
    assert ingest_record(record) == (True, "ok")


def test_mined_raw_context_missing_when_ring_has_no_coverage(tmp_path: Path) -> None:
    window = _window(11)
    ring = WakeCaptureRing(16000 * 10)
    ring.append(np.zeros(1600, dtype="<i2").tobytes() + window.tobytes())
    raw_ring = WakeCaptureRing(32000)
    raw_ring.append(np.zeros(33600, dtype="<i2").tobytes())
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        pre_context_ms=100,
        post_context_ms=0,
        below_sample_rate=0.0,
    )
    miner.bind_ring(ring)
    miner.bind_raw_ring(raw_ring)
    miner.start()
    try:
        miner.snapshot_pre_trigger(33599)
        capture_id = miner.offer(0.8, window, sample_index=33599)
    finally:
        miner.stop(timeout=3.0)
        if miner._flush_timer is not None:
            miner._flush_timer.cancel()
            miner._flush_timer.join(timeout=3.0)
    assert capture_id
    record = tmp_path / "records" / capture_id
    meta = json.loads((record / "record.json").read_text())
    assert meta["streams"]["raw"] == {
        "sample_rate": 16000,
        "channels": 1,
        "raw_missing": True,
    }
    assert not (record / "raw_pre.wav").exists()
    assert ingest_record(record) == (True, "ok")


def test_raw_ring_drift_warns_without_changing_outputs(tmp_path: Path, caplog) -> None:
    raw_pre = np.arange(1600, dtype="<i2")
    window = _window(14)
    ring = WakeCaptureRing(16000 * 10)
    ring.append((raw_pre * 2).tobytes())
    ring.append(window.tobytes())
    ring.append(np.zeros(2560, dtype="<i2").tobytes())
    raw_ring = WakeCaptureRing(16000 * 10)
    raw_ring.append(raw_pre.tobytes())
    raw_ring.append(window.tobytes())
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        pre_context_ms=100,
        post_context_ms=0,
        below_sample_rate=0.0,
    )
    miner.bind_ring(ring)
    miner.bind_raw_ring(raw_ring)
    miner.start()
    try:
        with caplog.at_level(logging.WARNING, logger="satellite.sayso.wake.mining"):
            miner.snapshot_pre_trigger(33599)
            capture_id = miner.offer(0.8, window, sample_index=33599)
        miner._flush_clusters(force=True)
    finally:
        miner.stop(timeout=3.0)
        if miner._flush_timer is not None:
            miner._flush_timer.cancel()
            miner._flush_timer.join(timeout=3.0)
    assert capture_id
    assert any("drifted" in rec.message for rec in caplog.records)
    record = tmp_path / "records" / capture_id
    assert np.array_equal(_read_wav(record / "window.wav"), window)
    assert _read_wav(record / "pre.wav").tobytes() == (raw_pre * 2).tobytes()
    assert _read_wav(record / "raw_pre.wav").tobytes() == raw_pre.tobytes()
    meta = json.loads((record / "record.json").read_text())
    assert meta["streams"]["raw"] == {
        "sha256": hashlib.sha256((record / "raw_pre.wav").read_bytes()).hexdigest(),
        "sample_rate": 16000,
        "channels": 1,
        "raw_missing": False,
    }
    assert ingest_record(record) == (True, "ok")


def test_mined_processing_state_includes_live_native_clip_count(tmp_path: Path) -> None:
    tally = SimpleNamespace(count=2)
    settings = {
        "capture_rate": 44100,
        "channels": 2,
        "gain_db": 6.0,
        "noise_suppression": 2,
        "auto_gain": 3,
    }
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        post_context_ms=0,
        below_sample_rate=0.0,
        processing_settings=settings,
        native_clip_count=lambda: tally.count,
    )
    capture_id = miner.offer(0.8, _window(12), sample_index=31999)
    tally.count = 7
    miner.start()
    try:
        miner._flush_clusters(force=True)
    finally:
        miner.stop(timeout=3.0)
        if miner._flush_timer is not None:
            miner._flush_timer.cancel()
            miner._flush_timer.join(timeout=3.0)
    assert capture_id
    record = tmp_path / "records" / capture_id
    meta = json.loads((record / "record.json").read_text())
    assert meta["processing_state"] == {**settings, "native_clips_at_publish": 7}


def test_legacy_record_without_streams_or_processing_state_ingests(
    tmp_path: Path,
) -> None:
    with wave.open(str(tmp_path / "window.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(_window(13).tobytes())
    (tmp_path / "record.json").write_text(
        json.dumps({"capture_id": "legacy"}), encoding="utf-8"
    )
    assert ingest_record(tmp_path) == (True, "ok")


def test_sampling_classes_and_metadata(tmp_path: Path) -> None:
    window = _window(2)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    near_id = _publish(miner, 0.22, window, 31999)
    det_id = _publish(miner, 0.8, window, 63999)
    assert near_id and det_id and near_id != det_id
    records = {
        p.name: json.loads((p / "record.json").read_text())
        for p in (tmp_path / "records").iterdir()
    }
    reasons = {meta["sampling_reason"] for meta in records.values()}
    assert SAMPLING_NEAR in reasons
    assert SAMPLING_DETECTION in reasons
    for meta in records.values():
        assert meta["label"] is None
        assert "hashes" in meta and meta["hashes"]["window_sha256"]
        assert meta["capture_id"] == meta["capture_id"]


def test_overlapping_windows_keep_triggering_window(tmp_path: Path) -> None:
    window_a = _window(3)
    window_b = _window(4)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    first_id = miner.offer(0.55, window_a, sample_index=31999)
    second_id = miner.offer(0.7, window_b, sample_index=32000)
    miner._flush_clusters(force=True)
    miner.stop(timeout=3.0)
    assert len(list((tmp_path / "records").iterdir())) == 1
    record = next((tmp_path / "records").iterdir())
    meta = json.loads((record / "record.json").read_text())
    assert meta["score"] == 0.7
    assert np.array_equal(_read_wav(record / "window.wav"), window_b)
    assert first_id == second_id
    assert meta["capture_id"] == first_id
    assert miner.latest_detection_capture_id == first_id


def test_queue_full_counts_loss_without_evicting_published(tmp_path: Path) -> None:
    window = _window(5)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        queue_size=1,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    blocker = threading.Event()

    def slow_run() -> None:
        while not miner._stop.is_set():
            try:
                job = miner._queue.get(timeout=0.05)
            except Exception:
                continue
            blocker.wait(timeout=0.2)
            miner._inflight += 1
            try:
                miner._publish_job(job)
            finally:
                miner._inflight -= 1

    miner.start()
    miner._thread = threading.Thread(target=slow_run, daemon=True)
    miner._thread.start()
    first = miner.offer(0.2, window, sample_index=31999)
    second = miner.offer(0.21, window, sample_index=63999)
    miner._flush_clusters(force=True)
    time.sleep(0.05)
    assert first is not None
    assert miner.losses.queue_full >= 1 or second is None
    blocker.set()
    miner.stop(timeout=3.0)
    assert len(list((tmp_path / "records").iterdir())) >= 1


def _ordered_stamps(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = itertools.count()
    monkeypatch.setattr(
        "satellite.sayso.wake.mining._utc_stamp",
        lambda: f"2026-01-01T00:00:{next(ticks):02d}Z",
    )


def _record_ids(root: Path) -> set[str]:
    return {p.name for p in (root / "records").iterdir()}


def test_spool_full_evicts_oldest_and_resumes_after_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ordered_stamps(monkeypatch)
    window = _window(6)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        max_records=1,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    first = _publish(miner, 0.2, window, 31999)
    second = _publish(miner, 0.25, window, 63999)
    assert _record_ids(tmp_path) == {second}
    assert miner.losses.evicted == 1
    assert miner._spool_full
    write_ack(tmp_path, second)
    miner.drain_acks()
    miner._reconcile_spool_state()
    assert not miner._spool_full
    third = _publish(miner, 0.26, window, 95999)
    assert _record_ids(tmp_path) == {third}
    assert not (tmp_path / "records" / first).exists()


def test_class_cap_evicts_oldest_of_class_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ordered_stamps(monkeypatch)
    window = _window(15)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        near_cap=2,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    detection = _publish(miner, 0.8, window, 31999)
    near_old = _publish(miner, 0.22, window, 63999)
    miner.publish_wake_outcome(near_old, accepted=False, reason="test")
    miner.publish_stt_outcome(near_old, stt_run_id="cmd-old", transcript="lamp")
    near_mid = _publish(miner, 0.23, window, 95999)
    near_new = _publish(miner, 0.24, window, 127999)

    assert _record_ids(tmp_path) == {detection, near_mid, near_new}
    assert not (tmp_path / "outcomes" / f"{near_old}.wake.json").exists()
    assert not (tmp_path / "outcomes" / f"{near_old}.stt.json").exists()
    assert miner._class_counts == {SAMPLING_DETECTION: 1, SAMPLING_NEAR: 2, SAMPLING_BELOW: 0}
    assert miner.published_record_count == 3
    assert miner.losses.evicted == 1
    assert miner.losses.spool_full == 0


def test_record_and_byte_limits_evict_oldest_overall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ordered_stamps(monkeypatch)
    window = _window(16)
    probe = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    probe.start()
    first = _publish(probe, 0.2, window, 31999)
    record_size = sum(p.stat().st_size for p in (tmp_path / "records" / first).iterdir())

    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        max_bytes=int(record_size * 2.5),
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    second = _publish(miner, 0.25, window, 63999)
    third = _publish(miner, 0.26, window, 95999)
    assert _record_ids(tmp_path) == {second, third}
    assert miner._published_bytes == sum(
        p.stat().st_size for d in (tmp_path / "records").iterdir() for p in d.iterdir()
    )
    assert miner.losses.evicted == 1


def test_record_that_cannot_fit_is_dropped_without_evicting(tmp_path: Path) -> None:
    window = _window(17)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        max_bytes=1,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    _publish(miner, 0.2, window, 31999)
    assert _record_ids(tmp_path) == set()
    assert miner.losses.spool_full == 1


def test_drain_and_reconcile_after_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ordered_stamps(monkeypatch)
    window = _window(18)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        max_records=2,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    evicted = _publish(miner, 0.2, window, 31999)
    kept = _publish(miner, 0.25, window, 63999)
    newest = _publish(miner, 0.26, window, 95999)
    assert _record_ids(tmp_path) == {kept, newest}
    miner.stop(timeout=3.0)
    write_ack(tmp_path, evicted)
    write_ack(tmp_path, newest)
    assert miner.drain_acks() == 1
    assert _record_ids(tmp_path) == {kept}
    miner._reconcile_spool_state()
    assert miner.published_record_count == 1
    assert miner._class_counts[SAMPLING_NEAR] == 1
    assert not miner._spool_full
    miner.start()
    resumed = _publish(miner, 0.27, window, 127999)
    assert _record_ids(tmp_path) == {kept, resumed}
    assert miner.losses.evicted == 1


def test_ingest_verifies_hashes_and_partial_writes_ignored(tmp_path: Path) -> None:
    window = _window(7)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    _publish(miner, 0.2, window, 31999)
    record = next((tmp_path / "records").iterdir())
    ok, _ = ingest_record(record)
    assert ok
    bad = tmp_path / ".staging" / "partial"
    bad.mkdir(parents=True)
    (bad / "record.json").write_text("{}", encoding="utf-8")
    miner._reconcile_spool_state()
    assert not (tmp_path / "records" / "partial").exists()


def test_outcomes_are_separate_and_idempotent(tmp_path: Path) -> None:
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    capture_id = "00000000-0000-4000-8000-000000000001"
    miner.publish_wake_outcome(
        capture_id, accepted=True, suppressed=False, reason="test"
    )
    miner.publish_wake_outcome(
        capture_id, accepted=True, suppressed=False, reason="test"
    )
    miner.publish_stt_outcome(capture_id, stt_run_id="cmd-1", transcript="lights on")
    wake = json.loads((tmp_path / "outcomes" / f"{capture_id}.wake.json").read_text())
    stt = json.loads((tmp_path / "outcomes" / f"{capture_id}.stt.json").read_text())
    assert wake["accepted"] is True
    assert stt["transcript"] == "lights on"


def test_ring_context_flags_and_late_post(tmp_path: Path) -> None:
    ring = WakeCaptureRing(16000 * 10)
    pcm = np.arange(32000, dtype="<i2").tobytes()
    ring.append(pcm)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        pre_context_ms=100,
        post_context_ms=100,
        post_deadline_ms=50,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.bind_ring(ring)
    miner.start()
    miner.snapshot_pre_trigger(31999)
    capture_id = _publish(miner, 0.2, np.frombuffer(pcm, dtype="<i2"), 31999)
    assert capture_id
    record = tmp_path / "records" / capture_id
    meta = json.loads((record / "record.json").read_text())
    assert meta["quality_flags"]["post_context_missing"] is True


def test_drain_acks_while_running_without_restart(tmp_path: Path) -> None:
    window = _window(9)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.start()
    capture_id = miner.offer(0.55, window, sample_index=31999)
    miner._flush_clusters(force=True)
    time.sleep(0.2)
    record_dir = tmp_path / "records" / capture_id
    assert record_dir.is_dir()
    write_ack(tmp_path, capture_id)
    deadline = time.monotonic() + 2.0
    while record_dir.is_dir() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not record_dir.is_dir()
    resumed_id = miner.offer(0.56, window, sample_index=63999)
    miner._flush_clusters(force=True)
    miner.stop(timeout=3.0)
    assert resumed_id is not None
    assert len(list((tmp_path / "records").iterdir())) >= 1


def test_near_miss_snapshots_ring_pre_context(tmp_path: Path) -> None:
    ring = WakeCaptureRing(16000 * 10)
    pre = np.arange(1600, dtype="<i2")
    window_pcm = np.arange(32000, dtype="<i2")
    ring.append(pre.tobytes())
    ring.append(window_pcm.tobytes())
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        pre_context_ms=100,
        below_sample_rate=0.0,
        rng=random.Random(0),
    )
    miner.bind_ring(ring)
    miner.start()
    capture_id = miner.offer(0.22, window_pcm, sample_index=33599)
    miner._flush_clusters(force=True)
    miner.stop(timeout=3.0)
    assert capture_id
    meta = json.loads((tmp_path / "records" / capture_id / "record.json").read_text())
    assert meta["quality_flags"]["pre_context_missing"] is False
    assert (tmp_path / "records" / capture_id / "pre.wav").is_file()


def test_below_threshold_independent_sample(tmp_path: Path) -> None:
    window = _window(8)
    miner = HardNegativeMiner(
        tmp_path,
        mine_threshold=0.1,
        detect_threshold=0.5,
        below_sample_rate=1.0,
        below_cap=2,
        rng=random.Random(0),
    )
    miner.start()
    capture_id = _publish(miner, 0.02, window, 31999)
    assert capture_id
    meta = json.loads(
        (next((tmp_path / "records").iterdir()) / "record.json").read_text()
    )
    assert meta["sampling_reason"] == SAMPLING_BELOW
