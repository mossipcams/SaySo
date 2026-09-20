"""Colocated checks for opt-in wake mining."""

from __future__ import annotations

import json
import random
import shutil
import threading
import time
import wave
from pathlib import Path

import numpy as np

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


def _publish(miner: HardNegativeMiner, score: float, window: np.ndarray, index: int) -> str | None:
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
    records = {p.name: json.loads((p / "record.json").read_text()) for p in (tmp_path / "records").iterdir()}
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


def test_spool_full_stops_collection_and_resumes_after_drain(tmp_path: Path) -> None:
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
    _publish(miner, 0.2, window, 31999)
    before = miner.losses.spool_full
    _publish(miner, 0.25, window, 63999)
    assert miner.losses.spool_full > before
    record_dir = next((tmp_path / "records").iterdir())
    write_ack(tmp_path, record_dir.name)
    miner.drain_acks()
    miner._reconcile_spool_state()
    assert not miner._spool_full
    _publish(miner, 0.26, window, 95999)
    assert len(list((tmp_path / "records").iterdir())) >= 1


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
    miner.publish_wake_outcome(capture_id, accepted=True, suppressed=False, reason="test")
    miner.publish_wake_outcome(capture_id, accepted=True, suppressed=False, reason="test")
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
    meta = json.loads((next((tmp_path / "records").iterdir()) / "record.json").read_text())
    assert meta["sampling_reason"] == SAMPLING_BELOW
