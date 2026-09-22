"""Colocated tests for scripts/wake_feature_replay.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_feature_replay as replay  # noqa: E402
from scripts import wake_hard_negative_mine as mine  # noqa: E402


def _write_stream(path: Path, n_rows: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n_rows, replay.EMBEDDING_DIM), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def _constant_scorer(value: float) -> replay.ScoreBatchFn:
    def score_batch(batch: np.ndarray) -> np.ndarray:
        return np.full(batch.shape[0], value, dtype=np.float64)

    return score_batch


def _index_scorer() -> replay.ScoreBatchFn:
    def score_batch(batch: np.ndarray) -> np.ndarray:
        return batch.reshape(batch.shape[0], -1).mean(axis=1).astype(np.float64)

    return score_batch


def test_plan_stream_windows_drops_incomplete_trailing_rows() -> None:
    assert replay.plan_stream_windows(10) == (0, 0, 10, 0)
    assert replay.plan_stream_windows(15) == (0, 0, 15, 0)
    assert replay.plan_stream_windows(16) == (1, 16, 0, 0)
    assert replay.plan_stream_windows(20) == (1, 16, 4, 0)
    assert replay.plan_stream_windows(21) == (1, 16, 5, 0)
    assert replay.plan_stream_windows(32) == (2, 32, 0, 16)


def test_window_starts_non_overlapping_blocks() -> None:
    assert np.array_equal(replay.window_starts(32), np.array([0, 16], dtype=np.int64))
    assert np.array_equal(replay.window_starts(20), np.array([0], dtype=np.int64))
    assert replay.window_starts(10).size == 0


def test_scorer_never_receives_cross_boundary_windows(tmp_path: Path) -> None:
    source_path = tmp_path / "stream.npy"
    n_rows = 32
    arr = np.zeros((n_rows, replay.EMBEDDING_DIM), dtype=np.float32)
    for row in range(n_rows):
        arr[row, 0] = float(row)
    np.save(source_path, arr)
    stream = replay.open_feature_stream(source_path)

    def scorer(batch: np.ndarray) -> np.ndarray:
        for window in batch:
            rows = window[:, 0].astype(np.int64)
            start = int(rows[0])
            assert start % replay.FRAMES_PER_WINDOW == 0
            assert np.array_equal(rows, np.arange(start, start + replay.FRAMES_PER_WINDOW))
        return np.zeros(batch.shape[0], dtype=np.float64)

    replay.score_windows_in_chunks(stream, scorer, chunk_size=4)


def test_validate_feature_stream_shape_rejects_malformed() -> None:
    bad_dim = np.zeros((4, 95), dtype=np.float32)
    with pytest.raises(ValueError, match="96"):
        replay.validate_feature_stream_shape(bad_dim)
    with pytest.raises(ValueError, match="ndim"):
        replay.validate_feature_stream_shape(np.zeros((4, 16, 96), dtype=np.float32))


def test_score_windows_in_chunks_materializes_bounded_batches(tmp_path: Path) -> None:
    source_path = tmp_path / "stream.npy"
    _write_stream(source_path, 21, seed=1)
    stream = replay.open_feature_stream(source_path)
    max_batch = 0

    def scorer(batch: np.ndarray) -> np.ndarray:
        nonlocal max_batch
        max_batch = max(max_batch, batch.shape[0])
        assert batch.shape[1:] == (replay.FRAMES_PER_WINDOW, replay.EMBEDDING_DIM)
        assert not isinstance(batch, np.memmap)
        return _index_scorer()(batch)

    scores = replay.score_windows_in_chunks(stream, scorer, chunk_size=2)
    assert scores.shape == (1,)
    assert max_batch == 1


def test_cluster_crossings_merges_adjacent_and_refractory() -> None:
    scores = np.array([0.1, 0.9, 0.85, 0.2, 0.95, 0.94, 0.1], dtype=np.float64)
    events = replay.cluster_crossings(scores, threshold=0.5, refractory_seconds=0.0)
    assert len(events) == 2
    assert events[0]["start_window"] == 1
    assert events[0]["peak_window"] == 1
    assert events[0]["end_window"] == 2
    assert events[1]["start_window"] == 4
    assert events[1]["peak_window"] == 4


def test_cluster_crossings_refractory_suppresses_nearby_second_peak() -> None:
    scores = np.zeros(6, dtype=np.float64)
    scores[0] = 0.9
    scores[1] = 0.8
    scores[3] = 0.99
    events = replay.cluster_crossings(scores, threshold=0.5, refractory_seconds=10.0)
    assert len(events) == 1
    assert events[0]["start_window"] == 0
    assert events[0]["end_window"] == 3
    assert events[0]["peak_window"] == 3


def test_duration_and_fpph_in_summary(tmp_path: Path) -> None:
    source_path = tmp_path / "stream.npy"
    _write_stream(source_path, 16, seed=2)
    scores_path = tmp_path / "scores.npy"
    events_path = tmp_path / "events.jsonl"
    summary_path = tmp_path / "summary.json"
    model_stub = tmp_path / "model.onnx"
    model_stub.write_bytes(b"stub-onnx")

    config = replay.ReplayConfig(
        source_path=source_path,
        model_path=model_stub,
        scores_path=scores_path,
        events_path=events_path,
        summary_path=summary_path,
        threshold=0.5,
        refractory_seconds=0.0,
        chunk_size=4,
    )
    result = replay.run_replay(config, scorer=_constant_scorer(0.9))
    summary = result.summary
    assert summary["windows"]["window_seconds"] == pytest.approx(1.28)
    assert summary["duration_seconds"] == pytest.approx(16 * replay.FRAME_SECONDS)
    assert summary["total_hours"] == pytest.approx(summary["duration_seconds"] / 3600.0)
    assert summary["clustered_event_count"] == 1
    assert summary["false_activations_per_hour"] == pytest.approx(3600.0 / summary["duration_seconds"])


def test_run_replay_writes_deterministic_artifacts(tmp_path: Path) -> None:
    source_path = tmp_path / "stream.npy"
    _write_stream(source_path, 20, seed=3)
    out_dir = tmp_path / "run"
    scores_path = out_dir / "scores.npy"
    events_path = out_dir / "events.jsonl"
    summary_path = out_dir / "summary.json"
    model_stub = tmp_path / "model.onnx"
    model_stub.write_bytes(b"stub-onnx")

    config = replay.ReplayConfig(
        source_path=source_path,
        model_path=model_stub,
        scores_path=scores_path,
        events_path=events_path,
        summary_path=summary_path,
        threshold=0.5,
        refractory_seconds=2.0,
        chunk_size=3,
    )
    first = replay.run_replay(config, scorer=_index_scorer())
    second = replay.run_replay(config, scorer=_index_scorer())

    assert np.array_equal(first.scores, second.scores)
    assert first.events == second.events
    assert first.summary["outputs"]["scores_sha256"] == second.summary["outputs"]["scores_sha256"]
    assert first.summary["outputs"]["events_sha256"] == second.summary["outputs"]["events_sha256"]

    loaded_scores = np.load(scores_path)
    assert loaded_scores.shape == first.scores.shape
    doc = json.loads(summary_path.read_text(encoding="utf-8"))
    assert doc["source"]["sha256"] == mine._sha256_file(source_path)
    assert doc["model"]["sha256"] == mine._sha256_file(model_stub)
    assert doc["source"]["dropped_trailing_rows"] == 4
    assert doc["windows"]["shape"][1:] == [replay.FRAMES_PER_WINDOW, replay.EMBEDDING_DIM]


def test_run_replay_does_not_mutate_source_or_model(tmp_path: Path) -> None:
    source_path = tmp_path / "stream.npy"
    _write_stream(source_path, 18, seed=4)
    model_stub = tmp_path / "model.onnx"
    model_stub.write_bytes(b"stub-onnx")
    source_before = source_path.read_bytes()
    model_before = model_stub.read_bytes()

    config = replay.ReplayConfig(
        source_path=source_path,
        model_path=model_stub,
        scores_path=tmp_path / "scores.npy",
        events_path=tmp_path / "events.jsonl",
        summary_path=tmp_path / "summary.json",
        threshold=0.5,
        refractory_seconds=2.0,
        chunk_size=2,
    )
    replay.run_replay(config, scorer=_index_scorer())
    assert source_path.read_bytes() == source_before
    assert model_stub.read_bytes() == model_before


def test_run_replay_rejects_output_overwriting_source(tmp_path: Path) -> None:
    source_path = tmp_path / "stream.npy"
    _write_stream(source_path, 16, seed=5)
    model_stub = tmp_path / "model.onnx"
    model_stub.write_bytes(b"stub-onnx")
    config = replay.ReplayConfig(
        source_path=source_path,
        model_path=model_stub,
        scores_path=source_path,
        events_path=tmp_path / "events.jsonl",
        summary_path=tmp_path / "summary.json",
        threshold=0.5,
        refractory_seconds=2.0,
        chunk_size=2,
    )
    with pytest.raises(ValueError, match="overwrite source"):
        replay.run_replay(config, scorer=_index_scorer())


def test_main_help_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "wake_feature_replay.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--source" in proc.stdout


def test_main_rejects_missing_source(tmp_path: Path) -> None:
    rc = replay.main(
        [
            "--source",
            str(tmp_path / "missing.npy"),
            "--model",
            str(tmp_path / "model.onnx"),
            "--scores",
            str(tmp_path / "scores.npy"),
            "--events",
            str(tmp_path / "events.jsonl"),
            "--summary",
            str(tmp_path / "summary.json"),
        ]
    )
    assert rc == 1
