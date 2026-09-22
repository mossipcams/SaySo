"""Colocated tests for scripts/wake_hard_negative_mine.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_hard_negative_mine as mine  # noqa: E402


def _write_acav_npy(path: Path, n: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, 16, 96), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def _index_mean_scorer() -> mine.ScoreBatchFn:
    def score_batch(batch: np.ndarray) -> np.ndarray:
        # Deterministic per-row score from content so tests need no ONNX.
        return batch.reshape(batch.shape[0], -1).mean(axis=1).astype(np.float64)

    return score_batch


def _select_diverse_top_k_brute_force(
    scores: np.ndarray,
    *,
    top_k: int,
    diversity_gap: int,
) -> list[tuple[int, float]]:
    n = int(scores.shape[0])
    if top_k == 0 or n == 0:
        return []
    order = sorted(range(n), key=lambda idx: (-float(scores[idx]), idx))
    selected: list[tuple[int, float]] = []
    picked: list[int] = []
    for idx in order:
        if len(selected) >= top_k:
            break
        if diversity_gap > 0 and any(abs(idx - prior) <= diversity_gap for prior in picked):
            continue
        selected.append((idx, float(scores[idx])))
        picked.append(idx)
    return selected


@pytest.mark.parametrize(
    ("n", "top_k", "gap", "seed"),
    [
        (1, 1, 0, 0),
        (5, 3, 0, 1),
        (5, 3, 1, 2),
        (20, 8, 2, 3),
        (50, 10, 3, 4),
        (30, 100, 1, 5),
        (12, 4, 0, 6),
    ],
)
def test_select_diverse_top_k_matches_brute_force(
    n: int,
    top_k: int,
    gap: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal(n, dtype=np.float64)
    # Inject tied scores so index tie-break is exercised.
    if n >= 3:
        scores[1] = scores[0]
        scores[2] = scores[0]
    optimized = mine.select_diverse_top_k(scores, top_k=top_k, diversity_gap=gap)
    brute = _select_diverse_top_k_brute_force(scores, top_k=top_k, diversity_gap=gap)
    assert optimized == brute


def test_select_diverse_top_k_all_equal_scores_fewer_than_top_k() -> None:
    scores = np.full(6, 0.5, dtype=np.float64)
    optimized = mine.select_diverse_top_k(scores, top_k=5, diversity_gap=1)
    brute = _select_diverse_top_k_brute_force(scores, top_k=5, diversity_gap=1)
    assert optimized == brute
    assert len(optimized) < 5


def test_select_diverse_top_k_is_deterministic() -> None:
    scores = np.array([0.1, 0.9, 0.85, 0.8, 0.2], dtype=np.float64)
    first = mine.select_diverse_top_k(scores, top_k=3, diversity_gap=0)
    second = mine.select_diverse_top_k(scores, top_k=3, diversity_gap=0)
    assert first == second
    assert first == [(1, 0.9), (2, 0.85), (3, 0.8)]


def test_select_diverse_top_k_skips_adjacent_rows() -> None:
    scores = np.array([0.5, 0.95, 0.94, 0.93, 0.1], dtype=np.float64)
    selected = mine.select_diverse_top_k(scores, top_k=3, diversity_gap=1)
    indices = [idx for idx, _ in selected]
    assert indices == [1, 3]
    assert all(abs(indices[i] - indices[j]) > 1 for i in range(len(indices)) for j in range(i + 1, len(indices)))


def test_validate_acav_shape_rejects_malformed() -> None:
    bad_rows = np.zeros((4, 15, 96), dtype=np.float32)
    with pytest.raises(ValueError, match="16, 96"):
        mine.validate_acav_shape(bad_rows)
    with pytest.raises(ValueError, match="ndim"):
        mine.validate_acav_shape(np.zeros((4, 16), dtype=np.float32))


def test_open_acav_features_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npy"
    with pytest.raises(FileNotFoundError):
        mine.open_acav_features(missing)


def test_score_features_in_chunks_invokes_scorer_per_chunk() -> None:
    n = 10
    features = np.arange(n * 16 * 96, dtype=np.float32).reshape(n, 16, 96)
    calls: list[tuple[int, int]] = []

    def scorer(batch: np.ndarray) -> np.ndarray:
        calls.append((batch.shape[0], int(batch[0, 0, 0])))
        return np.full(batch.shape[0], 0.5, dtype=np.float64)

    scores = mine.score_features_in_chunks(features, scorer, chunk_size=3)
    assert scores.shape == (n,)
    assert calls == [(3, 0), (3, 3 * 16 * 96), (3, 6 * 16 * 96), (1, 9 * 16 * 96)]


def test_run_mine_writes_provenance_and_features(tmp_path: Path) -> None:
    source_path = tmp_path / "acav.npy"
    _write_acav_npy(source_path, 12, seed=1)
    out_features = tmp_path / "selected.npy"
    provenance_path = tmp_path / "selected.provenance.json"
    model_stub = tmp_path / "model.onnx"
    model_stub.write_bytes(b"stub-onnx")

    config = mine.MineConfig(
        features_path=source_path,
        output_features_path=out_features,
        provenance_path=provenance_path,
        model_path=model_stub,
        top_k=4,
        chunk_size=5,
        diversity_gap=1,
    )
    result = mine.run_mine(config, scorer=_index_mean_scorer())

    assert out_features.is_file()
    assert provenance_path.is_file()
    loaded = np.load(out_features)
    assert loaded.shape[1:] == (16, 96)
    assert loaded.shape[0] == len(result.selected_indices)

    doc = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert doc["version"] == mine.PROVENANCE_VERSION
    assert doc["source"]["sha256"] == mine._sha256_file(source_path)
    assert doc["source"]["shape"] == [12, 16, 96]
    assert doc["model"]["sha256"] == mine._sha256_file(model_stub)
    assert doc["selection"]["top_k"] == 4
    assert doc["selection"]["chunk_size"] == 5
    assert doc["selection"]["diversity_gap"] == 1
    assert doc["selection"]["selected_indices"] == list(result.selected_indices)
    assert doc["selection"]["selected_scores"] == list(result.selected_scores)
    assert doc["output"]["shape"] == list(loaded.shape)
    assert doc["output"]["sha256"] == mine._sha256_file(out_features)


def test_run_mine_does_not_mutate_source_file(tmp_path: Path) -> None:
    source_path = tmp_path / "acav.npy"
    _write_acav_npy(source_path, 8, seed=3)
    before = source_path.read_bytes()
    config = mine.MineConfig(
        features_path=source_path,
        output_features_path=tmp_path / "out.npy",
        provenance_path=tmp_path / "out.provenance.json",
        model_path=None,
        top_k=3,
        chunk_size=2,
        diversity_gap=0,
    )
    mine.run_mine(config, scorer=_index_mean_scorer())
    after = source_path.read_bytes()
    assert before == after


def test_mmap_scoring_materializes_bounded_chunks(tmp_path: Path) -> None:
    source_path = tmp_path / "acav.npy"
    _write_acav_npy(source_path, 20, seed=4)
    features = mine.open_acav_features(source_path)
    max_batch = 0

    def scorer(batch: np.ndarray) -> np.ndarray:
        nonlocal max_batch
        max_batch = max(max_batch, batch.shape[0])
        assert not isinstance(batch, np.memmap)
        return _index_mean_scorer()(batch)

    scores = mine.score_features_in_chunks(features, scorer, chunk_size=4)
    assert scores.shape == (20,)
    assert max_batch == 4


def test_atomic_outputs_replace_partial_files(tmp_path: Path) -> None:
    source_path = tmp_path / "acav.npy"
    _write_acav_npy(source_path, 6, seed=5)
    out_features = tmp_path / "selected.npy"
    provenance_path = tmp_path / "selected.provenance.json"
    out_features.write_bytes(b"stale")
    provenance_path.write_text('{"stale": true}\n', encoding="utf-8")

    config = mine.MineConfig(
        features_path=source_path,
        output_features_path=out_features,
        provenance_path=provenance_path,
        model_path=None,
        top_k=2,
        chunk_size=3,
        diversity_gap=0,
    )
    mine.run_mine(config, scorer=_index_mean_scorer())

    loaded = np.load(out_features)
    assert loaded.ndim == 3
    doc = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert "stale" not in doc
