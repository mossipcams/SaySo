"""Colocated tests for scripts/wake_audio_hard_negative_mine.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_audio_hard_negative_mine as audio_mine  # noqa: E402
from scripts import wake_audio_replay as replay  # noqa: E402
from scripts import wake_hard_negative_mine as mine  # noqa: E402


def _write_wav(
    path: Path,
    *,
    rate: int = 16000,
    n_samples: int = replay.WINDOW_SAMPLES,
    fill: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full(n_samples, fill, dtype=np.float32)
    sf.write(path, data, rate)


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


class _FakeEmbeddingScorer:
    """Deterministic scores/embeddings without ONNX."""

    def __init__(self, *, base_score: float = 0.0, score_step: float = 0.05) -> None:
        self.base_score = base_score
        self.score_step = score_step
        self.reset_count = 0
        self._last: np.ndarray | None = None
        self._window_index = 0

    def reset(self) -> None:
        self.reset_count += 1
        self._window_index = 0

    def score(self, window: np.ndarray) -> float:
        assert window.size == replay.WINDOW_SAMPLES
        value = self.base_score + self.score_step * self._window_index
        self._window_index += 1
        seed = int(np.abs(window[0])) % 997
        self._last = np.full(mine.ACAV_ROW_SHAPE, float(seed) + value, dtype=np.float32)
        return float(value)

    def last_embedding(self) -> np.ndarray:
        if self._last is None:
            raise RuntimeError("score() must be called before last_embedding()")
        return self._last


def _run_with_scorer(
    tmp_path: Path,
    sources: list[Path],
    scorer: _FakeEmbeddingScorer,
    **kwargs: object,
) -> audio_mine.AudioMineResult:
    tmp_path.mkdir(parents=True, exist_ok=True)
    out_features = tmp_path / "selected.npy"
    provenance_path = tmp_path / "selected.provenance.json"
    model_stub = tmp_path / "model.onnx"
    model_stub.write_bytes(b"stub-onnx")
    config = audio_mine.AudioMineConfig(
        input_paths=tuple(sources),
        output_features_path=out_features,
        provenance_path=provenance_path,
        model_path=model_stub,
        top_k=int(kwargs.get("top_k", 4)),
        chunk_size=int(kwargs.get("chunk_size", 2)),
        diversity_gap=int(kwargs.get("diversity_gap", 0)),
        score_floor=float(kwargs.get("score_floor", 0.25)),
        gain_db=float(kwargs.get("gain_db", 0.0)),
    )
    return audio_mine.run_audio_mine(config, window_scorer=scorer)


@pytest.mark.parametrize(
    ("n", "top_k", "gap", "seed"),
    [
        (1, 1, 0, 0),
        (5, 3, 0, 1),
        (5, 3, 1, 2),
        (20, 8, 2, 3),
        (50, 10, 3, 4),
    ],
)
def test_select_from_candidates_matches_brute_force(
    n: int,
    top_k: int,
    gap: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal(n, dtype=np.float64)
    if n >= 3:
        scores[1] = scores[0]
        scores[2] = scores[0]
    candidates = [
        audio_mine.CandidateWindow(
            path=Path(f"/tmp/f{idx}.wav"),
            start_sample=idx * replay.HOP_SAMPLES,
            score=float(scores[idx]),
            embedding=np.zeros(mine.ACAV_ROW_SHAPE, dtype=np.float32),
        )
        for idx in range(n)
    ]
    optimized = audio_mine.select_from_candidates(
        candidates,
        top_k=top_k,
        diversity_gap=gap,
    )
    brute = _select_diverse_top_k_brute_force(scores, top_k=top_k, diversity_gap=gap)
    assert [(idx, candidates[idx].score) for idx, _ in optimized] == brute


def test_run_audio_mine_writes_shape_and_provenance(tmp_path: Path) -> None:
    wav = tmp_path / "one.wav"
    n = replay.WINDOW_SAMPLES + replay.HOP_SAMPLES * 5
    _write_wav(wav, n_samples=n)
    scorer = _FakeEmbeddingScorer(base_score=0.2, score_step=0.05)
    result = _run_with_scorer(
        tmp_path,
        [wav],
        scorer,
        top_k=3,
        diversity_gap=1,
        score_floor=0.25,
    )
    out_features = tmp_path / "selected.npy"
    provenance_path = tmp_path / "selected.provenance.json"
    assert out_features.is_file()
    assert provenance_path.is_file()
    loaded = np.load(out_features)
    assert loaded.shape[1:] == mine.ACAV_ROW_SHAPE
    assert loaded.shape[0] == len(result.selected_indices)
    assert loaded.ndim == 3

    doc = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert doc["version"] == audio_mine.PROVENANCE_VERSION
    assert doc["source"]["shape"][1:] == [16, 96]
    assert doc["source"]["score_floor"] == 0.25
    assert doc["model"]["sha256"] == mine._sha256_file(tmp_path / "model.onnx")
    assert doc["selection"]["selected_indices"] == list(result.selected_indices)
    assert doc["selection"]["selected_scores"] == list(result.selected_scores)
    assert doc["output"]["shape"] == list(loaded.shape)
    assert doc["output"]["sha256"] == mine._sha256_file(out_features)
    assert doc["source"]["files"][0]["sha256"] == mine._sha256_file(wav)
    for entry, (idx, score) in zip(
        doc["selection"]["selected_windows"],
        zip(result.selected_indices, result.selected_scores),
        strict=True,
    ):
        assert entry["candidate_index"] == idx
        assert entry["score"] == score
        assert "path" in entry
        assert "start_sample" in entry


def test_run_audio_mine_does_not_mutate_source_audio(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    _write_wav(wav, n_samples=replay.WINDOW_SAMPLES + replay.HOP_SAMPLES)
    before = wav.read_bytes()
    scorer = _FakeEmbeddingScorer(base_score=0.9)
    _run_with_scorer(tmp_path, [wav], scorer, score_floor=0.25, top_k=2)
    after = wav.read_bytes()
    assert before == after


def test_score_floor_excludes_low_windows(tmp_path: Path) -> None:
    wav = tmp_path / "quiet.wav"
    _write_wav(wav, n_samples=replay.WINDOW_SAMPLES)
    low = _FakeEmbeddingScorer(base_score=0.1)
    result_low = _run_with_scorer(tmp_path / "a", [wav], low, score_floor=0.25, top_k=5)
    assert result_low.candidate_count == 0
    assert result_low.output_shape == (0, 16, 96)

    high = _FakeEmbeddingScorer(base_score=0.8)
    result_high = _run_with_scorer(tmp_path / "b", [wav], high, score_floor=0.25, top_k=5)
    assert result_high.candidate_count == 1
    assert result_high.output_shape[0] == 1


def test_collect_candidates_respects_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "long.wav"
    n = replay.WINDOW_SAMPLES + replay.HOP_SAMPLES * 3
    _write_wav(path, n_samples=n)
    pcm, _ = replay.load_mono_16k(path)
    scorer = _FakeEmbeddingScorer(base_score=0.5)
    for chunk_size in (1, 2, 4):
        scorer.reset()
        got = audio_mine.collect_candidates_from_pcm(
            pcm,
            scorer,
            file_path=path,
            score_floor=0.25,
            chunk_size=chunk_size,
        )
        assert len(got) == replay.plan_audio_windows(pcm.size)[0]
