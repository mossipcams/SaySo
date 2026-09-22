"""Colocated tests for scripts/wake_audio_replay.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_audio_replay as replay  # noqa: E402
from scripts import wake_hard_negative_mine as mine  # noqa: E402


def _write_wav(
    path: Path,
    *,
    rate: int = 16000,
    channels: int = 1,
    n_samples: int = replay.WINDOW_SAMPLES,
    fill: float = 0.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 1:
        data = np.full(n_samples, fill, dtype=np.float32)
    else:
        left = np.full(n_samples, fill, dtype=np.float32)
        right = np.full(n_samples, fill + 0.5, dtype=np.float32)
        data = np.stack([left, right], axis=1)
    sf.write(path, data, rate)


class _TrackingScorer:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.reset_count = 0

    def score(self, window: np.ndarray) -> float:
        assert window.size == replay.WINDOW_SAMPLES
        return self.value

    def reset(self) -> None:
        self.reset_count += 1


def _run_with_scorer(tmp_path: Path, sources: list[Path], scorer: _TrackingScorer, **kwargs) -> replay.ReplayResult:
    scores_path = tmp_path / "scores.npy"
    events_path = tmp_path / "events.jsonl"
    summary_path = tmp_path / "summary.json"
    model_stub = tmp_path / "model.onnx"
    model_stub.write_bytes(b"stub-onnx")
    config = replay.ReplayConfig(
        input_paths=tuple(sources),
        model_path=model_stub,
        scores_path=scores_path,
        events_path=events_path,
        summary_path=summary_path,
        threshold=kwargs.get("threshold", 0.5),
        refractory_seconds=kwargs.get("refractory_seconds", 2.0),
        chunk_size=kwargs.get("chunk_size", 2),
        gain_db=kwargs.get("gain_db", 0.0),
    )

    class _Adapter:
        def __init__(self, inner: _TrackingScorer) -> None:
            self._inner = inner

        def score(self, window: np.ndarray) -> float:
            return self._inner.score(window)

        def reset(self) -> None:
            self._inner.reset()

    return replay.run_replay(config, window_scorer=_Adapter(scorer))


def test_enumerate_audio_inputs_deterministic_and_recursive(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    (root / "b").mkdir(parents=True)
    (root / "a").mkdir()
    late = root / "b" / "second.wav"
    early = root / "a" / "first.flac"
    _write_wav(late)
    _write_wav(early)
    top = tmp_path / "solo.wav"
    _write_wav(top)

    first = replay.enumerate_audio_inputs([root, top])
    second = replay.enumerate_audio_inputs([top, root])
    assert first == second
    assert [p.name for p in first] == ["first.flac", "second.wav", "solo.wav"]


def test_enumerate_rejects_missing_and_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        replay.enumerate_audio_inputs([tmp_path / "missing"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no audio"):
        replay.enumerate_audio_inputs([empty])


def test_plan_audio_windows_hop_geometry() -> None:
    assert replay.plan_audio_windows(replay.WINDOW_SAMPLES - 1) == (0, 0, replay.WINDOW_SAMPLES - 1, 0)
    assert replay.plan_audio_windows(replay.WINDOW_SAMPLES) == (1, replay.WINDOW_SAMPLES, 0, 0)
    wc, used, dropped, last = replay.plan_audio_windows(replay.WINDOW_SAMPLES + replay.HOP_SAMPLES)
    assert wc == 2
    assert last == replay.HOP_SAMPLES
    assert used == last + replay.WINDOW_SAMPLES
    assert dropped == 0


def test_load_mono_16k_downmix_and_resample(tmp_path: Path) -> None:
    stereo_path = tmp_path / "stereo.wav"
    _write_wav(stereo_path, channels=2, fill=0.25)
    pcm, meta = replay.load_mono_16k(stereo_path)
    assert meta["source_channels"] == 2
    assert pcm.dtype == np.int16
    assert pcm.size == replay.WINDOW_SAMPLES

    eight_k = tmp_path / "eight.wav"
    _write_wav(eight_k, rate=8000, n_samples=8000)
    pcm8, meta8 = replay.load_mono_16k(eight_k)
    assert meta8["source_sample_rate"] == 8000
    assert pcm8.size == 16000


def test_score_audio_chunk_size_is_independent_of_window_count(tmp_path: Path) -> None:
    path = tmp_path / "one.wav"
    n = replay.WINDOW_SAMPLES + replay.HOP_SAMPLES * 3
    _write_wav(path, n_samples=n)
    pcm, _ = replay.load_mono_16k(path)
    tracker = _TrackingScorer()
    expected = replay.plan_audio_windows(pcm.size)[0]
    for chunk_size in (1, 2, 4):
        scores = replay.score_audio_in_chunks(
            pcm,
            tracker,
            chunk_size=chunk_size,
            file_path=path,
        )
        assert scores.size == expected == 4


def test_no_cross_file_windows(tmp_path: Path) -> None:
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    _write_wav(a, n_samples=replay.WINDOW_SAMPLES)
    _write_wav(b, n_samples=replay.WINDOW_SAMPLES + replay.HOP_SAMPLES)
    tracker = _TrackingScorer()
    result = _run_with_scorer(tmp_path, [a, b], tracker, chunk_size=1)
    assert tracker.reset_count >= 2
    assert result.scores.size == 3
    files = result.summary["files"]
    assert files[0]["score_offset"] == 0
    assert files[0]["score_count"] == 1
    assert files[1]["score_offset"] == 1
    assert files[1]["score_count"] == 2


def test_cluster_crossings_audio_samples_and_refractory() -> None:
    scores = np.array([0.1, 0.9, 0.85, 0.2, 0.95, 0.94, 0.1], dtype=np.float64)
    path = Path("/tmp/example.wav")
    events = replay.cluster_crossings(
        scores,
        threshold=0.5,
        refractory_seconds=0.0,
        file_path=path,
    )
    assert len(events) == 2
    assert events[0]["start_sample"] == replay.HOP_SAMPLES
    assert events[0]["end_sample"] == 2 * replay.HOP_SAMPLES + replay.WINDOW_SAMPLES
    assert events[0]["file"].endswith("example.wav")

    refractory_scores = np.zeros(6, dtype=np.float64)
    refractory_scores[0] = 0.9
    refractory_scores[1] = 0.8
    refractory_scores[3] = 0.99
    merged = replay.cluster_crossings(
        refractory_scores,
        threshold=0.5,
        refractory_seconds=10.0,
        file_path=path,
    )
    assert len(merged) == 1
    assert merged[0]["peak_window"] == 3


def test_duration_fpph_and_threshold_counts(tmp_path: Path) -> None:
    wav = tmp_path / "short.wav"
    _write_wav(wav, n_samples=replay.WINDOW_SAMPLES)
    tracker = _TrackingScorer(value=0.9)
    result = _run_with_scorer(tmp_path, [wav], tracker, refractory_seconds=0.0)
    summary = result.summary
    assert summary["duration_seconds"] == pytest.approx(replay.WINDOW_SAMPLES / replay.SAMPLE_RATE)
    assert summary["total_hours"] == pytest.approx(summary["duration_seconds"] / 3600.0)
    assert summary["clustered_event_count"] == 1
    assert summary["false_activations_per_hour"] == pytest.approx(3600.0 / summary["duration_seconds"])
    assert summary["threshold_counts"]["ge_0_4"] == 1
    assert summary["threshold_counts"]["ge_threshold"] == 1


def test_run_replay_writes_deterministic_artifacts(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    _write_wav(wav, n_samples=replay.WINDOW_SAMPLES + replay.HOP_SAMPLES)
    tracker = _TrackingScorer(value=0.42)
    config_scores = tmp_path / "out" / "scores.npy"
    config_events = tmp_path / "out" / "events.jsonl"
    config_summary = tmp_path / "out" / "summary.json"
    model_stub = tmp_path / "model.onnx"
    model_stub.write_bytes(b"stub-onnx")
    config = replay.ReplayConfig(
        input_paths=(wav,),
        model_path=model_stub,
        scores_path=config_scores,
        events_path=config_events,
        summary_path=config_summary,
        threshold=0.5,
        refractory_seconds=2.0,
        chunk_size=3,
    )
    first = replay.run_replay(config, window_scorer=tracker)
    second = replay.run_replay(config, window_scorer=tracker)
    assert np.array_equal(first.scores, second.scores)
    assert first.summary["outputs"]["scores_sha256"] == second.summary["outputs"]["scores_sha256"]
    doc = json.loads(config_summary.read_text(encoding="utf-8"))
    assert doc["files"][0]["sha256"] == mine._sha256_file(wav)
    assert doc["model"]["sha256"] == mine._sha256_file(model_stub)


def test_run_replay_immutability_and_output_guards(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    _write_wav(wav)
    model_stub = tmp_path / "model.onnx"
    model_stub.write_bytes(b"stub-onnx")
    before_audio = wav.read_bytes()
    before_model = model_stub.read_bytes()
    tracker = _TrackingScorer()
    _run_with_scorer(tmp_path, [wav], tracker)
    assert wav.read_bytes() == before_audio
    assert model_stub.read_bytes() == before_model

    bad = replay.ReplayConfig(
        input_paths=(wav,),
        model_path=model_stub,
        scores_path=wav,
        events_path=tmp_path / "events.jsonl",
        summary_path=tmp_path / "summary.json",
        threshold=0.5,
        refractory_seconds=2.0,
        chunk_size=2,
    )
    with pytest.raises(ValueError, match="overwrite input"):
        replay.run_replay(bad, window_scorer=tracker)


def test_main_help_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "wake_audio_replay.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--input" in proc.stdout
    assert "--gain-db" in proc.stdout


def test_gain_db_zero_is_identity(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    fill = 0.25
    _write_wav(path, fill=fill)
    pcm_default, meta_default = replay.load_mono_16k(path)
    pcm_explicit, meta_explicit = replay.load_mono_16k(path, gain_db=0.0)
    assert np.array_equal(pcm_default, pcm_explicit)
    assert meta_default["gain_db"] == 0.0
    assert meta_default["gain_linear"] == pytest.approx(1.0)
    assert meta_default["pre_gain_peak"] == pytest.approx(fill, abs=1e-6)
    assert meta_default["post_gain_clip_count"] == 0


def test_gain_db_10_increases_amplitude(tmp_path: Path) -> None:
    path = tmp_path / "quiet.wav"
    fill = 0.1
    _write_wav(path, fill=fill)
    pcm0, _ = replay.load_mono_16k(path, gain_db=0.0)
    pcm10, meta10 = replay.load_mono_16k(path, gain_db=10.0)
    expected_linear = 10.0 ** (10.0 / 20.0)
    assert meta10["gain_linear"] == pytest.approx(expected_linear)
    assert meta10["pre_gain_peak"] == pytest.approx(fill, rel=0.01)
    assert meta10["post_gain_clip_count"] == 0
    ratio = float(np.max(np.abs(pcm10.astype(np.float64)))) / float(
        np.max(np.abs(pcm0.astype(np.float64)))
    )
    assert ratio == pytest.approx(expected_linear, rel=0.02)


def test_gain_db_clipping_accounted(tmp_path: Path) -> None:
    path = tmp_path / "hot.wav"
    _write_wav(path, fill=0.5)
    _, meta = replay.load_mono_16k(path, gain_db=10.0)
    assert meta["post_gain_clip_count"] > 0
    assert meta["post_gain_clip_fraction"] == pytest.approx(1.0)


def test_validate_gain_db_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        replay.validate_gain_db(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        replay.validate_gain_db(float("inf"))


def test_main_rejects_non_finite_gain_db(tmp_path: Path) -> None:
    wav = tmp_path / "x.wav"
    _write_wav(wav)
    rc = replay.main(
        [
            "--input",
            str(wav),
            "--model",
            str(tmp_path / "model.onnx"),
            "--scores",
            str(tmp_path / "scores.npy"),
            "--events",
            str(tmp_path / "events.jsonl"),
            "--summary",
            str(tmp_path / "summary.json"),
            "--gain-db",
            "nan",
        ]
    )
    assert rc == 2


def test_run_replay_summary_propagates_gain_db(tmp_path: Path) -> None:
    wav = tmp_path / "clip.wav"
    _write_wav(wav, n_samples=replay.WINDOW_SAMPLES, fill=0.2)
    tracker = _TrackingScorer(value=0.1)
    result = _run_with_scorer(tmp_path, [wav], tracker, gain_db=10.0)
    assert result.summary["gain_db"] == 10.0
    assert result.summary["gain_linear"] == pytest.approx(10.0 ** 0.5)
    file_meta = result.summary["files"][0]
    assert file_meta["gain_db"] == 10.0
    assert file_meta["pre_gain_peak"] > 0.0
    assert "post_gain_clip_fraction" in file_meta


def test_main_rejects_missing_model(tmp_path: Path) -> None:
    wav = tmp_path / "x.wav"
    _write_wav(wav)
    rc = replay.main(
        [
            "--input",
            str(wav),
            "--model",
            str(tmp_path / "missing.onnx"),
            "--scores",
            str(tmp_path / "scores.npy"),
            "--events",
            str(tmp_path / "events.jsonl"),
            "--summary",
            str(tmp_path / "summary.json"),
        ]
    )
    assert rc == 1
