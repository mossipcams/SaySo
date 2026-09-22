#!/usr/bin/env python3
"""Replay raw wake audio through the production LiveKit frontend and log scores.

Enumerates ``.wav`` / ``.flac`` inputs (files or directories), resamples to 16 kHz
mono, scores every 2 s / 160 ms hop window per file with ``CachedEmbeddingScorer``,
and writes concatenated scores, clustered threshold events, and provenance.

    python scripts/wake_audio_replay.py \\
        --input /data/negatives \\
        --model /path/sayso.onnx \\
        --scores /runs/audio_replay/scores.npy \\
        --events /runs/audio_replay/events.jsonl \\
        --summary /runs/audio_replay/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_hard_negative_mine as mine  # noqa: E402

PROVENANCE_VERSION = 1
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 32000
HOP_SAMPLES = 2560
HOP_SECONDS = HOP_SAMPLES / SAMPLE_RATE
WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE
AUDIO_SUFFIXES = frozenset({".wav", ".flac"})
REFRACTORY_DEFAULT = 2.0


class WindowScorer(Protocol):
    def score(self, window: np.ndarray) -> float: ...

    def reset(self) -> None: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def enumerate_audio_inputs(paths: Sequence[Path]) -> list[Path]:
    """Deterministically collect ``.wav`` / ``.flac`` under repeatable ``--input`` paths."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            if path.suffix.lower() in AUDIO_SUFFIXES:
                found.append(path.resolve())
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in AUDIO_SUFFIXES:
                    found.append(child.resolve())
            continue
        raise FileNotFoundError(f"input not found: {path}")
    unique = sorted(set(found), key=lambda p: str(p).casefold())
    if not unique:
        raise ValueError("no audio files found in --input paths")
    return unique


def validate_gain_db(gain_db: float) -> None:
    if not math.isfinite(gain_db):
        raise ValueError("gain_db must be a finite float")


def gain_linear_from_db(gain_db: float) -> float:
    """Match satellite ``gain_scalar_from_db`` (bounded linear gain from dB)."""
    validate_gain_db(gain_db)
    satellite_root = _REPO_ROOT / "satellite"
    if str(satellite_root) not in sys.path:
        sys.path.insert(0, str(satellite_root))
    from sayso.wake.capture import gain_scalar_from_db

    return gain_scalar_from_db(gain_db)


def apply_gain_before_quantize(
    mono: np.ndarray,
    gain_db: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply fixed gain after resample; clip to [-1, 1] before int16 (Pi capture path)."""
    gain_linear = gain_linear_from_db(gain_db)
    if mono.size == 0:
        amp: dict[str, Any] = {
            "gain_db": float(gain_db),
            "gain_linear": gain_linear,
            "pre_gain_peak": 0.0,
            "pre_gain_rms": 0.0,
            "post_gain_clip_count": 0,
            "post_gain_clip_fraction": 0.0,
        }
        return mono, amp
    mono64 = mono.astype(np.float64, copy=False)
    pre_peak = float(np.max(np.abs(mono64)))
    pre_rms = float(np.sqrt(np.mean(mono64**2)))
    scaled = mono64 * gain_linear
    clip_mask = np.abs(scaled) > 1.0
    clip_count = int(np.count_nonzero(clip_mask))
    clip_fraction = float(clip_count) / float(mono64.size)
    clipped = np.clip(scaled, -1.0, 1.0)
    amp = {
        "gain_db": float(gain_db),
        "gain_linear": gain_linear,
        "pre_gain_peak": pre_peak,
        "pre_gain_rms": pre_rms,
        "post_gain_clip_count": clip_count,
        "post_gain_clip_fraction": clip_fraction,
    }
    return clipped, amp


def _resample_mono(mono: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    from scipy.signal import resample_poly

    if source_rate == target_rate:
        return mono
    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return resample_poly(mono, up, down).astype(np.float64, copy=False)


def load_mono_16k(path: Path, *, gain_db: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Read audio with soundfile, downmix to mono, resample accurately to 16 kHz."""
    import soundfile as sf

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"audio file not found: {resolved}")
    data, rate = sf.read(resolved, always_2d=True, dtype="float64")
    channels = int(data.shape[1])
    mono = data.mean(axis=1)
    source_rate = int(rate)
    if source_rate <= 0:
        raise ValueError(f"invalid sample rate {source_rate} in {resolved}")
    mono = _resample_mono(mono, source_rate, SAMPLE_RATE)
    mono, amp = apply_gain_before_quantize(mono, gain_db)
    pcm = np.round(mono * 32767.0).astype(np.int16)
    meta = {
        "source_sample_rate": source_rate,
        "source_channels": channels,
        "sample_count": int(pcm.size),
        "duration_seconds": float(pcm.size) / SAMPLE_RATE,
        **amp,
    }
    return pcm, meta


def plan_audio_windows(n_samples: int) -> tuple[int, int, int, int]:
    """Return (window_count, used_samples, dropped_trailing_samples, last_start_sample)."""
    if n_samples < WINDOW_SAMPLES:
        return 0, 0, n_samples, 0
    window_count = (n_samples - WINDOW_SAMPLES) // HOP_SAMPLES + 1
    last_start = (window_count - 1) * HOP_SAMPLES
    used_samples = last_start + WINDOW_SAMPLES
    dropped = n_samples - used_samples
    return window_count, used_samples, dropped, last_start


def window_start_samples(window_index: int) -> int:
    return window_index * HOP_SAMPLES


def window_start_seconds(window_index: int) -> float:
    return float(window_index * HOP_SECONDS)


def window_end_seconds(window_index: int) -> float:
    return float(window_index * HOP_SECONDS + WINDOW_SECONDS)


def raw_crossing_indices(scores: np.ndarray, threshold: float) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.asarray(scores, dtype=np.float64) >= threshold)]


def _flush_audio_crossing(
    start_w: int,
    end_w: int,
    peak_w: int,
    peak: float,
    *,
    threshold: float,
    file_path: Path,
) -> dict[str, Any]:
    start_sample = window_start_samples(start_w)
    peak_sample = window_start_samples(peak_w)
    end_sample = window_start_samples(end_w) + WINDOW_SAMPLES
    return {
        "file": str(file_path.resolve()),
        "start_window": start_w,
        "peak_window": peak_w,
        "end_window": end_w,
        "start_sample": start_sample,
        "peak_sample": peak_sample,
        "end_sample": end_sample,
        "start_seconds": window_start_seconds(start_w),
        "peak_seconds": window_start_seconds(peak_w),
        "end_seconds": window_end_seconds(end_w),
        "peak_score": peak,
        "threshold": threshold,
    }


def cluster_crossings(
    scores: np.ndarray,
    *,
    threshold: float,
    refractory_seconds: float,
    file_path: Path,
) -> list[dict[str, Any]]:
    """Cluster adjacent threshold crossings; merge within refractory on peak time."""
    if refractory_seconds < 0:
        raise ValueError("refractory_seconds must be non-negative")
    raw = raw_crossing_indices(scores, threshold)
    if not raw:
        return []

    runs: list[tuple[int, int, int, float]] = []
    group_start = raw[0]
    group_end = raw[0]
    peak_window = raw[0]
    peak_score = float(scores[peak_window])
    for w in raw[1:]:
        if w == group_end + 1:
            group_end = w
            if float(scores[w]) > peak_score:
                peak_window = w
                peak_score = float(scores[w])
            continue
        runs.append((group_start, group_end, peak_window, peak_score))
        group_start = group_end = peak_window = w
        peak_score = float(scores[w])
    runs.append((group_start, group_end, peak_window, peak_score))

    if refractory_seconds > 0 and len(runs) > 1:
        merged: list[tuple[int, int, int, float]] = [runs[0]]
        for start_w, end_w, peak_w, peak in runs[1:]:
            prev_start, prev_end, prev_peak_w, prev_peak = merged[-1]
            gap = window_start_seconds(start_w) - window_start_seconds(prev_peak_w)
            if gap < refractory_seconds:
                merged_start = prev_start
                merged_end = max(prev_end, end_w)
                if peak > prev_peak:
                    new_peak_w, new_peak = peak_w, peak
                else:
                    new_peak_w, new_peak = prev_peak_w, prev_peak
                merged[-1] = (merged_start, merged_end, new_peak_w, new_peak)
            else:
                merged.append((start_w, end_w, peak_w, peak))
        runs = merged

    return [
        _flush_audio_crossing(start_w, end_w, peak_w, peak, threshold=threshold, file_path=file_path)
        for start_w, end_w, peak_w, peak in runs
    ]


def false_activations_per_hour(*, activation_count: int, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return float("inf")
    hours = duration_seconds / 3600.0
    return activation_count / hours if hours > 0 else float("inf")


def score_summary_stats(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    if scores.size == 0:
        percentiles = {
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p99_9": 0.0,
            "max": 0.0,
        }
        threshold_counts = {
            "ge_0_1": 0,
            "ge_0_2": 0,
            "ge_0_3": 0,
            "ge_0_4": 0,
            "ge_threshold": 0,
            "windows": 0,
        }
        return {"percentiles": percentiles, "threshold_counts": threshold_counts}
    arr = np.asarray(scores, dtype=np.float64)
    percentiles = {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p99_9": float(np.percentile(arr, 99.9)),
        "max": float(np.max(arr)),
    }
    threshold_counts = {
        "ge_0_1": int(np.count_nonzero(arr >= 0.1)),
        "ge_0_2": int(np.count_nonzero(arr >= 0.2)),
        "ge_0_3": int(np.count_nonzero(arr >= 0.3)),
        "ge_0_4": int(np.count_nonzero(arr >= 0.4)),
        "ge_threshold": int(np.count_nonzero(arr >= threshold)),
        "windows": int(arr.size),
    }
    return {"percentiles": percentiles, "threshold_counts": threshold_counts}


def append_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(path)


def score_audio_in_chunks(
    audio: np.ndarray,
    scorer: WindowScorer,
    *,
    chunk_size: int,
    file_path: Path,
) -> np.ndarray:
    """Score every complete window; never retain more than ``chunk_size`` windows at once."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n_samples = int(audio.size)
    window_count, _, _, _ = plan_audio_windows(n_samples)
    scores = np.empty(window_count, dtype=np.float64)
    if window_count == 0:
        return scores
    for chunk_start in range(0, window_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, window_count)
        for wi in range(chunk_start, chunk_end):
            start = window_start_samples(wi)
            window = audio[start : start + WINDOW_SAMPLES]
            if window.size != WINDOW_SAMPLES:
                raise ValueError(
                    f"internal window size mismatch for {file_path}: "
                    f"index={wi} size={window.size}"
                )
            scores[wi] = scorer.score(window)
    return scores


@dataclass(frozen=True)
class ReplayConfig:
    input_paths: tuple[Path, ...]
    model_path: Path | None
    scores_path: Path
    events_path: Path
    summary_path: Path
    threshold: float
    refractory_seconds: float
    chunk_size: int
    gain_db: float = 0.0


@dataclass(frozen=True)
class ReplayResult:
    scores: np.ndarray
    events: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def build_production_scorer(model_path: Path) -> WindowScorer:
    satellite_root = _REPO_ROOT / "satellite"
    if str(satellite_root) not in sys.path:
        sys.path.insert(0, str(satellite_root))
    from livekit.wakeword import WakeWordModel

    from sayso.wake.streaming import CachedEmbeddingScorer, single_threaded_ort

    key = model_path.stem

    class _ProductionScorer:
        def __init__(self) -> None:
            with single_threaded_ort():
                self._model = WakeWordModel(models=[str(model_path)])
                self._scorer = CachedEmbeddingScorer(self._model)
            self._key = key

        def score(self, window: np.ndarray) -> float:
            return float(self._scorer.score(window)[self._key])

        def reset(self) -> None:
            self._scorer.reset()

    return _ProductionScorer()


def _validate_output_paths(config: ReplayConfig, sources: Sequence[Path]) -> None:
    model = config.model_path.resolve() if config.model_path is not None else None
    source_set = {p.resolve() for p in sources}
    for label, path in (
        ("scores", config.scores_path),
        ("events", config.events_path),
        ("summary", config.summary_path),
    ):
        resolved = path.resolve()
        if resolved in source_set:
            raise ValueError(f"{label} path must not overwrite input audio: {path}")
        if model is not None and resolved == model:
            raise ValueError(f"{label} path must not overwrite model: {path}")
        if resolved.parent == Path("/"):
            raise ValueError(f"{label} path must include a directory: {path}")


def build_summary(
    *,
    config: ReplayConfig,
    sources: Sequence[Path],
    file_records: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    raw_crossing_count: int,
    scores_sha256: str,
    events_sha256: str,
) -> dict[str, Any]:
    duration_seconds = float(sum(float(r["duration_seconds"]) for r in file_records))
    window_count = int(scores.size)
    stats = score_summary_stats(scores, config.threshold)
    payload: dict[str, Any] = {
        "version": PROVENANCE_VERSION,
        "created_at": _utc_now(),
        "inputs": [str(p.resolve()) for p in sources],
        "files": list(file_records),
        "windows": {
            "count": window_count,
            "window_samples": WINDOW_SAMPLES,
            "hop_samples": HOP_SAMPLES,
            "sample_rate": SAMPLE_RATE,
            "hop_seconds": HOP_SECONDS,
            "window_seconds": WINDOW_SECONDS,
        },
        "outputs": {
            "scores_path": str(config.scores_path.resolve()),
            "scores_sha256": scores_sha256,
            "events_path": str(config.events_path.resolve()),
            "events_sha256": events_sha256,
            "summary_path": str(config.summary_path.resolve()),
        },
        "threshold": config.threshold,
        "refractory_seconds": config.refractory_seconds,
        "duration_seconds": duration_seconds,
        "total_hours": duration_seconds / 3600.0,
        "raw_crossing_count": raw_crossing_count,
        "clustered_event_count": len(events),
        "false_activations_per_hour": false_activations_per_hour(
            activation_count=len(events),
            duration_seconds=duration_seconds,
        ),
        "score_percentiles": stats["percentiles"],
        "threshold_counts": stats["threshold_counts"],
        "chunk_size": config.chunk_size,
        "gain_db": float(config.gain_db),
        "gain_linear": gain_linear_from_db(config.gain_db),
    }
    if config.model_path is not None:
        payload["model"] = {
            "path": str(config.model_path.resolve()),
            "sha256": mine._sha256_file(config.model_path),
        }
    return payload


def run_replay(
    config: ReplayConfig,
    *,
    window_scorer: WindowScorer | None = None,
) -> ReplayResult:
    sources = enumerate_audio_inputs(config.input_paths)
    _validate_output_paths(config, sources)

    scorer = window_scorer
    if scorer is None:
        if config.model_path is None:
            raise ValueError("either window_scorer or model_path is required")
        scorer = build_production_scorer(config.model_path)

    all_scores: list[float] = []
    all_events: list[dict[str, Any]] = []
    file_records: list[dict[str, Any]] = []
    raw_crossing_count = 0
    score_offset = 0

    for source in sources:
        scorer.reset()
        before = source.read_bytes()
        pcm, meta = load_mono_16k(source, gain_db=config.gain_db)
        after = source.read_bytes()
        if before != after:
            raise RuntimeError(f"input audio mutated during read: {source}")

        window_count, used_samples, dropped, _ = plan_audio_windows(pcm.size)
        file_scores = score_audio_in_chunks(
            pcm,
            scorer,
            chunk_size=config.chunk_size,
            file_path=source,
        )
        crossings = raw_crossing_indices(file_scores, config.threshold)
        raw_crossing_count += len(crossings)
        events = cluster_crossings(
            file_scores,
            threshold=config.threshold,
            refractory_seconds=config.refractory_seconds,
            file_path=source,
        )
        all_events.extend(events)
        all_scores.extend(file_scores.tolist())

        file_records.append(
            {
                "path": str(source.resolve()),
                "sha256": mine._sha256_file(source),
                "source_sample_rate": meta["source_sample_rate"],
                "source_channels": meta["source_channels"],
                "sample_count": meta["sample_count"],
                "duration_seconds": meta["duration_seconds"],
                "used_samples": used_samples,
                "dropped_trailing_samples": dropped,
                "window_count": window_count,
                "score_offset": score_offset,
                "score_count": int(file_scores.size),
                "gain_db": meta["gain_db"],
                "gain_linear": meta["gain_linear"],
                "pre_gain_peak": meta["pre_gain_peak"],
                "pre_gain_rms": meta["pre_gain_rms"],
                "post_gain_clip_count": meta["post_gain_clip_count"],
                "post_gain_clip_fraction": meta["post_gain_clip_fraction"],
            }
        )
        score_offset += int(file_scores.size)

    scores_arr = np.asarray(all_scores, dtype=np.float64)
    mine.atomic_save_npy(config.scores_path, scores_arr)
    scores_sha256 = mine._sha256_file(config.scores_path)
    append_jsonl(config.events_path, all_events)
    events_sha256 = mine._sha256_file(config.events_path)
    summary = build_summary(
        config=config,
        sources=sources,
        file_records=file_records,
        scores=scores_arr,
        events=all_events,
        raw_crossing_count=raw_crossing_count,
        scores_sha256=scores_sha256,
        events_sha256=events_sha256,
    )
    mine.atomic_write_json(config.summary_path, summary)
    return ReplayResult(scores=scores_arr, events=tuple(all_events), summary=summary)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        dest="inputs",
        help="Audio file or directory (.wav / .flac); repeat for multiple roots",
    )
    ap.add_argument("--model", type=Path, required=True, help="LiveKit wake ONNX model")
    ap.add_argument("--scores", type=Path, required=True, help="Destination .npy for all window scores")
    ap.add_argument("--events", type=Path, required=True, help="Destination JSONL for clustered events")
    ap.add_argument("--summary", type=Path, required=True, help="Destination JSON summary and provenance")
    ap.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    ap.add_argument(
        "--refractory",
        type=float,
        default=REFRACTORY_DEFAULT,
        help="Seconds between clustered activation events",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Max windows scored per in-memory chunk within one file",
    )
    ap.add_argument(
        "--gain-db",
        type=float,
        default=0.0,
        help="Fixed mic gain in dB (matches satellite audio.mic_gain_db; default 0)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.threshold < 0 or args.threshold > 1:
        print("threshold must be between 0 and 1", file=sys.stderr)
        return 2
    if args.refractory < 0:
        print("refractory must be non-negative", file=sys.stderr)
        return 2
    if args.chunk_size <= 0:
        print("chunk-size must be positive", file=sys.stderr)
        return 2
    try:
        validate_gain_db(float(args.gain_db))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1

    config = ReplayConfig(
        input_paths=tuple(args.inputs),
        model_path=args.model,
        scores_path=args.scores,
        events_path=args.events,
        summary_path=args.summary,
        threshold=float(args.threshold),
        refractory_seconds=float(args.refractory),
        chunk_size=int(args.chunk_size),
        gain_db=float(args.gain_db),
    )
    try:
        result = run_replay(config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    summary = result.summary
    print(
        f"scored {summary['windows']['count']} windows "
        f"from {len(summary['files'])} file(s) "
        f"({summary['duration_seconds']:.3f} s)"
    )
    print(
        f"events raw={summary['raw_crossing_count']} "
        f"clustered={summary['clustered_event_count']} "
        f"fpph={summary['false_activations_per_hour']:.6f}"
    )
    print(f"scores   {config.scores_path}")
    print(f"events   {config.events_path}")
    print(f"summary  {config.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
