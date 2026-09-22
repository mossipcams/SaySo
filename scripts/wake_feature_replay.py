#!/usr/bin/env python3
"""Replay long LiveKit validation feature streams and log wake scores.

Memory-maps ``validation_set_features.npy`` in upstream shape ``(N, 96)``,
reshapes sequential non-overlapping windows to ``(W, 16, 96)``, scores an ONNX
classifier in bounded batches, and writes scores, clustered threshold events,
and a provenance summary suitable for multi-hour train-host replays.

    python scripts/wake_feature_replay.py \\
        --source /path/validation_set_features.npy \\
        --model /path/sayso.onnx \\
        --scores /runs/replay/scores.npy \\
        --events /runs/replay/events.jsonl \\
        --summary /runs/replay/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_hard_negative_mine as mine  # noqa: E402

PROVENANCE_VERSION = 1
EMBEDDING_DIM = 96
FRAMES_PER_WINDOW = 16
FRAME_SECONDS = 0.08
WINDOW_SECONDS = FRAMES_PER_WINDOW * FRAME_SECONDS  # 1.28 s

ScoreBatchFn = Callable[[np.ndarray], np.ndarray]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def open_feature_stream(path: Path) -> np.ndarray:
    """Memory-map upstream LiveKit features; require shape (N, 96)."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"feature file not found: {resolved}")
    features = np.load(resolved, mmap_mode="r")
    validate_feature_stream_shape(features)
    return features


def validate_feature_stream_shape(features: np.ndarray) -> int:
    if features.ndim != 2:
        raise ValueError(
            f"expected LiveKit feature stream with shape (N, 96); got ndim={features.ndim}"
        )
    if int(features.shape[1]) != EMBEDDING_DIM:
        raise ValueError(
            f"expected LiveKit feature stream with shape (N, 96); got {tuple(features.shape)}"
        )
    return int(features.shape[0])


def plan_stream_windows(n_rows: int) -> tuple[int, int, int, int]:
    """Return (window_count, used_rows, dropped_trailing_rows, last_start_row)."""
    if n_rows < FRAMES_PER_WINDOW:
        return 0, 0, n_rows, 0
    window_count = n_rows // FRAMES_PER_WINDOW
    used_rows = window_count * FRAMES_PER_WINDOW
    dropped = n_rows - used_rows
    last_start = (window_count - 1) * FRAMES_PER_WINDOW
    return window_count, used_rows, dropped, last_start


def window_starts(n_rows: int) -> np.ndarray:
    """Non-overlapping start row indices (0, 16, 32, ...) for every replay window."""
    window_count, _, _, _ = plan_stream_windows(n_rows)
    if window_count == 0:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, window_count * FRAMES_PER_WINDOW, FRAMES_PER_WINDOW, dtype=np.int64)


def materialize_windows(
    stream: np.ndarray,
    starts: np.ndarray,
) -> np.ndarray:
    """Build (W, 16, 96) windows from mmap rows without loading the full bank."""
    if starts.size == 0:
        return np.empty((0, FRAMES_PER_WINDOW, EMBEDDING_DIM), dtype=np.float32)
    batch = np.empty((starts.shape[0], FRAMES_PER_WINDOW, EMBEDDING_DIM), dtype=np.float32)
    for i, start in enumerate(starts):
        batch[i] = np.array(stream[start : start + FRAMES_PER_WINDOW], dtype=np.float32, copy=True)
    return batch


def score_windows_in_chunks(
    stream: np.ndarray,
    scorer: ScoreBatchFn,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Score every non-overlapping window via bounded batches."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n_rows = validate_feature_stream_shape(stream)
    starts = window_starts(n_rows)
    scores = np.empty(starts.shape[0], dtype=np.float64)
    for offset in range(0, starts.shape[0], chunk_size):
        chunk_starts = starts[offset : offset + chunk_size]
        batch = materialize_windows(stream, chunk_starts)
        chunk_scores = np.asarray(scorer(batch), dtype=np.float64).reshape(-1)
        if chunk_scores.shape[0] != batch.shape[0]:
            raise ValueError(
                f"scorer returned {chunk_scores.shape[0]} scores for batch of {batch.shape[0]}"
            )
        scores[offset : offset + batch.shape[0]] = chunk_scores
    return scores


def window_start_seconds(window_index: int) -> float:
    return float(window_index * WINDOW_SECONDS)


def window_end_seconds(window_index: int) -> float:
    return float((window_index + 1) * WINDOW_SECONDS)


def raw_crossing_indices(scores: np.ndarray, threshold: float) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.asarray(scores, dtype=np.float64) >= threshold)]


def _flush_crossing_group(
    start_w: int,
    end_w: int,
    peak_w: int,
    peak: float,
    *,
    threshold: float,
) -> dict[str, Any]:
    row_start = start_w * FRAMES_PER_WINDOW
    row_end = (end_w + 1) * FRAMES_PER_WINDOW
    return {
        "start_window": start_w,
        "peak_window": peak_w,
        "end_window": end_w,
        "source_row_start": row_start,
        "source_row_end": row_end,
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
) -> list[dict[str, Any]]:
    """Cluster adjacent crossings, then merge nearby groups within refractory."""
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
        merged_runs: list[tuple[int, int, int, float]] = [runs[0]]
        for start_w, end_w, peak_w, peak in runs[1:]:
            prev_start, prev_end, prev_peak_w, prev_peak = merged_runs[-1]
            gap = window_start_seconds(start_w) - window_start_seconds(prev_peak_w)
            if gap < refractory_seconds:
                merged_start = prev_start
                merged_end = max(prev_end, end_w)
                if peak > prev_peak:
                    new_peak_w, new_peak = peak_w, peak
                else:
                    new_peak_w, new_peak = prev_peak_w, prev_peak
                merged_runs[-1] = (merged_start, merged_end, new_peak_w, new_peak)
            else:
                merged_runs.append((start_w, end_w, peak_w, peak))
        runs = merged_runs

    return [
        _flush_crossing_group(start_w, end_w, peak_w, peak, threshold=threshold)
        for start_w, end_w, peak_w, peak in runs
    ]


def false_activations_per_hour(*, activation_count: int, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return float("inf")
    hours = duration_seconds / 3600.0
    return activation_count / hours if hours > 0 else float("inf")


def score_summary_stats(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    if scores.size == 0:
        return {
            "percentiles": {
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "p99_9": 0.0,
                "max": 0.0,
            },
            "threshold_counts": {
                "ge_threshold": 0,
                "windows": 0,
            },
        }
    arr = np.asarray(scores, dtype=np.float64)
    percentiles = {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p99_9": float(np.percentile(arr, 99.9)),
        "max": float(np.max(arr)),
    }
    return {
        "percentiles": percentiles,
        "threshold_counts": {
            "ge_threshold": int(np.count_nonzero(arr >= threshold)),
            "windows": int(arr.size),
        },
    }


def append_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(path)


@dataclass(frozen=True)
class ReplayConfig:
    source_path: Path
    model_path: Path | None
    scores_path: Path
    events_path: Path
    summary_path: Path
    threshold: float
    refractory_seconds: float
    chunk_size: int


@dataclass(frozen=True)
class ReplayResult:
    scores: np.ndarray
    events: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def build_summary(
    *,
    config: ReplayConfig,
    source: np.ndarray,
    scores: np.ndarray,
    events: Sequence[Mapping[str, Any]],
    raw_crossing_count: int,
    scores_sha256: str,
    events_sha256: str,
) -> dict[str, Any]:
    n_rows = int(source.shape[0])
    window_count, used_rows, dropped_rows, _ = plan_stream_windows(n_rows)
    duration_seconds = used_rows * FRAME_SECONDS
    stats = score_summary_stats(scores, config.threshold)
    payload: dict[str, Any] = {
        "version": PROVENANCE_VERSION,
        "created_at": _utc_now(),
        "source": {
            "path": str(config.source_path.resolve()),
            "sha256": mine._sha256_file(config.source_path),
            "shape": [n_rows, EMBEDDING_DIM],
            "dtype": str(source.dtype),
            "used_rows": used_rows,
            "dropped_trailing_rows": dropped_rows,
        },
        "windows": {
            "count": window_count,
            "shape": [window_count, FRAMES_PER_WINDOW, EMBEDDING_DIM],
            "frame_seconds": FRAME_SECONDS,
            "stride_frames": FRAMES_PER_WINDOW,
            "stride_seconds": WINDOW_SECONDS,
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
    }
    if config.model_path is not None:
        payload["model"] = {
            "path": str(config.model_path.resolve()),
            "sha256": mine._sha256_file(config.model_path),
        }
    return payload


def _validate_output_paths(config: ReplayConfig) -> None:
    source = config.source_path.resolve()
    model = config.model_path.resolve() if config.model_path is not None else None
    for label, path in (
        ("scores", config.scores_path),
        ("events", config.events_path),
        ("summary", config.summary_path),
    ):
        resolved = path.resolve()
        if resolved == source:
            raise ValueError(f"{label} path must not overwrite source features: {path}")
        if model is not None and resolved == model:
            raise ValueError(f"{label} path must not overwrite model: {path}")
        if resolved.parent == Path("/"):
            raise ValueError(f"{label} path must include a directory: {path}")


def run_replay(
    config: ReplayConfig,
    *,
    scorer: ScoreBatchFn | None = None,
) -> ReplayResult:
    _validate_output_paths(config)
    stream = open_feature_stream(config.source_path)
    if scorer is None:
        if config.model_path is None:
            raise ValueError("either scorer or model_path is required")
        scorer = mine.build_onnx_batch_scorer(config.model_path)

    scores = score_windows_in_chunks(stream, scorer, chunk_size=config.chunk_size)
    raw_crossings = raw_crossing_indices(scores, config.threshold)
    events = cluster_crossings(
        scores,
        threshold=config.threshold,
        refractory_seconds=config.refractory_seconds,
    )

    mine.atomic_save_npy(config.scores_path, scores.astype(np.float64, copy=False))
    scores_sha256 = mine._sha256_file(config.scores_path)
    append_jsonl(config.events_path, events)
    events_sha256 = mine._sha256_file(config.events_path)
    summary = build_summary(
        config=config,
        source=stream,
        scores=scores,
        events=events,
        raw_crossing_count=len(raw_crossings),
        scores_sha256=scores_sha256,
        events_sha256=events_sha256,
    )
    mine.atomic_write_json(config.summary_path, summary)
    return ReplayResult(scores=scores, events=tuple(events), summary=summary)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="LiveKit validation feature stream .npy (N, 96)",
    )
    ap.add_argument("--model", type=Path, required=True, help="Wake classifier ONNX model")
    ap.add_argument("--scores", type=Path, required=True, help="Destination .npy for per-window scores")
    ap.add_argument("--events", type=Path, required=True, help="Destination JSONL for clustered threshold events")
    ap.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Destination JSON summary and provenance",
    )
    ap.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    ap.add_argument(
        "--refractory",
        type=float,
        default=2.0,
        help="Seconds between clustered activation events",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Window batch size while scanning the mmap source",
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
    if not args.source.is_file():
        print(f"source not found: {args.source}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1

    config = ReplayConfig(
        source_path=args.source,
        model_path=args.model,
        scores_path=args.scores,
        events_path=args.events,
        summary_path=args.summary,
        threshold=float(args.threshold),
        refractory_seconds=float(args.refractory),
        chunk_size=int(args.chunk_size),
    )
    try:
        result = run_replay(config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    summary = result.summary
    print(
        f"scored {summary['windows']['count']} windows "
        f"from N={summary['source']['shape'][0]} "
        f"(dropped {summary['source']['dropped_trailing_rows']} trailing rows)"
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
