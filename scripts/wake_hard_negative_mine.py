#!/usr/bin/env python3
"""Mine hard-negative ACAV speech-embedding features for wake training.

Scores a large (N, 16, 96) feature bank with a wake classifier ONNX model in
bounded chunks (memory-mapped source, no full-RAM load), then deterministically
selects top-K rows with a diversity guard against adjacent indices.

    python scripts/wake_hard_negative_mine.py \\
        --features /data/ACAV100M/speech_embeddings.npy \\
        --model /path/to/sayso.onnx \\
        --output-features /runs/hard_neg.npy \\
        --provenance /runs/hard_neg.provenance.json \\
        --top-k 512 --chunk-size 256 --diversity-gap 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

PROVENANCE_VERSION = 1
ACAV_ROW_SHAPE = (16, 96)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_acav_features(path: Path) -> np.ndarray:
    """Memory-map an ACAV feature array; require shape (N, 16, 96)."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"feature file not found: {resolved}")
    features = np.load(resolved, mmap_mode="r")
    validate_acav_shape(features)
    return features


def validate_acav_shape(features: np.ndarray) -> int:
    if features.ndim != 3:
        raise ValueError(
            f"expected ACAV features with shape (N, 16, 96); got ndim={features.ndim}"
        )
    if tuple(features.shape[1:]) != ACAV_ROW_SHAPE:
        raise ValueError(
            f"expected ACAV features with shape (N, 16, 96); got {tuple(features.shape)}"
        )
    return int(features.shape[0])


ScoreBatchFn = Callable[[np.ndarray], np.ndarray]


def score_features_in_chunks(
    features: np.ndarray,
    scorer: ScoreBatchFn,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Score every row via ``scorer(batch)`` without loading all rows into RAM."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n = validate_acav_shape(features)
    scores = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch = np.array(features[start:end], dtype=np.float32, copy=True)
        chunk_scores = np.asarray(scorer(batch), dtype=np.float64).reshape(-1)
        if chunk_scores.shape[0] != end - start:
            raise ValueError(
                f"scorer returned {chunk_scores.shape[0]} scores for batch of {end - start}"
            )
        scores[start:end] = chunk_scores
    return scores


def _top_lex_order_indices(scores: np.ndarray, m: int) -> np.ndarray:
    """First ``m`` indices in deterministic order: score descending, index ascending."""
    n = int(scores.shape[0])
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    if m >= n:
        idx = np.arange(n, dtype=np.int64)
        return idx[np.lexsort((idx, -scores))]

    neg = -scores
    kth_neg = np.partition(neg, m - 1)[m - 1]
    definitely_in = neg < kth_neg
    tied = neg == kth_neg
    n_def = int(np.count_nonzero(definitely_in))
    need_from_tied = m - n_def
    if need_from_tied > 0:
        tied_idx = np.flatnonzero(tied)
        take_tied = np.sort(tied_idx)[:need_from_tied]
        candidates = np.concatenate((np.flatnonzero(definitely_in), take_tied))
    else:
        candidates = np.flatnonzero(definitely_in)
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def _apply_diversity_gap(
    order: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
    diversity_gap: int,
) -> list[tuple[int, float]]:
    selected: list[tuple[int, float]] = []
    picked: list[int] = []
    for idx in order:
        idx_int = int(idx)
        if len(selected) >= top_k:
            break
        if diversity_gap > 0 and any(abs(idx_int - prior) <= diversity_gap for prior in picked):
            continue
        selected.append((idx_int, float(scores[idx_int])))
        picked.append(idx_int)
    return selected


def select_diverse_top_k(
    scores: np.ndarray,
    *,
    top_k: int,
    diversity_gap: int,
) -> list[tuple[int, float]]:
    """Deterministic top-K by score with tie-break on index; skip near neighbors."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if diversity_gap < 0:
        raise ValueError("diversity_gap must be non-negative")
    n = int(scores.shape[0])
    if top_k == 0 or n == 0:
        return []

    scores_f = np.asarray(scores, dtype=np.float32 if scores.dtype == np.float32 else np.float64)
    # ponytail: greedy needs at most top_k * (2*gap+1) candidates in score order.
    candidate_cap = min(n, top_k * (2 * diversity_gap + 1))
    order = _top_lex_order_indices(scores_f, candidate_cap)
    return _apply_diversity_gap(
        order,
        scores_f,
        top_k=top_k,
        diversity_gap=diversity_gap,
    )


def build_onnx_batch_scorer(model_path: Path) -> ScoreBatchFn:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - train host installs onnxruntime
        raise RuntimeError("onnxruntime is required for --model scoring") from exc

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    def score_batch(batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 3 or tuple(batch.shape[1:]) != ACAV_ROW_SHAPE:
            raise ValueError(
                f"ONNX scorer expected batch shape (B, 16, 96); got {batch.shape}"
            )
        feed = np.ascontiguousarray(batch, dtype=np.float32)
        outputs = session.run(None, {input_name: feed})
        raw = np.asarray(outputs[0], dtype=np.float32)
        if raw.ndim == 0:
            raise ValueError("ONNX classifier returned a scalar for a batch input")
        flat = raw.reshape(batch.shape[0], -1)
        return flat[:, 0].astype(np.float64, copy=False)

    return score_batch


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as fh:
        np.save(fh, array, allow_pickle=False)
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class MineConfig:
    features_path: Path
    output_features_path: Path
    provenance_path: Path
    model_path: Path | None
    top_k: int
    chunk_size: int
    diversity_gap: int


@dataclass(frozen=True)
class MineResult:
    selected_indices: tuple[int, ...]
    selected_scores: tuple[float, ...]
    output_shape: tuple[int, ...]
    provenance: dict[str, Any]


def build_provenance(
    *,
    config: MineConfig,
    source: np.ndarray,
    selected: Sequence[tuple[int, float]],
    output_array: np.ndarray,
    output_features_sha256: str,
) -> dict[str, Any]:
    indices = [idx for idx, _ in selected]
    score_values = [score for _, score in selected]
    payload: dict[str, Any] = {
        "version": PROVENANCE_VERSION,
        "created_at": _utc_now(),
        "source": {
            "path": str(config.features_path.resolve()),
            "sha256": _sha256_file(config.features_path),
            "shape": list(source.shape),
            "dtype": str(source.dtype),
        },
        "output": {
            "features_path": str(config.output_features_path.resolve()),
            "provenance_path": str(config.provenance_path.resolve()),
            "shape": list(output_array.shape),
            "dtype": str(output_array.dtype),
            "sha256": output_features_sha256,
        },
        "selection": {
            "top_k": config.top_k,
            "chunk_size": config.chunk_size,
            "diversity_gap": config.diversity_gap,
            "selected_count": len(indices),
            "selected_indices": indices,
            "selected_scores": score_values,
        },
    }
    if config.model_path is not None:
        payload["model"] = {
            "path": str(config.model_path.resolve()),
            "sha256": _sha256_file(config.model_path),
        }
    return payload


def run_mine(
    config: MineConfig,
    *,
    scorer: ScoreBatchFn | None = None,
) -> MineResult:
    features = open_acav_features(config.features_path)
    if scorer is None:
        if config.model_path is None:
            raise ValueError("either scorer or model_path is required")
        scorer = build_onnx_batch_scorer(config.model_path)

    scores = score_features_in_chunks(
        features,
        scorer,
        chunk_size=config.chunk_size,
    )
    selected = select_diverse_top_k(
        scores,
        top_k=config.top_k,
        diversity_gap=config.diversity_gap,
    )
    if selected:
        indices = [idx for idx, _ in selected]
        output = np.asarray(features[indices], dtype=np.float32)
    else:
        output = np.empty((0, *ACAV_ROW_SHAPE), dtype=np.float32)

    atomic_save_npy(config.output_features_path, output)
    output_sha256 = _sha256_file(config.output_features_path)
    provenance = build_provenance(
        config=config,
        source=features,
        selected=selected,
        output_array=output,
        output_features_sha256=output_sha256,
    )
    atomic_write_json(config.provenance_path, provenance)

    return MineResult(
        selected_indices=tuple(idx for idx, _ in selected),
        selected_scores=tuple(score for _, score in selected),
        output_shape=tuple(output.shape),
        provenance=provenance,
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--features", type=Path, required=True, help="Source ACAV .npy (N, 16, 96)")
    ap.add_argument("--model", type=Path, required=True, help="Wake classifier ONNX model")
    ap.add_argument(
        "--output-features",
        type=Path,
        required=True,
        help="Destination .npy for selected (K, 16, 96) rows",
    )
    ap.add_argument(
        "--provenance",
        type=Path,
        required=True,
        help="Destination JSON provenance for this mining run",
    )
    ap.add_argument("--top-k", type=int, default=512, help="Maximum rows to select")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Classifier batch size while scanning the mmap source",
    )
    ap.add_argument(
        "--diversity-gap",
        type=int,
        default=1,
        help="Skip candidates within this many indices of an already selected row",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.top_k < 0:
        print("top-k must be non-negative", file=sys.stderr)
        return 2
    if args.chunk_size <= 0:
        print("chunk-size must be positive", file=sys.stderr)
        return 2
    if args.diversity_gap < 0:
        print("diversity-gap must be non-negative", file=sys.stderr)
        return 2
    if not args.features.is_file():
        print(f"features not found: {args.features}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1

    config = MineConfig(
        features_path=args.features,
        output_features_path=args.output_features,
        provenance_path=args.provenance,
        model_path=args.model,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        diversity_gap=args.diversity_gap,
    )
    try:
        result = run_mine(config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"selected {len(result.selected_indices)} / top_k={config.top_k} "
        f"from N={result.provenance['source']['shape'][0]}"
    )
    print(f"features  {config.output_features_path}")
    print(f"manifest  {config.provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
