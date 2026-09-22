#!/usr/bin/env python3
"""Mine hard-negative (16, 96) embeddings from production wake audio replay.

Scores listed ``.wav`` / ``.flac`` inputs with the LiveKit frontend (16 kHz mono,
2 s windows, 160 ms hop, optional fixed gain), keeps windows with score ≥
``--score-floor`` (default 0.25), applies deterministic diversity-gap top-K
selection, and writes ``(K, 16, 96)`` features plus provenance.

    python scripts/wake_audio_hard_negative_mine.py \\
        --input /data/ami/train \\
        --model /path/sayso.onnx \\
        --output-features /runs/corpus_hn.npy \\
        --provenance /runs/corpus_hn.provenance.json \\
        --top-k 512 --gain-db 10
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_audio_replay as replay  # noqa: E402
from scripts import wake_hard_negative_mine as mine  # noqa: E402

PROVENANCE_VERSION = 1
DEFAULT_SCORE_FLOOR = 0.25


class EmbeddingWindowScorer(Protocol):
    def score(self, window: np.ndarray) -> float: ...

    def reset(self) -> None: ...

    def last_embedding(self) -> np.ndarray: ...


@dataclass(frozen=True)
class CandidateWindow:
    path: Path
    start_sample: int
    score: float
    embedding: np.ndarray


@dataclass(frozen=True)
class AudioMineConfig:
    input_paths: tuple[Path, ...]
    output_features_path: Path
    provenance_path: Path
    model_path: Path | None
    top_k: int
    chunk_size: int
    diversity_gap: int
    score_floor: float
    gain_db: float = 0.0


@dataclass(frozen=True)
class AudioMineResult:
    selected_indices: tuple[int, ...]
    selected_scores: tuple[float, ...]
    output_shape: tuple[int, ...]
    provenance: dict[str, Any]
    candidate_count: int


def build_production_embedding_scorer(model_path: Path) -> EmbeddingWindowScorer:
    satellite_root = _REPO_ROOT / "satellite"
    if str(satellite_root) not in sys.path:
        sys.path.insert(0, str(satellite_root))
    from livekit.wakeword import WakeWordModel

    from sayso.wake.streaming import CachedEmbeddingScorer, single_threaded_ort

    key = model_path.stem

    class _ProductionEmbeddingScorer:
        def __init__(self) -> None:
            with single_threaded_ort():
                self._model = WakeWordModel(models=[str(model_path)])
                self._scorer = CachedEmbeddingScorer(self._model)
            self._key = key

        def score(self, window: np.ndarray) -> float:
            return float(self._scorer.score(window)[self._key])

        def reset(self) -> None:
            self._scorer.reset()

        def last_embedding(self) -> np.ndarray:
            emb = self._scorer.last_embeddings
            if emb is None or emb.shape != mine.ACAV_ROW_SHAPE:
                raise RuntimeError(
                    "production scorer did not produce (16, 96) embeddings for the last window"
                )
            return np.asarray(emb, dtype=np.float32)

    return _ProductionEmbeddingScorer()


def collect_candidates_from_pcm(
    pcm: np.ndarray,
    scorer: EmbeddingWindowScorer,
    *,
    file_path: Path,
    score_floor: float,
    chunk_size: int,
) -> list[CandidateWindow]:
    """Score every complete window; retain embeddings only for score >= floor."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    n_samples = int(pcm.size)
    window_count, _, _, _ = replay.plan_audio_windows(n_samples)
    candidates: list[CandidateWindow] = []
    if window_count == 0:
        return candidates

    for chunk_start in range(0, window_count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, window_count)
        for wi in range(chunk_start, chunk_end):
            start = replay.window_start_samples(wi)
            window = pcm[start : start + replay.WINDOW_SAMPLES]
            if window.size != replay.WINDOW_SAMPLES:
                raise ValueError(
                    f"internal window size mismatch for {file_path}: "
                    f"index={wi} size={window.size}"
                )
            score = float(scorer.score(window))
            if score < score_floor:
                continue
            embedding = np.asarray(scorer.last_embedding(), dtype=np.float32, copy=True)
            if tuple(embedding.shape) != mine.ACAV_ROW_SHAPE:
                raise ValueError(
                    f"expected embedding shape {mine.ACAV_ROW_SHAPE}; got {embedding.shape}"
                )
            candidates.append(
                CandidateWindow(
                    path=file_path,
                    start_sample=start,
                    score=score,
                    embedding=embedding,
                )
            )
    return candidates


def select_from_candidates(
    candidates: Sequence[CandidateWindow],
    *,
    top_k: int,
    diversity_gap: int,
) -> list[tuple[int, CandidateWindow]]:
    if not candidates:
        return []
    scores = np.asarray([c.score for c in candidates], dtype=np.float64)
    picked = mine.select_diverse_top_k(scores, top_k=top_k, diversity_gap=diversity_gap)
    return [(idx, candidates[idx]) for idx, _ in picked]


def build_provenance(
    *,
    config: AudioMineConfig,
    sources: Sequence[Path],
    file_records: Sequence[Mapping[str, Any]],
    candidates: Sequence[CandidateWindow],
    selected: Sequence[tuple[int, CandidateWindow]],
    output_array: np.ndarray,
    output_features_sha256: str,
) -> dict[str, Any]:
    indices = [idx for idx, _ in selected]
    score_values = [cand.score for _, cand in selected]
    windows = [
        {
            "path": str(cand.path.resolve()),
            "start_sample": cand.start_sample,
            "score": cand.score,
            "candidate_index": idx,
        }
        for idx, cand in selected
    ]
    pool_count = len(candidates)
    payload: dict[str, Any] = {
        "version": PROVENANCE_VERSION,
        "created_at": replay._utc_now(),
        "source": {
            "inputs": [str(p.resolve()) for p in sources],
            "files": list(file_records),
            "shape": [pool_count, *mine.ACAV_ROW_SHAPE],
            "dtype": "float32",
            "score_floor": float(config.score_floor),
            "gain_db": float(config.gain_db),
            "gain_linear": replay.gain_linear_from_db(config.gain_db),
            "windows": {
                "window_samples": replay.WINDOW_SAMPLES,
                "hop_samples": replay.HOP_SAMPLES,
                "sample_rate": replay.SAMPLE_RATE,
                "hop_seconds": replay.HOP_SECONDS,
                "window_seconds": replay.WINDOW_SECONDS,
            },
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
            "score_floor": float(config.score_floor),
            "candidate_count": pool_count,
            "selected_count": len(indices),
            "selected_indices": indices,
            "selected_scores": score_values,
            "selected_windows": windows,
        },
    }
    if config.model_path is not None:
        payload["model"] = {
            "path": str(config.model_path.resolve()),
            "sha256": mine._sha256_file(config.model_path),
        }
    return payload


def _validate_output_paths(
    config: AudioMineConfig,
    sources: Sequence[Path],
) -> None:
    model = config.model_path.resolve() if config.model_path is not None else None
    source_set = {p.resolve() for p in sources}
    for label, path in (
        ("output-features", config.output_features_path),
        ("provenance", config.provenance_path),
    ):
        resolved = path.resolve()
        if resolved in source_set:
            raise ValueError(f"{label} path must not overwrite input audio: {path}")
        if model is not None and resolved == model:
            raise ValueError(f"{label} path must not overwrite model: {path}")


def run_audio_mine(
    config: AudioMineConfig,
    *,
    window_scorer: EmbeddingWindowScorer | None = None,
) -> AudioMineResult:
    sources = replay.enumerate_audio_inputs(config.input_paths)
    _validate_output_paths(config, sources)

    scorer = window_scorer
    if scorer is None:
        if config.model_path is None:
            raise ValueError("either window_scorer or model_path is required")
        scorer = build_production_embedding_scorer(config.model_path)

    all_candidates: list[CandidateWindow] = []
    file_records: list[dict[str, Any]] = []

    for source in sources:
        scorer.reset()
        before = source.read_bytes()
        pcm, meta = replay.load_mono_16k(source, gain_db=config.gain_db)
        after = source.read_bytes()
        if before != after:
            raise RuntimeError(f"input audio mutated during read: {source}")

        window_count, used_samples, dropped, _ = replay.plan_audio_windows(pcm.size)
        candidates = collect_candidates_from_pcm(
            pcm,
            scorer,
            file_path=source,
            score_floor=config.score_floor,
            chunk_size=config.chunk_size,
        )
        all_candidates.extend(candidates)
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
                "candidate_count": len(candidates),
                "gain_db": meta["gain_db"],
                "gain_linear": meta["gain_linear"],
            }
        )

    selected = select_from_candidates(
        all_candidates,
        top_k=config.top_k,
        diversity_gap=config.diversity_gap,
    )
    if selected:
        output = np.stack([cand.embedding for _, cand in selected], axis=0).astype(np.float32)
    else:
        output = np.empty((0, *mine.ACAV_ROW_SHAPE), dtype=np.float32)

    mine.atomic_save_npy(config.output_features_path, output)
    output_sha256 = mine._sha256_file(config.output_features_path)
    provenance = build_provenance(
        config=config,
        sources=sources,
        file_records=file_records,
        candidates=all_candidates,
        selected=selected,
        output_array=output,
        output_features_sha256=output_sha256,
    )
    mine.atomic_write_json(config.provenance_path, provenance)

    indices = tuple(idx for idx, _ in selected)
    scores = tuple(cand.score for _, cand in selected)
    return AudioMineResult(
        selected_indices=indices,
        selected_scores=scores,
        output_shape=tuple(output.shape),
        provenance=provenance,
        candidate_count=len(all_candidates),
    )


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
    ap.add_argument(
        "--output-features",
        type=Path,
        required=True,
        help="Destination .npy for selected (K, 16, 96) embeddings",
    )
    ap.add_argument(
        "--provenance",
        type=Path,
        required=True,
        help="Destination JSON provenance for this mining run",
    )
    ap.add_argument("--top-k", type=int, default=512, help="Maximum windows to select")
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Max windows scored per in-memory chunk within one file",
    )
    ap.add_argument(
        "--diversity-gap",
        type=int,
        default=1,
        help="Skip candidates within this many pool indices of an already selected row",
    )
    ap.add_argument(
        "--score-floor",
        type=float,
        default=DEFAULT_SCORE_FLOOR,
        help="Minimum classifier score to enter the candidate pool (default 0.25)",
    )
    ap.add_argument(
        "--gain-db",
        type=float,
        default=0.0,
        help="Fixed mic gain in dB (matches satellite audio.mic_gain_db; default 0)",
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
    if args.score_floor < 0 or args.score_floor > 1:
        print("score-floor must be between 0 and 1", file=sys.stderr)
        return 2
    try:
        replay.validate_gain_db(float(args.gain_db))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1

    config = AudioMineConfig(
        input_paths=tuple(args.inputs),
        output_features_path=args.output_features,
        provenance_path=args.provenance,
        model_path=args.model,
        top_k=int(args.top_k),
        chunk_size=int(args.chunk_size),
        diversity_gap=int(args.diversity_gap),
        score_floor=float(args.score_floor),
        gain_db=float(args.gain_db),
    )
    try:
        result = run_audio_mine(config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(
        f"selected {len(result.selected_indices)} / top_k={config.top_k} "
        f"from {result.candidate_count} candidate window(s)"
    )
    print(f"features  {config.output_features_path}")
    print(f"manifest  {config.provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
