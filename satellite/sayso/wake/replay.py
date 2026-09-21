"""Production-path replay of long-form sessions for wake candidate mining."""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .buffer import WakeAudioBuffer
from .eval import CHUNK_SAMPLES
from .livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES, LiveKitWakeWordProvider
from .mining import HardNegativeMiner
from .sessions import RecordingSession, read_session_pcm

REPLAY_SPOOL_DIR = ".replay_spool"

CONTEXT_PRE_MS = 2000
CONTEXT_POST_MS = 2000


@dataclass(frozen=True)
class ReplayConfig:
    sample_rate: int = SAMPLE_RATE
    window_samples: int = WINDOW_SAMPLES
    hop_samples: int = HOP_SAMPLES
    chunk_samples: int = CHUNK_SAMPLES
    context_pre_ms: int = CONTEXT_PRE_MS
    context_post_ms: int = CONTEXT_POST_MS


@dataclass(frozen=True)
class ReplayStats:
    windows_scored: int
    detections: int
    mined_records: int
    duration_seconds: float


def production_replay_constants() -> dict[str, int]:
    return {
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_SAMPLES,
    }


def session_replay_spool_dir(corpus_root: Path, session_id: str) -> Path:
    """Per-session replay spool so re-replay does not accumulate stale records."""
    return Path(corpus_root) / REPLAY_SPOOL_DIR / session_id


def _context_bounds(
    sample_index: int,
    total_samples: int,
    *,
    window_samples: int,
    pre_ms: int,
    post_ms: int,
    sample_rate: int,
) -> tuple[int, int]:
    pre_samples = pre_ms * sample_rate // 1000
    post_samples = post_ms * sample_rate // 1000
    window_start = sample_index - window_samples + 1
    start = max(0, window_start - pre_samples)
    end = min(total_samples, sample_index + post_samples)
    return start, end


def extract_context_pcm(
    pcm: bytes,
    sample_index: int,
    *,
    config: ReplayConfig | None = None,
) -> bytes:
    cfg = config or ReplayConfig()
    samples = np.frombuffer(pcm, dtype="<i2")
    start, end = _context_bounds(
        sample_index,
        samples.size,
        window_samples=cfg.window_samples,
        pre_ms=cfg.context_pre_ms,
        post_ms=cfg.context_post_ms,
        sample_rate=cfg.sample_rate,
    )
    return samples[start:end].tobytes()


def collect_session_activations(
    session: RecordingSession,
    provider: LiveKitWakeWordProvider,
    *,
    config: ReplayConfig | None = None,
    predict: Optional[Callable[..., object]] = None,
) -> tuple[tuple[int, ...], list[float], float, bool]:
    """Scan one session with production hop/lag; return activation sample indices."""
    cfg = config or ReplayConfig()
    if cfg.sample_rate != SAMPLE_RATE:
        raise ValueError(f"replay requires {SAMPLE_RATE} Hz audio")
    if cfg.window_samples != WINDOW_SAMPLES or cfg.hop_samples != HOP_SAMPLES:
        raise ValueError("replay window/hop must match production livekit constants")

    pcm = read_session_pcm(session)
    samples = np.frombuffer(pcm, dtype="<i2")
    buffer = WakeAudioBuffer(cfg.window_samples, cfg.hop_samples)
    provider.reset()
    provider.start()

    activation_samples: list[int] = []
    inference_ms: list[float] = []
    predict_fn = predict

    for start in range(0, samples.size, cfg.chunk_samples):
        chunk = samples[start : start + cfg.chunk_samples]
        if chunk.size == 0:
            continue
        window_end = start + chunk.size
        if not buffer.feed(chunk.tobytes()):
            continue
        window = buffer.window()
        lag = buffer.pending_lag
        sample_index = window_end - lag
        started = time.perf_counter()
        if predict_fn is not None:
            detection = predict_fn(window, sample_index=sample_index)
        else:
            try:
                detection = provider.predict_window(window, sample_index=sample_index)
            except TypeError:
                detection = provider.predict_window(window)
        inference_ms.append((time.perf_counter() - started) * 1000.0)
        if detection is not None:
            stamped = getattr(detection, "sample_index", None)
            activation_samples.append(int(stamped if stamped is not None else sample_index))

    duration_seconds = samples.size / float(cfg.sample_rate)
    return tuple(activation_samples), inference_ms, duration_seconds, bool(activation_samples)


def replay_session(
    session: RecordingSession,
    provider: LiveKitWakeWordProvider,
    *,
    miner: HardNegativeMiner | None = None,
    config: ReplayConfig | None = None,
    predict: Optional[Callable[..., object]] = None,
) -> ReplayStats:
    """Stream one session through the production sliding-window wake path."""
    cfg = config or ReplayConfig()
    if cfg.sample_rate != SAMPLE_RATE:
        raise ValueError(f"replay requires {SAMPLE_RATE} Hz audio")
    if cfg.window_samples != WINDOW_SAMPLES or cfg.hop_samples != HOP_SAMPLES:
        raise ValueError("replay window/hop must match production livekit constants")

    pcm = read_session_pcm(session)
    samples = np.frombuffer(pcm, dtype="<i2")
    buffer = WakeAudioBuffer(cfg.window_samples, cfg.hop_samples)
    provider.reset()
    provider.start()

    previous_miner = provider._miner
    if miner is not None:
        provider._miner = miner

    windows_scored = 0
    detections = 0
    predict_fn = predict
    started = time.perf_counter()

    try:
        for start in range(0, samples.size, cfg.chunk_samples):
            chunk = samples[start : start + cfg.chunk_samples]
            if chunk.size == 0:
                continue
            window_end = start + chunk.size
            if not buffer.feed(chunk.tobytes()):
                continue
            window = buffer.window()
            lag = buffer.pending_lag
            sample_index = window_end - lag
            if predict_fn is not None:
                detection = predict_fn(window, sample_index=sample_index)
            else:
                try:
                    detection = provider.predict_window(window, sample_index=sample_index)
                except TypeError:
                    detection = provider.predict_window(window)
            windows_scored += 1
            if detection is not None:
                detections += 1
    finally:
        if miner is not None:
            provider._miner = previous_miner

    if miner is not None:
        miner._flush_clusters(force=True)
        miner.stop(timeout=5.0)

    mined_records = miner.published_record_count if miner is not None else 0

    duration_seconds = samples.size / float(cfg.sample_rate)
    _ = started  # reserved for future latency metrics
    return ReplayStats(
        windows_scored=windows_scored,
        detections=detections,
        mined_records=mined_records,
        duration_seconds=duration_seconds,
    )


def replay_session_to_spool(
    session: RecordingSession,
    provider: LiveKitWakeWordProvider,
    spool_dir: Path,
    *,
    mine_threshold: float,
    detect_threshold: float,
    model_path: Path | None = None,
    below_sample_rate: float = 0.002,
    predict: Optional[Callable[..., object]] = None,
) -> ReplayStats:
    """Replay a session and publish candidate windows into a mining spool."""
    miner = HardNegativeMiner(
        spool_dir,
        mine_threshold=mine_threshold,
        detect_threshold=detect_threshold,
        model_path=model_path,
        session_id=session.session_id,
        below_sample_rate=below_sample_rate,
    )
    miner.start()
    return replay_session(session, provider, miner=miner, predict=predict)


def replay_and_import_session(
    corpus_root: Path,
    session: RecordingSession,
    provider: LiveKitWakeWordProvider,
    *,
    mine_threshold: float,
    detect_threshold: float,
    model_path: Path | None = None,
    below_sample_rate: float = 0.002,
    predict: Optional[Callable[..., object]] = None,
) -> tuple[ReplayStats, list]:
    """Replay one session into a session-scoped spool and import corpus events."""
    from .corpus import import_spool_records, prune_unlabeled_session_events

    spool_dir = session_replay_spool_dir(corpus_root, session.session_id)
    if spool_dir.is_dir():
        shutil.rmtree(spool_dir)
    spool_dir.mkdir(parents=True, exist_ok=True)

    prune_unlabeled_session_events(corpus_root, session.session_id)

    stats = replay_session_to_spool(
        session,
        provider,
        spool_dir,
        mine_threshold=mine_threshold,
        detect_threshold=detect_threshold,
        model_path=model_path,
        below_sample_rate=below_sample_rate,
        predict=predict,
    )
    imported = import_spool_records(
        spool_dir,
        corpus_root,
        sessions_by_id={session.session_id: session},
        provider=provider,
    )
    return stats, imported
