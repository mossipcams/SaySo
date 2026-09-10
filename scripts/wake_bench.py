#!/usr/bin/env python3
"""Verify embedding reuse is score-identical to stateless predict(), and time it.

Run on the satellite, where the real livekit mel/embedding ONNX models live:

    systemctl --user stop sayso-satellite
    PYTHONPATH=/opt/sayso-satellite /opt/sayso-satellite/.venv/bin/python \
        scripts/wake_bench.py /opt/sayso-satellite/models/sayso.onnx AUDIO.wav
    systemctl --user start sayso-satellite

The equivalence assertion is the point. The speedup is worthless if the cached
path scores differently from what the model was evaluated with.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

HOP_SAMPLES = 2560
WINDOW_SAMPLES = 32000


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise SystemExit(f"expected 16-bit PCM: {path}")
        rate = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
        if wf.getnchannels() > 1:
            data = data.reshape(-1, wf.getnchannels())[:, 0]
    if rate != 16000:
        raise SystemExit(f"expected 16 kHz, got {rate}: {path}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--windows", type=int, default=60)
    args = ap.parse_args()

    from livekit.wakeword import WakeWordModel

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
    from sayso.wake.streaming import CachedEmbeddingScorer

    audio = read_wav(args.audio)
    starts = list(range(0, audio.size - WINDOW_SAMPLES + 1, HOP_SAMPLES))[: args.windows]
    if not starts:
        raise SystemExit("audio too short for even one 2 s window")
    windows = [audio[s : s + WINDOW_SAMPLES] for s in starts]

    model = WakeWordModel(models=[str(args.model)])
    key = args.model.stem
    scorer = CachedEmbeddingScorer(model)
    if not scorer.supported:
        raise SystemExit("embedding reuse unsupported against this livekit-wakeword build")

    def timed(fn, items):
        out, times = [], []
        for w in items:
            t0 = time.perf_counter()
            out.append(fn(w))
            times.append((time.perf_counter() - t0) * 1000.0)
        return out, sorted(times)

    baseline, t_base = timed(lambda w: model.predict(w)[key], windows)
    cached, t_cached = timed(lambda w: scorer.score(w)[key], windows)

    diffs = np.abs(np.array(baseline) - np.array(cached))
    n = len(windows)
    computed, reused = scorer.stats

    print(f"windows            {n}  (hop {HOP_SAMPLES} samples)")
    print(f"stateless  p50     {t_base[n // 2]:7.1f} ms   p95 {t_base[int(n * 0.95)]:7.1f} ms")
    print(f"cached     p50     {t_cached[n // 2]:7.1f} ms   p95 {t_cached[int(n * 0.95)]:7.1f} ms")
    print(f"speedup            {t_base[n // 2] / max(t_cached[n // 2], 1e-9):7.2f}x")
    print(f"hop budget 160 ms  realtime factor {t_cached[n // 2] / 160.0:.2f}x "
          f"(was {t_base[n // 2] / 160.0:.2f}x)")
    print(f"embeddings         computed {computed}, reused {reused} "
          f"(stateless would compute {n * 16})")
    print(f"max score diff     {diffs.max():.3e}")

    if diffs.max() > 1e-6:
        print("\nFAIL: cached scores diverge from stateless predict()")
        return 1
    print("\nOK: cached path is score-identical to stateless predict()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
