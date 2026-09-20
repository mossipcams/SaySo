"""Compare LiveKit and NanoWakeWord scores on recorded WAV clips."""

from __future__ import annotations

import argparse
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from satellite.sayso.wake.buffer import WakeAudioBuffer
from satellite.sayso.wake.eval import read_wav_pcm
from satellite.sayso.wake.livekit import (
    HOP_SAMPLES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    LiveKitWakeWordProvider,
)
from satellite.sayso.wake.nanowakeword import NanoWakeWordProvider

CHUNK_SAMPLES = 512


@dataclass(frozen=True)
class ClipCompareResult:
    rel_path: str
    livekit_max: float
    nano_max: float


def collect_wav_paths(audio_dir: Path) -> list[Path]:
    return sorted(path for path in audio_dir.rglob("*.wav") if path.is_file())


def scan_max_score(
    pcm: bytes,
    sample_rate: int,
    score_fn: Callable[[np.ndarray], float],
) -> float:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected sample rate {SAMPLE_RATE}, got {sample_rate}")
    buffer = WakeAudioBuffer(WINDOW_SAMPLES, HOP_SAMPLES)
    max_score = 0.0
    samples = np.frombuffer(pcm, dtype="<i2")
    for start in range(0, samples.size, CHUNK_SAMPLES):
        chunk = samples[start : start + CHUNK_SAMPLES]
        if chunk.size == 0:
            continue
        if buffer.feed(chunk.tobytes()):
            window = buffer.window()
            max_score = max(max_score, float(score_fn(window)))
    return max_score


def livekit_window_score(provider: LiveKitWakeWordProvider, window: np.ndarray) -> float:
    if not provider.available or provider._model is None:
        return 0.0
    if window.size < WINDOW_SAMPLES:
        return 0.0
    scorer = provider._scorer
    scores = scorer.score(window) if scorer else provider._model.predict(window)
    if not scores:
        return 0.0
    score_key = provider._score_key
    if score_key and score_key in scores:
        return float(scores[score_key])
    return float(next(iter(scores.values())))


def nano_window_score(provider: NanoWakeWordProvider, window: np.ndarray) -> float:
    return provider.feed_window_score(window)


def score_clip(
    wav_path: Path,
    rel_path: str,
    livekit: LiveKitWakeWordProvider,
    nano: NanoWakeWordProvider,
) -> ClipCompareResult:
    pcm, rate = read_wav_pcm(wav_path)
    livekit_max = scan_max_score(pcm, rate, lambda w: livekit_window_score(livekit, w))
    nano.reset()
    # Isolated 2s takes starve Nano's streaming features. 1s of silence on
    # each side matches live lead-in; LiveKit still scores the raw clip.
    pad = np.zeros(SAMPLE_RATE, dtype="<i2").tobytes()
    nano_max = scan_max_score(pad + pcm + pad, rate, lambda w: nano_window_score(nano, w))
    return ClipCompareResult(rel_path=rel_path, livekit_max=livekit_max, nano_max=nano_max)


def print_header(clip_count: int) -> None:
    print(f"clips={clip_count} scores=max (no threshold)")
    print(f"{'clip':<40} {'lk_max':>8} {'nano_max':>9}")


def print_result(result: ClipCompareResult) -> None:
    print(
        f"{result.rel_path:<40} "
        f"{result.livekit_max:>8.4f} "
        f"{result.nano_max:>9.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare LiveKit and NanoWakeWord scores on recorded WAV clips",
    )
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory of WAV clips")
    parser.add_argument("--livekit", type=Path, required=True, help="LiveKit ONNX model path")
    parser.add_argument("--nano", type=Path, required=True, help="NanoWakeWord ONNX model path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audio_dir = args.audio_dir

    if not audio_dir.is_dir():
        print(f"audio dir missing: {audio_dir}", file=sys.stderr)
        return 2

    wav_paths = collect_wav_paths(audio_dir)
    print_header(len(wav_paths))
    if not wav_paths:
        return 0

    if not args.livekit.is_file():
        print(f"livekit model missing: {args.livekit}", file=sys.stderr)
        return 2
    if not args.nano.is_file():
        print(f"nano model missing: {args.nano}", file=sys.stderr)
        return 2

    # Dummy threshold: eval reports raw max scores and never uses fire/detect.
    livekit = LiveKitWakeWordProvider(
        args.livekit,
        phrase="SaySo",
        threshold=1.0,
        refractory_seconds=0.0,
    )
    nano = NanoWakeWordProvider(
        args.nano,
        phrase="SaySo",
        threshold=1.0,
        refractory_seconds=0.0,
    )
    if not livekit.available:
        print(f"failed to load livekit model: {args.livekit}", file=sys.stderr)
        return 1
    if not nano.available:
        print(f"failed to load nano model: {args.nano}", file=sys.stderr)
        return 1

    for wav_path in wav_paths:
        rel_path = str(wav_path.relative_to(audio_dir))
        try:
            result = score_clip(wav_path, rel_path, livekit, nano)
        except (OSError, ValueError, wave.Error) as exc:
            print(f"{rel_path}: error: {exc}", file=sys.stderr)
            continue
        print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
