#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import wave
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, resample_poly

SAMPLE_RATE = 16000
FRAME = 2 * SAMPLE_RATE
REFERENCE_M = 0.3
DISTANCES_M = (1.0, 2.0, 3.0)
DIRECT_MS = 2.5
SEED_CLASSES = ("wake", "missed_wake")
KODA_LIKE = re.compile(r"[kc]oda|toda|poda|code up", re.I)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16 kHz mono")
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def load_asset(path: Path) -> np.ndarray:
    import soundfile

    x, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = np.gcd(rate, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, rate // g).astype(np.float32)
    return x


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def is_holdout(sha256: str) -> bool:
    return int(sha256[:8], 16) % 3 == 0


def select(spool: Path) -> list[dict]:
    seeds = []
    for check in sorted((spool / "records").glob("*/check.json")):
        info = json.loads(check.read_text(encoding="utf-8"))
        if info.get("class") not in SEED_CLASSES:
            continue
        sha = json.loads((check.parent / "record.json").read_text(encoding="utf-8"))["hashes"]["window_sha256"]
        seeds.append({
            "capture_id": check.parent.name,
            "sha256": sha,
            "transcript": info.get("transcript"),
            "split": "holdout" if is_holdout(sha) else "train",
            "verified": False,
        })
    return seeds


def trim_speech(x: np.ndarray, pad_ms: int = 80, floor_db: float = -35.0) -> np.ndarray:
    win = SAMPLE_RATE // 100
    n = len(x) // win
    rms = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1)) + 1e-9
    db = 20 * np.log10(rms / rms.max())
    idx = np.flatnonzero(db > floor_db)
    if idx.size == 0:
        return x
    pad = pad_ms // 10
    lo = max(int(idx[0]) - pad, 0) * win
    hi = min(int(idx[-1]) + 1 + pad, n) * win
    return x[lo:hi]


def place(speech: np.ndarray, rng: np.random.Generator, tail_s: float = 0.5) -> np.ndarray:
    frame = np.zeros(FRAME, dtype=np.float32)
    speech = speech[-FRAME:]
    end = FRAME - int(rng.uniform(0.0, tail_s) * SAMPLE_RATE)
    end = max(end, len(speech))
    frame[end - len(speech): end] = speech
    return frame


def speed_pitch(x: np.ndarray, factor: float) -> np.ndarray:
    return resample_poly(x, 100, int(round(100 * factor))).astype(np.float32)


def split_rir(rir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peak = int(np.argmax(np.abs(rir)))
    cut = peak + int(DIRECT_MS * SAMPLE_RATE / 1000)
    direct = np.zeros_like(rir)
    tail = np.zeros_like(rir)
    direct[:cut] = rir[:cut]
    tail[cut:] = rir[cut:]
    return direct, tail


def distance_rir(rir: np.ndarray, distance_m: float) -> np.ndarray:
    direct, tail = split_rir(rir)
    return direct * (REFERENCE_M / distance_m) + tail


def apply_rir(x: np.ndarray, rir: np.ndarray, distance_m: float) -> np.ndarray:
    direct, _ = split_rir(rir)
    near_rms = np.sqrt((fftconvolve(x, direct)[: len(x)] ** 2).mean()) + 1e-9
    far = fftconvolve(x, distance_rir(rir, distance_m))[: len(x)]
    ref_rms = np.sqrt((x ** 2).mean()) + 1e-9
    return (far * (ref_rms / near_rms)).astype(np.float32)


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < len(x):
        noise = np.tile(noise, len(x) // len(noise) + 1)
    start = int(rng.integers(0, len(noise) - len(x) + 1))
    noise = noise[start: start + len(x)]
    sig = np.sqrt((x ** 2).mean()) + 1e-9
    nz = np.sqrt((noise ** 2).mean()) + 1e-9
    return (x + noise * (sig / nz) * 10 ** (-snr_db / 20)).astype(np.float32)


def near_variant(seed: np.ndarray, noise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = speed_pitch(trim_speech(seed), float(rng.uniform(0.9, 1.1)))
    x = place(x, rng) * 10 ** (float(rng.uniform(-6, 6)) / 20)
    return add_noise(x, noise, float(rng.uniform(5, 25)), rng)


def far_variant(x: np.ndarray, rir: np.ndarray, noise: np.ndarray, distance_m: float, rng: np.random.Generator) -> np.ndarray:
    far = apply_rir(x, rir, distance_m)
    snr = float(rng.uniform(15, 25)) - 10 * np.log2(distance_m) * 1.5
    return add_noise(far, noise, snr, rng)


def build(seeds: list[dict], spool: Path, rirs: list[Path], backgrounds: list[Path], out: Path, n: int, rng_seed: int,
          start_index: int = 100000) -> int:
    rng = np.random.default_rng(rng_seed)
    out.mkdir(parents=True, exist_ok=True)
    rir_cache = [load_asset(p) for p in rirs]
    bg_cache = [load_asset(p) for p in backgrounds]
    written = 0
    manifest = {}
    for seed in seeds:
        if seed["split"] != "train" or not seed["verified"]:
            continue
        audio = read_wav(spool / "records" / seed["capture_id"] / "window.wav")
        if hashlib.sha256((spool / "records" / seed["capture_id"] / "window.wav").read_bytes()).hexdigest() != seed["sha256"]:
            raise ValueError(f"{seed['capture_id']}: window.wav sha256 mismatch")
        for i in range(n):
            base = near_variant(audio, bg_cache[int(rng.integers(len(bg_cache)))], rng)
            name = f"clip_{start_index + written:06d}.wav"
            write_wav(out / name, base)
            manifest[name] = f"{seed['capture_id']} near"
            written += 1
            for d in DISTANCES_M:
                rir = rir_cache[int(rng.integers(len(rir_cache)))]
                far = far_variant(base, rir, bg_cache[int(rng.integers(len(bg_cache)))], d, rng)
                name = f"clip_{start_index + written:06d}.wav"
                write_wav(out / name, far)
                manifest[name] = f"{seed['capture_id']} {int(d)}m"
                written += 1
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return written


def select_negatives(spool: Path) -> list[dict]:
    rows = []
    for check in sorted((spool / "records").glob("*/check.json")):
        transcript = json.loads(check.read_text(encoding="utf-8")).get("transcript") or ""
        if KODA_LIKE.search(transcript):
            continue
        sha = json.loads((check.parent / "record.json").read_text(encoding="utf-8"))["hashes"]["window_sha256"]
        rows.append({
            "capture_id": check.parent.name,
            "sha256": sha,
            "transcript": transcript,
            "split": "holdout" if int(sha[:8], 16) % 5 == 0 else "train",
        })
    return rows


def build_negatives(rows: list[dict], spool: Path, out: Path, n: int, rng_seed: int, start_index: int = 200000) -> int:
    rng = np.random.default_rng(rng_seed)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        if row["split"] != "train":
            continue
        path = spool / "records" / row["capture_id"] / "window.wav"
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"{row['capture_id']}: window.wav sha256 mismatch")
        audio = read_wav(path)
        for _ in range(n):
            shifted = np.roll(audio, int(rng.integers(-SAMPLE_RATE // 4, SAMPLE_RATE // 4)))
            write_wav(out / f"clip_{start_index + written:06d}.wav", shifted * 10 ** (float(rng.uniform(-6, 6)) / 20))
            written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("spool", type=Path)
    s.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("variants")
    v.add_argument("spool", type=Path)
    v.add_argument("--seeds", type=Path, required=True)
    v.add_argument("--rirs", type=Path, required=True)
    v.add_argument("--backgrounds", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("-n", type=int, default=50)
    v.add_argument("--rng-seed", type=int, default=0)
    ng = sub.add_parser("negatives")
    ng.add_argument("spool", type=Path)
    ng.add_argument("--out", type=Path, required=True)
    ng.add_argument("-n", type=int, default=5)
    ng.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "negatives":
        rows = select_negatives(args.spool)
        count = build_negatives(rows, args.spool, args.out, args.n, args.rng_seed)
        held = sum(r["split"] == "holdout" for r in rows)
        print(f"wrote {count} negative clips ({held} of {len(rows)} windows held out)")
        return 0
    if args.cmd == "select":
        seeds = select(args.spool)
        args.out.write_text(json.dumps(seeds, indent=2), encoding="utf-8")
        print(f"{len(seeds)} candidates; set verified=true after listening")
        return 0
    seeds = json.loads(args.seeds.read_text(encoding="utf-8"))
    count = build(seeds, args.spool, sorted(args.rirs.glob("*.wav")), sorted(args.backgrounds.rglob("*.wav")),
                  args.out, args.n, args.rng_seed)
    print(f"wrote {count} clips to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
