#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

RUN = Path("/srv/llm/data/wake/runs/koda-phonics-v6-20261018")
V5 = Path("/srv/llm/data/wake/runs/koda-phonics-v5-20261010/staging/realneg")
WIN = 32000


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x**2).mean()) + 1e-9)


def run(args: argparse.Namespace) -> None:
    import soundfile as sf
    from livekit.wakeword.eval import stream as S

    R = args.run
    OUT = R / "staging/v10v"
    (OUT / "examples").mkdir(parents=True, exist_ok=True)
    TRAIN_TV = sorted(glob.glob(str(args.v5 / "tv-20261008/*.wav")) + glob.glob(str(args.v5 / "podcast-tv-20261009/*.wav")))
    CLIPS = sorted(glob.glob(str(R / "output/koda-phonics-v6b/real_positive_train/clip_*.wav")))
    assert len(CLIPS) == args.n_clips
    rng = np.random.default_rng(args.seed)
    mel, emb = S.build_feature_pipeline()
    sc = S.build_onnx_scorer(R / "output/koda-phonics-lg_10t2_10k_10l/koda-phonics-lg_10t2_10k_10l.onnx")

    def tv() -> np.ndarray:
        while True:
            p = TRAIN_TV[rng.integers(len(TRAIN_TV))]
            s = int(rng.integers(0, sf.info(p).frames - WIN))
            x = sf.read(p, dtype="float32", start=s, stop=s + WIN)[0]
            x = x[:, 0] if x.ndim > 1 else x
            if 20 * np.log10(rms(x)) > -60:
                return x

    rows, man = [], []
    for i, p in enumerate(CLIPS):
        w = sf.read(p, dtype="float32")[0]
        assert len(w) == WIN
        for v in range(4):
            d = int(rng.uniform(args.shift_ms[0] / 1000, args.shift_ms[1] / 1000) * 16000)
            y = np.concatenate([w[d:], w[:d]])
            l = None
            if v >= 2:
                l = float(rng.uniform(*args.tv_dbfs))
                t = tv()
                y = np.clip(y + t * 10 ** (l / 20) / rms(t), -1, 1).astype(np.float32)
            f = S.window_features(mel, emb, y)
            rows.append(f)
            man.append({"clip": Path(p).name, "shift_ms": d / 16, "tv_dbfs": l, "ens": float(sc(f))})
            if i < 6:
                sf.write(OUT / f"examples/{Path(p).stem}_v{v}.wav", y, 16000, subtype="PCM_16")

    a = np.stack(rows).astype(np.float32)
    np.save(OUT / "shift_positive_features.npy", a)
    with open(OUT / "shift_positive_manifest.json", "w") as fh:
        json.dump(man, fh)
    s = np.array([m["ens"] for m in man])
    d = np.array([m["shift_ms"] for m in man])
    c = np.array([m["tv_dbfs"] is None for m in man])
    for lo, hi in ((10, 60), (60, 110), (110, 160)):
        k = (d >= lo) & (d < hi)
        print(f"shift {lo}-{hi} ms: ens >= .65 clean {np.mean(s[k & c] >= .65):.0%} tv {np.mean(s[k & ~c] >= .65):.0%}")
    print("done", a.shape)


def main() -> int:
    ap = argparse.ArgumentParser(description="v10v: hop-phase-jittered real Koda windows (2 clean + 2 over TRAIN TV per clip).")
    ap.add_argument("--run", type=Path, default=RUN, help="koda-phonics run dir (output/, staging/v10v out)")
    ap.add_argument("--v5", type=Path, default=V5, help="realneg dir holding the TRAIN TV wavs")
    ap.add_argument("--n-clips", type=int, default=361, help="expected number of real_positive_train clips")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--shift-ms", type=float, nargs=2, default=(10, 160), help="uniform left-shift range")
    ap.add_argument("--tv-dbfs", type=float, nargs=2, default=(-30, -16), help="TV level range for the TV variants")
    run(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
