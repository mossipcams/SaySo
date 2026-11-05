#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

RUN = Path("/srv/llm/data/wake/runs/koda-phonics-v6-20261018")
V5 = Path("/srv/llm/data/wake/runs/koda-phonics-v5-20261010/staging/realneg")
WIN, SPAN = 32000, 16000


def run(args: argparse.Namespace) -> None:
    import fastfeat
    import soundfile as sf
    from livekit.wakeword.eval import stream as S

    R = args.run
    OUT = R / "staging/v10u"
    OUT.mkdir(exist_ok=True)
    TRAIN_TV = sorted(glob.glob(str(args.v5 / "tv-20261008/*.wav")) + glob.glob(str(args.v5 / "podcast-tv-20261009/*.wav")))
    sc = S.build_onnx_scorer(R / "output/koda-phonics-lg_10t2_10k_10l/koda-phonics-lg_10t2_10k_10l.onnx")
    mel, emb = S.build_feature_pipeline()
    cand, hours = [], 0.0
    for i, p in enumerate(TRAIN_TV):
        a = sf.read(p, dtype="float32")[0]
        a = a[:, 0] if a.ndim > 1 else a
        hours += len(a) / 16000 / 3600
        best = {}
        for e, g in fastfeat.windows(mel, emb, a):
            s = float(sc(g))
            k = e // SPAN
            if s >= args.min and s > best.get(k, (0, 0))[1]:
                best[k] = (e, s)
        cand += [(s, i, e) for e, s in best.values()]
        print(Path(p).name, f"{len(a)/16000/60:.0f} min", len(best), flush=True)
    cand.sort(reverse=True)
    cand = cand[: args.cap]
    rows, man = [], []
    for s, i, e in cand:
        a = sf.read(TRAIN_TV[i], dtype="float32", start=e - WIN, stop=e)[0]
        a = a[:, 0] if a.ndim > 1 else a
        f = S.window_features(mel, emb, a)
        rows.append(f)
        man.append({"file": Path(TRAIN_TV[i]).name, "end": e, "fast": s, "exact": float(sc(f))})
    x = np.stack(rows).astype(np.float32)
    np.save(OUT / "tv_hard_features_train.npy", x)
    with open(OUT / "tv_hard_manifest.json", "w") as fh:
        json.dump(man, fh)
    ex = np.array([m["exact"] for m in man])
    print(f"train TV {hours:.1f} h, kept {len(x)}; exact score >=0.05 {np.sum(ex>=.05)} >=0.1 {np.sum(ex>=.1)} >=0.2 {np.sum(ex>=.2)} >=0.4 {np.sum(ex>=.4)} max {ex.max():.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="v10u: hard negatives mined from the TRAIN TV hours with the deployed ensemble (80 ms hop).")
    ap.add_argument("--run", type=Path, default=RUN, help="koda-phonics run dir (output/, staging/v10u out)")
    ap.add_argument("--v5", type=Path, default=V5, help="realneg dir holding the TRAIN TV wavs")
    ap.add_argument("--min", type=float, default=0.02, help="minimum fast score to keep a window")
    ap.add_argument("--cap", type=int, default=6000, help="keep at most this many rows by score")
    run(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
