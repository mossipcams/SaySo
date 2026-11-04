#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
from collections.abc import Callable
from pathlib import Path

Score = Callable[[object], float]

WIN, HOP, REFR = 32000, 1280, 25
CHUNK = HOP * 400

EXAMPLES = """\
examples (VM layout):
  python3 scripts/wake_eval_realtv.py real \\
    --pos /srv/llm/data/wake/runs/koda-phonics-v6-20261018/staging/holdout/pos \\
    --tv /srv/llm/data/wake/runs/koda-phonics-v6-20261018/staging/holdout-tv \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/output/koda-phonics-v9b/koda-phonics-v9b.onnx \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/output/koda-phonics-lg_9b_10k_10l/koda-phonics-lg_9b_10k_10l.onnx

  python3 scripts/wake_eval_realtv.py fa \\
    --tv /srv/llm/data/wake/runs/koda-phonics-v6-20261018/staging/holdout-tv \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/output/koda-phonics-lg_9b_10k_10l/koda-phonics-lg_9b_10k_10l.onnx
"""


def rms(x) -> float:
    import numpy as np

    return float(np.sqrt((x**2).mean()) + 1e-9)


def fast_windows(mel, emb, a):
    import numpy as np

    for c in range(0, len(a) - WIN + 1, CHUNK):
        x = a[c : c + CHUNK + WIN]
        e = emb.extract_embeddings(mel(x[np.newaxis, :]))[0]
        for m in range(0, min(CHUNK, len(x) - WIN + 1), HOP):
            j = m // HOP
            if j + 16 <= len(e):
                yield c + m + WIN, e[j : j + 16]


def real_eval(pos_dir: Path, tv_dir: Path, models: dict[str, Score]) -> None:
    import numpy as np
    import soundfile as sf
    from livekit.wakeword.eval import stream as S

    mel, emb = S.build_feature_pipeline()
    rng = np.random.default_rng(21)
    clips = sorted(glob.glob(str(pos_dir / "*.wav")))
    tv = [
        sf.read(p, dtype="float32")[0] for p in sorted(glob.glob(str(tv_dir / "*.wav")))
    ]
    assert len(clips) == 33 and len(tv) == 2, (len(clips), len(tv))

    def pick(n: int, r):
        while True:
            a = tv[r.integers(2)]
            s = int(r.integers(0, len(a) - n))
            x = a[s : s + n]
            if 20 * np.log10(rms(x)) > -60:
                return x

    cases: dict[str, list[object]] = {"clean": [], "tv -48..-38": [], "tv -38..-28": []}
    for p in clips:
        w = sf.read(p, dtype="float32")[0]
        w = np.pad(w, (max(0, WIN - len(w)), 0))
        cases["clean"].append(w)
        for band, lo, hi in (("tv -48..-38", -48, -38), ("tv -38..-28", -38, -28)):
            for _ in range(4):
                t = pick(len(w), rng)
                cases[band].append(
                    np.clip(
                        w + t * 10 ** (rng.uniform(lo, hi) / 20) / rms(t), -1, 1
                    ).astype(np.float32)
                )
    rng2 = np.random.default_rng(23)
    cases["tv -28..-18"] = []
    for w in cases["clean"]:
        for _ in range(4):
            t = pick(len(w), rng2)
            cases["tv -28..-18"].append(
                np.clip(
                    w + t * 10 ** (rng2.uniform(-28, -18) / 20) / rms(t), -1, 1
                ).astype(np.float32)
            )
    score = {b: {k: [] for k in models} for b in cases}
    for b, xs in cases.items():
        for x in xs:
            fs = [
                S.window_features(mel, emb, x[e - WIN : e])
                for e in range(WIN, len(x) + 1, HOP)
            ]
            for k, m in models.items():
                score[b][k].append(max(m(f) for f in fs))
    for th in (0.45, 0.55, 0.65, 0.75, 0.8, 0.85):
        print(
            f"th {th:.2f}: "
            + " || ".join(
                f"{b} (/{len(cases[b])}): "
                + " ".join(f"{k} {sum(s >= th for s in score[b][k])}" for k in models)
                for b in cases
            )
        )


def fa_eval(tv_dir: Path, models: dict[str, Score]) -> None:
    import soundfile as sf
    from livekit.wakeword.eval import stream as S

    mel, emb = S.build_feature_pipeline()
    hits: dict[str, list[tuple[int, int, float]]] = {k: [] for k in models}
    hours = 0.0
    for i, p in enumerate(sorted(glob.glob(str(tv_dir / "*.wav")))):
        a = sf.read(p, dtype="float32")[0]
        hours += len(a) / 16000 / 3600
        for e, g in fast_windows(mel, emb, a):
            cand = [k for k in models if models[k](g) >= 0.2]
            if not cand:
                continue
            f = S.window_features(mel, emb, a[e - WIN : e])
            for k in cand:
                hits[k].append((i, e // HOP, float(models[k](f))))

    def fires(h, th) -> int:
        n, last = 0, None
        for i, j, s in sorted(h):
            if s >= th and (last is None or last[0] != i or j - last[1] >= REFR):
                n, last = n + 1, (i, j)
        return n

    for th in (0.55, 0.65, 0.75, 0.85):
        print(
            f"th {th:.2f} FA/h on {hours:.1f} h: "
            + " | ".join(f"{k} {fires(hits[k], th) / hours:.2f}" for k in models)
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Real-voice-over-holdout-TV eval: `real` mixes the 33 held-out room Kodas over the 2 holdout TV hours "
            "at -48..-38, -38..-28 and -28..-18 dBFS (4 mixes per band) and scores max over 2 s windows at the 80 ms "
            "production hop; `fa` reports false wakes/h on the holdout TV hours (fast candidate features >= 0.2, "
            "exact re-score, 2 s refractory). Same seeds for every model."
        ),
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    preal = sub.add_parser("real", help="real voice over holdout TV bands")
    preal.add_argument(
        "--pos",
        type=Path,
        required=True,
        help="dir with the 33 held-out real room Koda wavs",
    )
    preal.add_argument(
        "--tv", type=Path, required=True, help="dir with the 2 holdout TV hour wavs"
    )
    preal.add_argument("models", nargs="+", type=Path)
    pfa = sub.add_parser("fa", help="holdout-TV false wakes/h per threshold")
    pfa.add_argument(
        "--tv", type=Path, required=True, help="dir with the 2 holdout TV hour wavs"
    )
    pfa.add_argument("models", nargs="+", type=Path)
    args = ap.parse_args()
    from livekit.wakeword.eval import stream as S

    models = {}
    for m in args.models:
        key = m.stem
        models[key] = S.build_onnx_scorer(m)
    if args.cmd == "real":
        real_eval(args.pos, args.tv, models)
    else:
        fa_eval(args.tv, models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
