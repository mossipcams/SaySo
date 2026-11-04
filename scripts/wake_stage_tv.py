#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

VARIANTS = {
    "v10k": {
        "seed": 17,
        "tv_dbfs": (-30.0, -16.0),
        "report_bands": [(-30, 7), (-23, 7)],
    },
    "v10l": {
        "seed": 19,
        "tv_dbfs": (-30.0, -20.0),
        "report_bands": [(-30, 5), (-25, 5)],
    },
}

EXAMPLES = """\
examples (VM layout):
  python3 scripts/wake_stage_tv.py --variant v10k \\
    /srv/llm/data/wake/runs/koda-phonics-v5-20261010/staging/realneg \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/output/koda-phonics-v6b/real_positive_train \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/staging/v10k \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/output/koda-phonics-v10j2/koda-phonics-v10j2.onnx

  python3 scripts/wake_stage_tv.py --variant v10l \\
    /srv/llm/data/wake/runs/koda-phonics-v5-20261010/staging/realneg \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/output/koda-phonics-v6b/real_positive_train \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/staging/v10l \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/output/koda-phonics-v10j2/koda-phonics-v10j2.onnx
"""


def rms(x) -> float:
    import numpy as np

    return float(np.sqrt((x**2).mean()) + 1e-9)


def stage(
    variant: str,
    realneg: Path,
    clips_dir: Path,
    out: Path,
    scorer_path: Path,
) -> None:
    import numpy as np
    import soundfile as sf
    from livekit.wakeword.eval import stream as S

    spec = VARIANTS[variant]
    (out / "examples").mkdir(parents=True, exist_ok=True)
    train_tv = sorted(
        glob.glob(str(realneg / "tv-20261008/*.wav"))
        + glob.glob(str(realneg / "podcast-tv-20261009/*.wav"))
    )
    assert len(train_tv) == 12, train_tv
    clips = sorted(glob.glob(str(clips_dir / "clip_*.wav")))
    assert len(clips) == 361, len(clips)
    win, mixes = 32000, 3
    rng = np.random.default_rng(spec["seed"])
    scorer = S.build_onnx_scorer(scorer_path)
    mel, emb = S.build_feature_pipeline()

    def tv() -> "np.ndarray":
        while True:
            p = train_tv[rng.integers(len(train_tv))]
            s = int(rng.integers(0, sf.info(p).frames - win))
            x = sf.read(p, dtype="float32", start=s, frames=win)[0]
            x = x[:, 0] if x.ndim > 1 else x
            if 20 * np.log10(rms(x)) > -60:
                return x

    rows, man = [], []
    for i, p in enumerate(clips):
        w, sr = sf.read(p, dtype="float32")
        assert sr == 16000 and len(w) == win, (p, sr, len(w))
        for m in range(mixes):
            l, g = float(rng.uniform(*spec["tv_dbfs"])), float(rng.uniform(-3, 3))
            t = tv()
            y = np.clip(w * 10 ** (g / 20) + t * 10 ** (l / 20) / rms(t), -1, 1).astype(
                np.float32
            )
            f = S.window_features(mel, emb, y)
            rows.append(f)
            man.append(
                {
                    "clip": Path(p).name,
                    "mix": m,
                    "tv_dbfs": l,
                    "gain_db": g,
                    "v10j2": float(scorer(f)),
                }
            )
            if i < 10 and m == 0:
                sf.write(
                    out / f"examples/{Path(p).stem}_tv{l:.0f}.wav",
                    y,
                    16000,
                    subtype="PCM_16",
                )
    a = np.stack(rows).astype(np.float32)
    assert a.shape == (len(clips) * mixes, 16, 96), a.shape
    np.save(out / "real_loudtv_positive_features.npy", a)
    with open(out / "real_loudtv_positive_manifest.json", "w") as fh:
        json.dump(man, fh, indent=0)
    s = np.array([m["v10j2"] for m in man])
    l = np.array([m["tv_dbfs"] for m in man])
    for lo, width in spec["report_bands"]:
        sel = s[(l >= lo) & (l < lo + width)]
        print(
            f"TV {lo}..{lo + width} dBFS: v10j2 >= .65 on {100 * (sel >= 0.65).mean():.0f}%"
        )
    print("done", a.shape)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Stage real-voice-over-loud-TV positive features (361 v6b Koda windows, 3 mixes each; "
            "v10k mixes TV at -30..-16 dBFS, v10l caps it at -20 dBFS)."
        ),
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--variant", choices=("v10k", "v10l"), required=True)
    ap.add_argument(
        "realneg",
        type=Path,
        help="realneg staging dir holding tv-20261008/ and podcast-tv-20261009/",
    )
    ap.add_argument(
        "clips_dir", type=Path, help="dir with the 361 real_positive clip_*.wav windows"
    )
    ap.add_argument(
        "out", type=Path, help="staging output dir (features .npy + manifest .json)"
    )
    ap.add_argument(
        "scorer",
        type=Path,
        help="v10j2 ONNX used only to record the v10j2 column in the manifest",
    )
    args = ap.parse_args()
    stage(args.variant, args.realneg, args.clips_dir, args.out, args.scorer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
