#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np

SR, WIN = 16000, 32000
FRAC = (0.35, 0.70)  # of the spoken span: keeps "Ko".."Kod", drops the final schwa
FADE = 160  # 10 ms, so the cut isn't a click the model can key on

EXAMPLES = """\
example (VM layout; v6 raw TTS positives):
  CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \\
  PYTHONPATH=/srv/llm/data/wake/runs/koda-phonics-v6-20261018/src \\
    /srv/llm/wake/runs/koda-phonics-v3-20261008/.venv/bin/python scripts/wake_stage_prefix.py \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/output/koda-phonics-v6/positive_train \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/data \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/staging/prefix \\
    --scorer v9b=output/koda-phonics-v9b/koda-phonics-v9b.onnx \\
    --scorer ens=output/koda-phonics-lg_9b_10k_10l/koda-phonics-lg_9b_10k_10l.onnx

Train with it by appending prefix_negative_features_train.npy to the run's
negative_features_train.npy (the adversarial_negative class), the same way
run_v10k.sh concatenates extra real_positive rows.
"""


def speech_span(audio: np.ndarray, floor_db: float = 35.0) -> tuple[int, int]:
    """First/last sample of 10 ms frames within floor_db of the loudest frame."""
    n = len(audio) // 160
    f = audio[: n * 160].reshape(n, 160)
    db = 10 * np.log10((f**2).mean(1) + 1e-12)
    on = np.flatnonzero(db > db.max() - floor_db)
    return int(on[0]) * 160, (int(on[-1]) + 1) * 160


def prefix_cut(audio: np.ndarray, frac: float) -> np.ndarray:
    """Keep audio up to frac of the spoken span, with a short fade-out."""
    s, e = speech_span(audio)
    cut = s + int(frac * (e - s))
    out = audio[:cut].astype(np.float32)
    k = min(FADE, len(out))
    out[len(out) - k :] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
    return out


def stage(
    pos_dir: Path,
    data_dir: Path,
    out: Path,
    n: int,
    seed: int,
    scorers: dict[str, Path],
) -> None:
    import soundfile as sf
    from livekit.wakeword.data.augment import AudioAugmentor, align_clip_to_end
    from livekit.wakeword.eval import stream as S

    random.seed(seed)  # LiveKit's augmentor draws from the stdlib RNG
    rng = np.random.default_rng(seed)
    clips = sorted(glob.glob(str(pos_dir / "clip_[0-9][0-9][0-9][0-9][0-9][0-9].wav")))
    assert clips, f"no raw clip_NNNNNN.wav in {pos_dir}"
    pick = rng.choice(len(clips), size=min(n, len(clips)), replace=False)
    aug = AudioAugmentor([data_dir / "backgrounds"], [data_dir / "rirs"], sample_rate=SR)
    assert aug.background_files and aug.rir_files, data_dir
    mel, emb = S.build_feature_pipeline()
    score = {k: S.build_onnx_scorer(p) for k, p in scorers.items()}
    (out / "examples").mkdir(parents=True, exist_ok=True)

    rows, man = [], []
    for i, j in enumerate(pick):
        w, sr = sf.read(clips[j], dtype="float32")
        assert sr == SR, (clips[j], sr)
        frac = float(rng.uniform(*FRAC))
        a = aug.augment_clip(prefix_cut(w, frac))
        # End-aligned like LiveKit positives: v9b scores a raw Koda ~0.9 at the
        # window end and ~0.02 mid-window, so a prefix only competes there.
        a = align_clip_to_end(aug.mix_with_background(aug.apply_rir(a)), WIN).astype(np.float32)
        f = S.window_features(mel, emb, a)
        rows.append(f)
        man.append(
            {"clip": Path(clips[j]).name, "frac": round(frac, 3)}
            | {k: float(s(f)) for k, s in score.items()}
        )
        if i < 10:
            sf.write(out / f"examples/{Path(clips[j]).stem}_f{frac:.2f}.wav", a, SR, subtype="PCM_16")

    x = np.stack(rows).astype(np.float32)
    assert x.shape == (len(pick), 16, 96), x.shape
    np.save(out / "prefix_negative_features_train.npy", x)
    with open(out / "prefix_negative_manifest.json", "w") as fh:
        json.dump(man, fh, indent=0)
    fr = np.array([m["frac"] for m in man])
    for k in score:
        s = np.array([m[k] for m in man])
        for lo, hi in ((0.35, 0.5), (0.5, 0.6), (0.6, 0.7)):
            sel = s[(fr >= lo) & (fr < hi)]
            print(f"{k} frac {lo:.2f}-{hi:.2f}: >= .65 on {100 * (sel >= 0.65).mean():.1f}% (n={len(sel)})")
    print("done", x.shape)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage TTS 'Koda' prefixes (Ko..Kod, final vowel cut) as negative features.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pos_dir", type=Path, help="LiveKit positive_train dir with raw clip_NNNNNN.wav")
    ap.add_argument("data_dir", type=Path, help="LiveKit data dir holding backgrounds/ and rirs/")
    ap.add_argument("out", type=Path, help="staging output dir (features .npy + manifest .json)")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--scorer", action="append", default=[], metavar="NAME=MODEL.onnx")
    args = ap.parse_args()
    scorers = {k: Path(v) for k, v in (s.split("=", 1) for s in args.scorer)}
    stage(args.pos_dir, args.data_dir, args.out, args.n, args.seed, scorers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
