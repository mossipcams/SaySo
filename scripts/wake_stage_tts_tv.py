#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np

SR, WIN = 16000, 32000

EXAMPLES = """\
example (VM layout): positives and the paired near-miss negatives over the same TV distribution
  R=/srv/llm/data/wake/runs/koda-phonics-v6-20261018 TV=/srv/llm/data/wake/runs/koda-phonics-v5-20261010/staging/realneg
  for s in positive negative; do
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= PYTHONPATH=$R/src \\
      /srv/llm/wake/runs/koda-phonics-v3-20261008/.venv/bin/python scripts/wake_stage_tts_tv.py \\
      $R/output/koda-phonics-v6/${s}_train $R/data/rirs $R/staging/ttstv/$s \\
      --tv $TV/tv-20261008 --tv $TV/podcast-tv-20261009 --scorer v9b=$R/output/koda-phonics-v9b/koda-phonics-v9b.onnx
  done
"""


def active_db(w: np.ndarray, floor_db: float = 30.0) -> float:
    """Mean power (dBFS) of 20 ms frames within floor_db of the loudest frame."""
    n = len(w) // 320
    db = 10 * np.log10((w[: n * 320].reshape(n, 320) ** 2).mean(1) + 1e-12)
    return float(10 * np.log10(np.mean(10 ** (db[db >= db.max() - floor_db] / 10))))


def ltas_db(clips: list[np.ndarray]) -> np.ndarray:
    """Mean normalised Welch spectrum (dB, 257 bins) over each clip's active frames."""
    from scipy.signal import welch

    acc = np.zeros(257)
    for x in clips:
        n = len(x) // 320
        fr = x[: n * 320].reshape(n, 320)
        db = 10 * np.log10((fr**2).mean(1) + 1e-12)
        a = fr[db >= db.max() - 30].ravel()
        if len(a) >= 1024:
            acc += (p := welch(a, SR, nperseg=512)[1]) / p.sum()
    return 10 * np.log10(acc + 1e-20)


def match_fir(target: list[np.ndarray], source: list[np.ndarray], taps: int = 257) -> np.ndarray:
    """Linear-phase FIR that moves source's average spectrum onto target's (0 dB at 400-800 Hz)."""
    from scipy.signal import firwin2

    f = np.linspace(0, SR / 2, 257)
    d = np.convolve(ltas_db(target) - ltas_db(source), np.ones(5) / 5, mode="same")
    d -= d[(f >= 400) & (f < 800)].mean()
    return firwin2(taps, f / (SR / 2), 10 ** (np.clip(d, -30, 12) / 20)).astype(np.float32)


def mix_at(voice: np.ndarray, tv: np.ndarray, voice_dbfs: float, tv_gain_db: float) -> np.ndarray:
    """Voice at an absolute active level plus the TV as recorded with a gain.

    Levels are independent: a speaker doesn't get quieter when the TV does, and
    scaling the voice to the TV's local power makes it vanish in the eMeet's gated gaps.
    """
    y = voice * 10 ** ((voice_dbfs - active_db(voice)) / 20) + tv * 10 ** (tv_gain_db / 20)
    pk = np.abs(y).max()
    return (y * (0.99 / pk) if pk > 0.99 else y).astype(np.float32)


def stage(
    clip_dir: Path,
    rir_dir: Path,
    out: Path,
    tv_dirs: list[Path],
    n: int,
    seed: int,
    voice_dbfs: tuple[float, float],
    tv_gain_db: tuple[float, float],
    scorers: dict[str, Path],
    match_dir: Path | None = None,
) -> None:
    import soundfile as sf
    from livekit.wakeword.data.augment import AudioAugmentor, align_clip_to_end
    from livekit.wakeword.eval import stream as S

    random.seed(seed)  # LiveKit's augmentor draws from the stdlib RNG
    rng = np.random.default_rng(seed)
    clips = sorted(glob.glob(str(clip_dir / "clip_[0-9][0-9][0-9][0-9][0-9][0-9].wav")))
    tvs = sorted(p for d in tv_dirs for p in glob.glob(str(d / "*.wav")))
    assert clips and tvs, (clip_dir, tv_dirs)
    pick = rng.choice(len(clips), size=min(n, len(clips)), replace=False)
    aug = AudioAugmentor([], [rir_dir], sample_rate=SR)
    files = {p: sf.SoundFile(p) for p in tvs}
    mel, emb = S.build_feature_pipeline()
    score = {k: S.build_onnx_scorer(p) for k, p in scorers.items()}
    (out / "examples").mkdir(parents=True, exist_ok=True)
    fir = None
    if match_dir is not None:  # the eMeet's average colouring (incl. its ~6.5 kHz band limit), not its gate
        real = [sf.read(p, dtype="float32")[0] for p in sorted(glob.glob(str(match_dir / "*.wav")))]
        tts = [sf.read(clips[k], dtype="float32")[0] for k in rng.choice(len(clips), 2000, replace=True)]
        fir = match_fir(real, tts)
        np.save(out / "match_fir.npy", fir)

    def tv_slice() -> tuple[np.ndarray, str, int]:
        while True:  # skip the eMeet's gated near-silence between programmes
            h = files[tvs[rng.integers(len(tvs))]]
            off = int(rng.integers(0, h.frames - WIN))
            h.seek(off)
            s = h.read(WIN, dtype="float32")
            s = s[:, 0] if s.ndim > 1 else s
            if 20 * np.log10(np.sqrt((s**2).mean()) + 1e-9) > -60:
                return s, Path(h.name).name, off

    rows, man = [], []
    for i, j in enumerate(pick):
        w, sr = sf.read(clips[j], dtype="float32")
        assert sr == SR, (clips[j], sr)
        if fir is not None:
            w = np.convolve(w, fir, mode="same").astype(np.float32)
        # Voice gets its own room; the TV hours were recorded through the room and the eMeet already.
        v = align_clip_to_end(aug.apply_rir(aug.augment_clip(w)), WIN)
        t, name, off = tv_slice()
        lv, tg = float(rng.uniform(*voice_dbfs)), float(rng.uniform(*tv_gain_db))
        y = mix_at(v, t, lv, tg)
        snr = lv - 20 * np.log10(np.sqrt((t**2).mean()) + 1e-9) - tg  # voice active vs 2 s TV RMS
        f = S.window_features(mel, emb, y)
        rows.append(f)
        man.append(
            {"clip": Path(clips[j]).name, "tv": name, "offset": off, "voice_dbfs": round(lv, 2), "tv_gain_db": round(tg, 2), "snr_db": round(float(snr), 2)}
            | {k: float(s(f)) for k, s in score.items()}
        )
        if i < 10:
            sf.write(out / f"examples/{Path(clips[j]).stem}_snr{snr:.0f}.wav", y, SR, subtype="PCM_16")

    x = np.stack(rows).astype(np.float32)
    assert x.shape == (len(pick), 16, 96), x.shape
    np.save(out / "tts_tv_features_train.npy", x)
    with open(out / "tts_tv_manifest.json", "w") as fh:
        json.dump(man, fh, indent=0)
    snr = np.array([m["snr_db"] for m in man])
    edges = [-np.inf, 5, 10, 15, np.inf]
    for k in score:
        s = np.array([m[k] for m in man])
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = s[(snr >= lo) & (snr < hi)]
            print(f"{k} snr {lo:.0f}..{hi:.0f} dB: >= .65 on {100 * (sel >= 0.65).mean():.1f}% (n={len(sel)})")
    print("done", x.shape)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage end-aligned TTS clips over real Pi TV at real-room levels (positives or paired negatives).",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("clip_dir", type=Path, help="LiveKit {positive,negative}_train dir with raw clip_NNNNNN.wav")
    ap.add_argument("rir_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tv", type=Path, action="append", required=True, help="dir of Pi-recorded TV WAVs (train split only)")
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=31)
    # ponytail: measured 2026-11-04. Room Kodas -20..-12 dBFS active; train TV hours -27 dBFS median vs -22 in the
    # real TV-on captures, so the gain centres the TV there. Re-measure if the room, TV volume or mic changes.
    ap.add_argument("--voice-dbfs", type=float, nargs=2, default=(-21.0, -11.0))
    ap.add_argument("--tv-gain-db", type=float, nargs=2, default=(-3.0, 9.0))
    ap.add_argument("--scorer", action="append", default=[], metavar="NAME=MODEL.onnx")
    ap.add_argument("--match", type=Path, help="dir of real eMeet Koda WAVs: EQ the TTS onto their average spectrum")
    args = ap.parse_args()
    scorers = {k: Path(v) for k, v in (s.split("=", 1) for s in args.scorer)}
    stage(args.clip_dir, args.rir_dir, args.out, args.tv, args.n, args.seed,
          tuple(args.voice_dbfs), tuple(args.tv_gain_db), scorers, args.match)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
