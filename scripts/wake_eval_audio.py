#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

RUN = Path("/srv/llm/data/wake/runs/koda-phonics-v6-20261018")
WIN, HOP, PH = 32000, 1280, 16


def run(args: argparse.Namespace) -> None:
    import fastfeat
    import onnxruntime as ort
    import soundfile as sf
    from livekit.wakeword.eval import stream as S

    R = args.run
    RUNS = R.parent
    mel, emb = S.build_feature_pipeline()
    rms = lambda x: float(np.sqrt((x**2).mean()) + 1e-9)

    class V2:
        def __init__(self, path):
            o = ort.SessionOptions()
            o.intra_op_num_threads = 1
            o.inter_op_num_threads = 1
            self.s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
            self.names = [x.name.removesuffix("_out") for x in self.s.get_outputs()[1:]]
            st = {i.name: np.zeros(i.shape, np.int64 if "int64" in i.type else np.float32) for i in self.s.get_inputs() if i.name != "chunk"}
            for _ in range(25):
                st = self._step(np.zeros((1, HOP), np.float32), st)[1]
            self.warm = st

        def _step(self, chunk, st):
            out = self.s.run(None, {"chunk": chunk, **st})
            return float(np.asarray(out[0]).reshape(-1)[0]), dict(zip(self.names, out[1:]))

        def steps(self, x, off=0):
            st, out = self.warm, []
            for e in range(off + HOP, len(x) + 1, HOP):
                s, st = self._step(x[None, e - HOP : e].astype(np.float32), st)
                out.append((e, s))
            return out

    class V1:
        def __init__(self, name):
            self.m = S.build_onnx_scorer(R / f"output/koda-phonics-{name}/koda-phonics-{name}.onnx")

        def steps(self, x, off=0):
            return [(e, float(self.m(S.window_features(mel, emb, x[e - WIN : e])))) for e in range(off + WIN, len(x) + 1, HOP)]

        def long_steps(self, a, off=0):
            out = []
            for e, g in fastfeat.windows(mel, emb, a[off:]):
                s = float(self.m(g))
                if s >= 0.2:
                    s = float(self.m(S.window_features(mel, emb, a[off + e - WIN : off + e])))
                out.append((off + e, s))
            return out

    M = {k: (V2(k) if k.endswith(".onnx") else V1(k)) for k in args.models}
    lab = {k: (Path(k).stem if k.endswith(".onnx") else k) for k in M}
    show = lambda d: " | ".join(f"{lab[k]} {v}" for k, v in d.items())

    rng = np.random.default_rng(21)
    CLIPS = sorted(glob.glob(str(R / "staging/holdout/pos/*.wav")))
    TV = [sf.read(p, dtype="float32")[0] for p in sorted(glob.glob(str(R / "staging/holdout-tv/*.wav")))]
    assert len(CLIPS) == 33 and len(TV) == 2

    def tv(n, rng=rng):
        while True:
            a = TV[rng.integers(2)]
            s = int(rng.integers(0, len(a) - n))
            x = a[s : s + n]
            if 20 * np.log10(rms(x)) > -60:
                return x

    cases = {"clean": [], "tv -48..-38": [], "tv -38..-28": []}
    for p in CLIPS:
        w = sf.read(p, dtype="float32")[0]
        w = np.pad(w, (max(0, WIN - len(w)), 0))
        cases["clean"].append(w)
        for band, (lo, hi) in (("tv -48..-38", (-48, -38)), ("tv -38..-28", (-38, -28))):
            for _ in range(4):
                t = tv(len(w))
                cases[band].append(np.clip(w + t * 10 ** (rng.uniform(lo, hi) / 20) / rms(t), -1, 1).astype(np.float32))
    rng2 = np.random.default_rng(23)
    cases["tv -28..-18"] = []
    for w in cases["clean"]:
        for _ in range(4):
            t = tv(len(w), rng2)
            cases["tv -28..-18"].append(np.clip(w + t * 10 ** (rng2.uniform(-28, -18) / 20) / rms(t), -1, 1).astype(np.float32))
    pos_tts = sorted(glob.glob(str(R / "output/koda-phonics-v6/positive_test/clip_*.wav")))[: args.n_tts]
    cases["tts clean"] = [np.pad(sf.read(p, dtype="float32")[0], (WIN, 4000)) for p in pos_tts]
    best = {b: {k: [max(s for _, s in m.steps(x)) for x in xs] for k, m in M.items()} for b, xs in cases.items()}
    for th in (0.45, 0.55, 0.65, 0.75):
        print(f"th {th:.2f}: " + " || ".join(f"{b} (/{len(cases[b])}): " + show({k: sum(s >= th for s in best[b][k]) for k in M}) for b in cases))

    sets = {
        "quiet": [
            d
            for d in sorted(glob.glob(str(RUNS / f"{args.quiet_run}/fired/*/")) + glob.glob(str(RUNS / f"{args.quiet_run}/blocked/*/")))
            if args.exclude not in d
        ],
        "tv_on": sorted(glob.glob(str(RUNS / f"{args.tv_run}/*/*/"))),
    }
    for name, dirs in sets.items():
        rate = {k: [] for k in M}
        for d in map(Path, dirs):
            rd = lambda n: sf.read(d / n, dtype="float32")[0] if (d / n).exists() else np.zeros(0, np.float32)
            x = np.concatenate([rd("pre.wav"), rd("window.wav"), rd("post.wav")])
            e0 = len(rd("pre.wav")) + WIN
            x = np.pad(x, (WIN, 0))
            e0 += WIN
            for k, m in M.items():
                hits = []
                for p in range(PH):
                    st = [s for e, s in m.steps(x, p * HOP // PH) if e0 - 6 * HOP <= e <= e0 + 6 * HOP]
                    hits.append(max(st) if st else 0.0)
                rate[k].append(np.array(hits))
        for th in (0.5, 0.65):
            print(f"{name} (/{len(dirs)}) expected hits @{th}: " + show({k: f"{sum((r >= th).mean() for r in rate[k]):.2f}" for k in M}))

    hours = sum(len(a) for a in TV) / 16000 / 3600
    for hop_offs in ((0,), (0, 640)):
        ev = {k: [] for k in M}
        for i, a in enumerate(TV):
            for off in hop_offs:
                for k, m in M.items():
                    ev[k] += [(i, e, s) for e, s in (m.long_steps(a, off) if isinstance(m, V1) else m.steps(a, off))]

        def fires(h, th):
            n, last = 0, None
            for i, e, s in sorted(h):
                if s >= th and (last is None or last[0] != i or e - last[1] >= WIN):
                    n, last = n + 1, (i, e)
            return n

        tag = "80" if len(hop_offs) == 1 else "40"
        for th in (0.35, 0.45, 0.55, 0.65, 0.75):
            print(f"holdout TV hop{tag} th {th:.2f} FA/h on {hours:.1f} h: " + show({k: f"{fires(ev[k], th) / hours:.2f}" for k in M}))
        print(f"holdout TV hop{tag} top scores: " + show({k: sorted((round(s, 2) for _, _, s in ev[k]), reverse=True)[:5] for k in M}))


def main() -> int:
    ap = argparse.ArgumentParser(description="Audio-level eval of v1 heads and v2 streaming ONNX on identical audio.")
    ap.add_argument("models", nargs="+", help="v1 names (koda-phonics-NAME) or v2 *.onnx paths")
    ap.add_argument("--run", type=Path, default=RUN, help="koda-phonics run dir (staging/, output/)")
    ap.add_argument("--quiet-run", default="fn-20261031", help="dir under the runs parent with fired/ and blocked/ quiet records")
    ap.add_argument("--tv-run", default="fn-20261102-tv", help="dir under the runs parent with TV-on records")
    ap.add_argument("--exclude", default="0b2a8daa", help="substring of quiet record dirs to skip")
    ap.add_argument("--n-tts", type=int, default=1000, help="raw TTS positive_test clips to score")
    run(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
