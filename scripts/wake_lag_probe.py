"""Locate the wake phrase end with the classifier instead of an energy proxy.

A mined clip is the exact 2 s window that fired, ending at detection_index.
Cut k ms off the tail and pad the front with silence to keep the window 2 s,
then re-score. While the cut removes only post-phrase audio the score holds;
once it eats into the phrase the score collapses. The largest k that holds is
the detection lag.
"""
import glob, json, os, sys, wave
import numpy as np
sys.path.insert(0, "/opt/sayso-satellite")
from sayso.wake.streaming import single_threaded_ort
from livekit.wakeword import WakeWordModel

SR = 16000
path = sys.argv[1]
with single_threaded_ort():
    model = WakeWordModel(models=[path])
key = os.path.splitext(os.path.basename(path))[0]

def score(x):
    s = model.predict(np.clip(x, -32768, 32767).astype("<i2"))
    return float(s.get(key, next(iter(s.values())) if s else 0.0))

print(f"{'clip':<34} {'base':>6} {'lag':>6}  score by tail cut 0..600ms step 40")
lags = []
for p in sorted(glob.glob("/var/lib/sayso-satellite/wake-mining/*.wav")):
    m = json.load(open(p[:-4] + ".json"))
    if not m.get("fired"):
        continue
    with wave.open(p) as w:
        d = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    base = score(d)
    if base < 0.5:
        print(f"{os.path.basename(p)[:-4]:<34} {base:>6.3f}  (stateless rescore below threshold, skip)")
        continue
    curve, lag = [], 0
    for cut in range(0, 640, 40):
        k = cut * SR // 1000
        x = np.concatenate([np.zeros(k), d[: len(d) - k]]) if k else d.copy()
        s = score(x)
        curve.append(s)
        if s >= 0.5 * base:
            lag = cut
        else:
            break
    lags.append(lag)
    print(f"{os.path.basename(p)[:-4]:<34} {base:>6.3f} {lag:>5}  " + " ".join(f"{c:.2f}" for c in curve))

if lags:
    lo, hi = max(lags), min(lags) + 100
    print(f"\nn={len(lags)}  lags={sorted(lags)}  min={min(lags)}  max={max(lags)}")
    print(f"feasible fixed window = [{lo}, {hi}] ms -> " + ("EMPTY" if lo > hi else f"{hi-lo+1} ms wide"))
