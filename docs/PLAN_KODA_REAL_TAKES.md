# Plan: Koda positives from real satellite takes, plus 1-3 m distance variants

**Date:** 2026-09-29
**Context:** `koda-full-20260929` trained (recall 0.836, 0.077 FP/h on synthetic
validation). It has never seen the real Pi mic, the room, or a far talker. The
mining spool on the Pi has a few real "Koda" windows.

## Policy change (deliberate)

`docs/SATELLITE_DATA_COLLECTION.md` keeps mined windows out of train. The user
has now asked for the few confirmed real Koda takes to seed training data. Scope
of the exception: only windows that are **listen-verified as Koda** enter
train, as *seeds*. Everything else in the spool stays out. Verified takes are
split first: about a third are held out for eval and never seeded.

## Scope

1. **Select seeds.** Spool pulled to `~/koda-spool`. Run
   `scripts/wake_mine_check.py SPOOL --phrase Koda`. Classes `wake` and
   `missed_wake` are candidates. Listen-verify each; write `seeds.json`
   (capture_id, sha256, train|holdout). Windows are 2 s, 16 kHz, +6 dB mic gain.
2. **Seed variants (near-field, "at the mic").** From each train seed, cut the
   speech region and make N variants: gain +-6 dB, speed 0.9-1.1, small pitch
   shift, background noise at 5-25 dB SNR (LiveKit `data/backgrounds`), placed at
   random offsets so the word ends in the last 0.5 s of the 2 s frame like the
   streaming model expects.
3. **Distance variants, 1 m / 2 m / 3 m.** Split each RIR in
   `data/rirs/16khz` at its direct-path peak (+2.5 ms). Scale the direct part by
   `r0/d` (`r0` = 0.3 m mic reference, so about -10 / -16 / -20 dB) and keep the
   reverberant tail, so the direct-to-reverb ratio falls with distance as in a
   real room. Convolve the seed, then add noise so far-field SNR drops with
   distance. Output is 2 s / 16 kHz, same as the mic path.
   The same transform is applied to the LiveKit TTS Koda positives so the
   far-field class is not only a handful of seeds.
4. **Inject.** Write variants as `clip_*.wav` into the run's
   `output/koda/positive_train/` (seeds' holdout takes never go here), then run
   `augment` -> `train` -> `export` -> `eval` from the existing
   `wake_livekit_run.py`. New run dir `koda-real-<date>`, `koda-full-20260929`
   stays untouched as baseline.
5. **Code.** One new script `scripts/wake_koda_seeds.py`: `select`, `variants`
   (near + distance), with a colocated `scripts/test_wake_koda_seeds.py`
   (direct/reverb split scaling, level drop per distance, word-position, holdout
   never emitted). Change `wake_mine_check.py` default `--phrase` to `Koda`.
6. **Same-day real far-field check.** Record ~20 real takes each at 1, 2, 3 m on
   the Pi mining path (unit running), with labels in the filename. They are eval
   only, and the only real proof the simulated distance works.

## Files to touch

- `scripts/wake_koda_seeds.py` (new), `scripts/test_wake_koda_seeds.py` (new)
- `scripts/wake_mine_check.py` (phrase default), `training/wake/README.md`
- VM only: `/srv/llm/wake/runs/koda-real-<date>/`

## Verification

- `python3 -m pytest scripts/test_wake_koda_seeds.py scripts/test_wake_mine_check.py`
- Compare `koda-full-20260929` vs new run on: held-out real Koda takes (recall),
  the real 1/2/3 m takes (recall per distance), the 200 below-threshold and
  127 near-threshold spool windows verified negative (fires), and LiveKit
  `koda_eval.json` (no regression in FP/h).
- Ship only if far-field recall improves and FP/h does not.

## Out of scope

Voice cloning / TTS from the seeds, LFM data, Atlas/SaySo data, Pi model swap.
Training needs `gpu train wake`; check `gpu status` first (GPU is shared).

## Round 2 (2026-09-30): hard negatives

Round 1 (`koda-real-20260929`) is on the Pi. It fixed near/far recall but cost
synthetic recall/FP (0.787 / 0.116 vs 0.836 / 0.077) and gave 4 vs 3 fires on 350
non-Koda spool windows. Round 2 adds the spool windows whose transcript is not
Koda-like as hard negatives (`wake_koda_seeds.py negatives`, 4/5 train, 1/5
held out by hash), on top of the round 1 positives, in `koda-real2-20260930`.
Success: fewer held-out negative fires and FP/h at or below baseline with
held-out Koda recall not lower. Otherwise keep round 1.
