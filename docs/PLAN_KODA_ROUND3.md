# Plan: Koda round-3 retrain — spool hard negatives + fresh real takes

**Date:** 2026-10-02
**Context:** `koda.onnx` (run `koda-real2-20260930`, deployed 2026-09-30,
threshold 0.40) has produced **4 fires in 2 days, all false positives, zero
real Koda wakes**.

## Evidence (measured 2026-10-02)

Windows pulled from the Pi mining spool were re-scored locally through the
exact `WakeWordModel.predict` path (local scores reproduce live scores to the
decimal) and STT'd (faster-whisper small) on the trigger audio itself:

| Score | Trigger audio (whisper) | Fired |
|------:|--------------------------|:-----:|
| 0.498 | "I want to" | yes (FP) |
| 0.463 | (silence) | yes (FP) |
| 0.429 | "Thank you." | yes (FP) |
| 0.419 | "Thank you." | yes (FP) |
| **0.400** | **"Coda." = real KODA** | **no (missed)** |
| 0.398 | "So, um…" | no |
| 0.392 | "All right." | no |
| 0.373 | "Hey world!" | no |
| 0.372 | "Okay. We've got more." | no |

**Diagnosis: no separation.** A clean real Koda scores 0.400 — dead center of
the false-positive cluster (0.37–0.50). Ordinary speech and even silence score
as high as a genuine wake word. No threshold fixes this: raising it to kill
the FPs also kills real Koda. Round 2's synthetic eval (recall 0.77, 0.04
FP/h) was optimistic, as `training/wake/README.md` warns.

Round history:

- Round 1 `koda-full-20260929`: synthetic recall 0.836, 0.077 FP/h; 10 FPs in
  one evening on the real mic (TV + command speech).
- Round 2 `koda-real2-20260930`: 25 000/5 000 synth + 2 000/500 background,
  185 seed variants (15 near-field, 10 @1 m, 30 @2 m, 30 @3 m) + 1 750 spool
  negatives (350 windows × 5). Synthetic recall 0.77, 0.04 FP/h. Live: 4 FPs
  in 2 days, 0 real hits.

## Goal

A `koda-real3` model that, on the **real mic in this room**:

1. fires on real Koda takes (near-field and 1–3 m), and
2. does not fire on the mined spool of room speech, TV, and silence.

Ship only if both hold on held-out data.

## Scope

- **Repo:** this plan doc only. No code changes expected.
- **VM (`llm`, 192.168.1.76):** new run dir, spool data, retrain, export, eval.
- **Pi (192.168.1.54):** spool pull, model deploy, service restart.
- **User:** one recording session (~20 real Koda takes) and listen-verification
  of a handful of mined clips.

Out of scope: threshold tuning as a fix, model-size changes, satellite code
changes, new phrases.

## Steps

### 1. Pull the spool

```bash
rsync -a pi@192.168.1.54:/var/lib/sayso-satellite/wake-mining/ SPOOL/
```

(~439 records since 2026-09-27; scp is broken on the Pi, rsync works.)

### 2. Classify the spool

```bash
uv run --no-project --with faster-whisper --with numpy \
  python scripts/wake_mine_check.py SPOOL --phrase Koda --export OUT
```

ASR class proposals only. **Listen-verify every `wake` / `missed_wake` clip
before it is used as a positive.** Expected finds: the 0.400 "Coda." window
(`eec5767b`) plus any other real Koda among near-threshold windows.

### 3. New hard negatives from the spool

```bash
python3 scripts/wake_koda_seeds.py negatives SPOOL --out OUT_NEG -n 5
```

Uses train-split non-Koda windows (round 2 used 350; this spool has ~410
non-fired records). These are the exact audio the current model fires on:
"Thank you", "I want to", "All right", silence.

### 4. New verified real Koda seeds + distance variants

```bash
python3 scripts/wake_koda_seeds.py select SPOOL --out seeds.json
# listen, then set "verified": true on real Koda takes
python3 scripts/wake_koda_seeds.py variants SPOOL --seeds seeds.json \
  --rirs data/rirs/16khz --backgrounds data/backgrounds --out OUT_POS -n 50
```

### 5. Record fresh real Koda takes (user)

Per `docs/PLAN_KODA_REAL_TAKES.md`: ~20 takes through the satellite mic path —
5 near-field, 5 each at 1, 2, 3 m. Normal pace, unstaged, no retakes. Third of
them held out for eval, never emitted as training data.

### 6. Retrain on the VM

Run dir `/srv/llm/wake/runs/koda-real3-<date>` with the round-2 `koda.yaml`
(25 000/5 000 synth, 2 000/500 background). Before `augment`:

- `output/koda/positive_train/`: round-2 positives + `OUT_POS` variants +
  verified spool seeds
- `output/koda/negative_train/`: round-2 spool negatives + `OUT_NEG`

Standard pipeline under the GPU lock (`/srv/llm/bin/gpu train wake
--serve-after`): setup → generate → augment → train → export → eval.

### 7. Eval gates — ship only if ALL pass

Eval on the **same held-out set** for round 2 and round 3:

1. **Real-take recall:** every held-out real Koda take (step 5) scores above
   0.40; recall ≥ round 2's on the same takes.
2. **Spool negatives:** 0 fires at 0.40 across round-2 + round-3 spool
   negative windows (held-out splits).
3. **Separation margin:** min real-take score − max spool-FP score ≥ 0.10.
   Round 2's margin was negative (0.400 real vs 0.498 FP); that is the number
   this round must fix.
4. **Synthetic guard:** recall ≥ 0.80 and FP/h ≤ 0.05 (regression check only;
   synthetic is not the shipping signal).

### 8. Deploy and monitor

- Publish `koda.onnx` to the Pi, restart `sayso-satellite.service`.
- Monitor the mining spool for 24–48 h: real wakes must appear in fired
  records; any new FP pattern goes back to step 3.

## Optional stopgap (independent of retrain)

Raise the live threshold on the Pi to ~0.55 while round 3 is in progress.
This silences the FPs but also silences real Koda (0.400 < 0.55) — it buys
quiet, not functionality. Revert on deploy.

## Files to touch

| Where | Path | Change |
|---|---|---|
| repo | `docs/PLAN_KODA_ROUND3.md` | this doc (new) |
| VM | `/srv/llm/wake/runs/koda-real3-*/` | new run dir |
| VM | spool + seed data under the run dir | new data |
| Pi | `/opt/sayso-satellite/models/koda.onnx` | replace on deploy |
| Pi | `/etc/sayso-satellite/config.yaml` | optional stopgap only |

No repo code, config, or satellite files change.

## Verification

- Plan doc: sections above present; every command matches the current CLI of
  `scripts/wake_mine_check.py` / `scripts/wake_koda_seeds.py` /
  `training/wake/README.md`.
- Retrained model: the four eval gates in step 7, run on held-out data, with
  scores recorded in the run dir before any Pi deploy.
- Post-deploy: 24–48 h spool review (step 8).

## Risks / stop conditions

- If step 7 gate 3 (separation margin) fails again, stop: the `small` model
  with LiveKit mel features may not have enough capacity for this room. Next
  options are a larger model, more/broader negatives, or a different detector
  approach — decide then, don't ship.
- If the user cannot record step 5, round 3 has no real-take eval set and the
  gates are unmeasurable — do not train.
