# Plan: Koda TV-on wakes via a logit ensemble (v9b + v10k + v10l)

**Date:** 2026-11-04
**Goal (user):** same threshold (0.65), fewer false wakes, more accurate wakes,
and wakes that work with the TV on.
**Source:** live test 2026-11-02 (7 TV-on attempts, 1 fired) and the v10a–v10q
runs in `/srv/llm/data/wake/runs/koda-phonics-v6-20261018/` on `llm`.

## Review findings

- **The miss is the model, not the hop.** With the TV playing, v9b scored the
  7 user-confirmed Kodas 0.14, 0.14, 0.15, 0.23, 0.46, 0.63, 0.97. Lowering
  the threshold to 0.45 catches 2 of 7. TV off, wakes score 0.80–0.97.
- **The real TV is loud.** TV-on captures have the TV at −18..−35 dBFS RMS
  (median −22), only 11–14 dB under the voice. Every other real Koda in the
  corpus was said in near-silence (~−85 dBFS).
- **Synthetic Koda over TV teaches "TV ⇒ Koda".** v10a/v10b/v10e/v10g/v10h
  (TTS Koda mixed over TV) all keep 1–41 false wakes/h on holdout TV at 0.65.
- **Real room Kodas over TV help, but alone they false-wake.** v10d/v10j2/v10k/
  v10l (the user's own Kodas mixed over training TV) raise real-voice-over-TV
  recall, but every single TV model gets 1–2.5 false wakes/h on holdout TV at
  0.65. More TV negatives (v10m), spool negatives (v10o), and dropping pitched
  rows (v10p) don't fix it; seeds move results a lot.
- **Logit averaging with v9b fixes the false wakes.** Mean of the three
  members' pre-sigmoid logits, then sigmoid, exported as one ONNX with the
  same interface (`embeddings [b,16,96] → score [b,1]`). Prob-averaging does
  not work.

| at 0.65 | v9b (deployed) | ensemble 9b+10k+10l |
| --- | --- | --- |
| Holdout TV false wakes | 0 | 0 (also 0 at 0.55) |
| LiveKit FPPH | 0.31 | 0.22 |
| Real TV-on, user (n=7) | 2/7 (expected 1.8) | 4/7 (expected 3.6) |
| Real room voice × holdout TV (n=132) | 41 | 57 |
| Synthetic clean (n=1000) | 869 | 908 |
| Quiet room, user (n=16) | 11/16 (expected 10.1) | 10/16 (expected 9.4) |
| CPU per window (VM) | 0.08 ms | 0.25 ms |

- **Open gap: the quiet room.** The one quiet loss is `0f51c839`, where v9b's
  0.72 is a single-hop spike and the TV members score ~0.1–0.2. With n=16 one
  clip is ±6%, so this is within seed noise, but "more accurate wakes" is not
  shown across the board yet. Only the live Pi test can settle it.
- v10q (fine-tune from v9b) finished training; its ensemble eval was
  interrupted and never ran. Optional, see Scope 4.
- The user ruled out new mining for this round. Evaluate with stored data only.

## Scope

1. **Bring the reusable pieces into the repo** (from the VM `staging/`, cleaned
   of hard-coded run paths; one-off `stage_v10*.py` drivers other than the
   final recipe stay on the VM):
   - `scripts/wake_make_ensemble.py`: build the logit-ensemble ONNX from member
     ONNX paths, with its self-check (ensemble output equals the mean of the
     member logits on random embeddings).
   - `scripts/wake_stage_tv.py`: the v10k staging recipe (room Kodas mixed over
     training TV at −30..−16 dBFS, plus the v10l variant's rows).
   - `scripts/wake_eval_realtv.py`: 33 locked-test real Kodas × holdout TV,
     including the new "tv −28..−18" band; plus the holdout FA/h check
     (`fa_holdout.py`: fast candidates, exact re-score).
   - One small `scripts/test_wake_make_ensemble.py`.
2. **Docs:** this plan; a "TV-on" section in `training/wake/README.md` with the
   recipe, the ensemble command, and the table above.
3. **Deploy (needs the user's OK):** copy
   `koda-phonics-lg_9b_10k_10l.onnx` (sha256 `5b5e7513…`) to
   `/opt/sayso-satellite/models/` on the Pi. The user edits
   `/etc/sayso-satellite/config.yaml` (root-owned) to point `wake_word.model`
   at it, keeping `threshold: 0.65`, and restarts the service. Keep the v9b
   file for rollback.
4. **Optional:** run the stored-data ensemble eval for v10q (9b+10q, 9b+10k+10q)
   and swap it in only if it beats the table on the quiet room without losing
   TV-on or FA.

Out of scope: threshold changes, lookback changes (see #49), new mining,
speaker checks, DMA-KWS.

## Verification

- `wake_make_ensemble.py` self-check passes, and the rebuilt ONNX matches sha
  `5b5e7513…` (or its scores match the VM copy to 1e-5).
- `wake_eval_realtv.py` on the repo copy reproduces the table's TV rows.
- Pi CPU: score time per 80 ms hop with the ensemble stays well under the hop
  (measure on the Pi before switching the config).
- **Live Pi test (user):** quiet room, TV on at normal volume, and an hour of
  TV with nobody speaking. Pass = TV-on wakes better than v9b's 1/7, quiet
  wakes no worse than v9b, 0 false wakes in the TV hour. Label the attempts
  into `wake-labelled/` before the miner evicts them.
- Fail → revert `wake_word.model` to `koda-phonics-v9b-20261102.onnx`.
