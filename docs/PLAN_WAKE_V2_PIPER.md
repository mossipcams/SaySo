# Plan: Koda wake experiments round 3 (v10u/v10v/v10x) and livekit-wakeword v2 on local TTS

**Date:** 2026-11-04
**Goal (user):** fewer false wakes and better Koda detection under TV. Run more
experiments, compare against the deployed ensemble (`lg_10t2_10k_10l`, threshold
0.65), keep iterating. Try livekit-wakeword v2 without a LiveKit Cloud account.
**Runs:** `/srv/llm/data/wake/runs/koda-phonics-v6-20261018/` (`run_x.sh`,
`run_x2.sh`, `chain.sh`) and `/srv/llm/data/wake/lk2/` on `llm`.

## Findings so far

- **Ensemble reshuffles are exhausted.** 4- and 5-member logit ensembles
  (10t+10t2+10k+10l, with or without v9b) don't beat 10t2+10k+10l. Adding v9b
  costs TV-on wakes.
- **The quiet-room set (n=16) has a ceiling of about 10-11.** Whisper hears no
  "Koda" in 5 of the 6 records every model misses. 0e088598 is a real Koda that
  scores 0.01.
- **Hop phase matters.** Scored at 20 ms, the deployed ensemble catches 5/7 TV-on
  Kodas, against 4.25 expected at the 80 ms hop. On real voice over TV
  (-38..-28 dBFS) a 40 ms hop gives +2.7/66.
- **The false-wake margin is thin.** The deployed ensemble's top score on the
  holdout TV at a 40 ms hop is 0.62. At 0.55 it fires 1/h. So neither a finer
  hop nor a lower threshold is free.
- **v10v (shift-jittered real Kodas) wins on recall.** Its ensemble
  10v+10k+10l at 0.65 gets real voice x TV 116/106/85 vs 108/100/65 and TV-on
  4.94 vs 4.25, with holdout FA 0. The cost is one real quiet-room "Hello." at
  0.67. At 0.75 it rejects that and matches the deployed recall.
- **livekit-wakeword v2 needs the account only for TTS.** The check for
  `LIVEKIT_URL`/`LIVEKIT_API_KEY`/`LIVEKIT_API_SECRET` sits in `setup` and
  guards `livekit.agents.inference.TTS`. The encoders are a public GitHub
  release asset, and the negatives, noise and RIRs are public Hugging Face
  datasets. A shim swaps bench's `Synthesizer.synthesize`, `load_voices` and
  `speed_kwargs` for local Piper. It also forces plain text, because bench's
  phoneme markup targets the cloud TTS. And it disables bench's
  auto-confusables, which split "Koda" into "ko da". Smoke data generation
  works with no account.

## Scope (this commit)

Port the reusable tools from the VM into `scripts/`, matching the existing
`wake_*.py` style:

1. `scripts/wake_stage_shift.py`: v10v staging. It takes real Koda windows,
   shifts each window left by d ~ U(10, 160) ms, and writes 2 clean and 2
   over-train-TV variants per clip (-30..-16 dBFS). Output is a features
   `.npy` plus a manifest.
2. `scripts/wake_mine_tv_hard.py`: v10u. It mines the train TV hours with a
   scorer and keeps the top window per 1 s with score >= MIN, up to CAP rows.
   It re-featurises those rows exactly.
3. `scripts/wake_eval_audio.py`: audio-level eval for both model kinds on the
   same audio. v1 heads use 16x96 windows; v2 streaming ONNX uses 1280-sample
   chunks plus state. It covers real voice x holdout TV bands, TV-on and quiet
   records at 16 phases, raw TTS clips, and holdout TV FA/h at 80 and 40 ms
   hops.
4. `scripts/wake_lk2_piper.py`: the shim that points livekit-wakeword-bench at
   local Piper (no LiveKit Cloud).
5. `scripts/wake_lk2_setup.py` and `scripts/wake_lk2_train.py`: v2
   `run_setup`/`run_train` drivers for "Koda". They force plain text, take the
   hard negatives from a recipe YAML, and use dummy `LIVEKIT_*` env vars that
   are never sent.
6. `training/wake/README.md`: a short "livekit-wakeword v2 on local TTS"
   section with the PYTHONPATH layout. In the setup process, v2's
   `wakeword_train` must not shadow 0.2.1's `livekit.wakeword`, which carries
   Piper. In the training process v2 comes first.
7. This plan.

Out of scope: the VM run scripts (`run_x*.sh`, `chain.sh`), the one-off
diagnostics (`hop_gap.py`, `fa_hop40.py`, superseded by `wake_eval_audio.py`),
satellite runtime changes, deploys, and threshold changes.

## Results (2026-11-05)

All runs use the v10t2 recipe (seed 7) with one change. Each ensemble is the
new model + v10k + v10l. Thresholds are 0.65. FA is on the 2 h holdout TV at
the 80 ms hop.

| | deployed 10t2 | **10u** | 10v | 10w (10u+10v data) | 10u+10v (4 members) | v2a (v2, Piper) | v2b (v2, +VoxCPM2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voice x TV -48..-38 / -38..-28 / -28..-18 (n=132 each) | 108 / 100 / 65 | **117 / 104 / 70** | 116 / 106 / 85 | 119 / 102 / 81 | 115 / 104 / 74 | 116 / 106 / 89 | lower |
| TV-on, expected (n=7) | 4.25 | **4.56** | 4.94 | 5.25 | 4.62 | 5.25 | 2.81 |
| Quiet room, expected (n=16) | 9.19 | **9.38** | 9.44 | 9.31 | 9.62 | 8.94 | 4.25 |
| FA/h at 0.35 / 0.45 / 0.55 / 0.65 | 2 / 1 / 0 / 0 | **0 / 0 / 0 / 0** | 2.5 / 2 / 0.5 / 0 | 1 / 0 / 0 / 0 | 0.5 / 0.5 / 0 / 0 | 19 / 10.5 / 5 / 2 | 19 / 10 / 5 / 2 |
| Quiet-room "Hello." (f5a1d96e) score | 0.52 | 0.54 | 0.67 (fires) | 0.69 (fires) | - | - | - |

- **v10u wins on every row and is deployed.** It adds 1264 hard negatives
  mined from the train TV hours (`scripts/wake_mine_tv_hard.py`).
- **v10v and v10w buy loud-TV recall at the cost of the "Hello." fire.** The
  shift-jittered positives cause it. At 0.75 the fire goes away, but recall
  falls back to the deployed level.
- **v10x** (TTS-over-TV v2 staging) ties the deployed ensemble.
- **livekit-wakeword v2 (v2a) has strong recall** but 2 FA/h on our TV, with
  top holdout score 0.83. It saw only 1.6 h of generated data and none of our
  TV.
- **v2b is worse.** It adds VoxCPM2 voices to about 30% of the voice pool:
  LibriSpeech and user clones plus voice design, with a CTC gate keeping only
  /k o d a/ readings. The likely cause is that the user-voice clones also
  speak hard negatives ("Kawda", "Kowda"), which match how the user actually
  says Koda. Leave user clones out of negatives if v2 is retried.

## Deployed (2026-11-05, 5:21 AM CT)

`/opt/sayso-satellite/models/koda-phonics-ens-10u-10k-10l-20261105.onnx`
(sha256 `e0e7268f…`, v10u + v10k + v10l), threshold 0.65. It takes about 7 ms
per window on the Pi. Config backup:
`/etc/sayso-satellite/config.yaml.bak-pre-ens-10u-20261105`.

Rollback: point `wake_word.model` back at
`koda-phonics-ens-10t2-10k-10l-20261104.onnx` and restart `sayso-satellite`.

## Next

- Live Pi test: Kodas with the TV on, Kodas in a quiet room, and 1 h of TV
  with nobody speaking.
- v2 needs our TV hours as negatives before it can compete on false wakes.

## Verification

- `pytest -q` on any new `scripts/test_wake_*.py`.
- CI gate: `ruff check .`, `ruff format --check .`,
  `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`,
  `pytest -q satellite/sayso`.
- `python3 -m py_compile` on each ported script. The scripts need the VM
  (livekit-wakeword 0.2.1/v2, data paths), so their runtime check is the VM
  runs recorded above.
