Place this LiveKit-exported Sayso classifier on the satellite:

  /opt/sayso-satellite/models/sayso.onnx

It detects the spoken phrase "Sayso" only. The generate-first model runs
**single-stage LiveKit** by default — omit `wake_word.verifier`. Do not use the
trainer `optimal_threshold` (~0.05). Do not substitute hey_livekit, hey_jarvis,
or another model.

Shipped on the Pi today: **`livekit-corpus-hn-v1`** (`0a3260c8`) at live threshold
**0.25** (`mine_threshold` **0.12**), single-stage LiveKit (no mel verifier).
Mic path: EMEET OfficeCore M0 Plus, `capture_rate` **16000**, `mic_gain_db`
**6.0** — see `satellite/config.yaml` and `docs/SATELLITE_DATA_COLLECTION.md`.
Pulse sink volume **90%** is set outside yaml (`pactl`).

Historical: living2 (`b840f51f312abcd5b205e1fc1e32b2ed`) at **0.28** with
legacy `sayso-verifier.npz` (`0c632e778ca263e51c92d9ca95f451af`).

## Next train (LiveKit generate-first)

`sayso.yaml` is the primary training contract for the next model. It follows
LiveKit `configs/prod.yaml` with SaySo-only deltas (phrase spelling variants,
homophone custom negatives, `conv_attention`/`small` for Pi inference). **Do
not skip generate.** Do not train the next model from the 50 living2 clips
alone.

Pipeline (host only, isolated work dir — never write into `output-living2/`):

```bash
pip install -r satellite/models/requirements-wake-train.txt
# system: espeak-ng ffmpeg (and sox on Linux)
python3 scripts/wake_livekit_run.py run \
  --config satellite/models/sayso.yaml \
  --work-dir /home/ubuntu/sayso-wakeword/runs/livekit-restart
```

Stages: `setup` → `generate` (Piper TTS) → `augment` → `train` → `export` →
`eval`. Smoke wiring: `sayso-smoke.yaml` + a temp `--work-dir`.

Do not run this on the Pi. Use a CUDA host. Do not set
`CUDA_VISIBLE_DEVICES` to empty. Do not stop LFM2 on the train host.

Real Snowball / miner audio stays eval and optional later overlay — not the
primary positive class for this restart. Holdouts (`holdout_living`,
`holdout_eval`, miner party 74, `verifier_live_fp` 19) stay out of training.
`200-positive` at `/home/ubuntu/sayso-wake-data/data/200-positive/` is clean
recorded data — keep it; `runs/pos200-*` were disposable workspaces.

## Shipped model (living2 — historical recipe)

`living2.yaml` documents the **skip-generate** recipe that produced the prior
Pi ONNX (replaced by `livekit-corpus-hn-v1`). It is not the current operating
point. The mix
is 50 this-room Snowball positives, 90 this-room train negatives, and 89
overlapping-talk clips as **val only**.

living1 used the same wavs with `positive: 16` / `ACAV100M_sample: 256` /
`max_negative_weight: 3000` and never fired at 0.50. living2 only changes
the class prior (`positive: 96`, `ACAV100M_sample: 64`,
`max_negative_weight: 200`) so 0.50 can fire. Mixing extra TTS positives
(blend) or more ACAV (living3) failed the this-room gates — do not repeat
those.

To reproduce the shipped artifact (not recommended for the next model):

```bash
pip install -r satellite/models/requirements-wake-train.txt
python -m livekit.wakeword augment living2.yaml
python -m livekit.wakeword train    living2.yaml
python -m livekit.wakeword export   living2.yaml
```

Ship `output-living2/sayso/sayso.onnx` back into this directory. Historical
host recipe scored at **0.50** with the mel verifier below (living2 only).

`sayso-training.yaml` is a deprecated VoxCPM experiment; use `sayso.yaml`.

## Mining / snapshot path (secondary)

Prefer the batch wrapper when retraining from the mining spool or corpus
snapshots (labeling and idempotency — not the primary LiveKit generate path):

```bash
python3 scripts/wake_train.py \
  --spool /var/lib/sayso-satellite/wake-mining \
  --seed-dir satellite/models/data/seed \
  --work-dir satellite/models/output/wake_runs
```

Source splits and holdout rules live in `satellite/eval/splits.json`. Holdouts
must not leak into training, calibration, background mixing, or derived features.
Freeze both the deployed threshold and the calibration-selected threshold in
`satellite/eval/baseline.json` before a later retrain.

## Optional embedding verifier (second stage)

The generate-first model already classifies from (16, 96) speech embeddings.
A compatible second stage must score those same embeddings — not a parallel
collapsed-mel pass.

Legacy `sayso-verifier.npz` (`feature_kind` absent, treated as `mel_union`) is
a this-room recording-envelope logistic on mel mean+std, fit on 50 Snowball
clips vs 19 `verifier_live_fp` windows. It is **not** a phrase check for the
generate-first export and does **not** AND-gate LiveKit when configured; the
satellite logs once and runs single-stage.

A future compatible artifact sets `feature_kind=speech_embedding` in the npz and
carries logistic weights for concat(mean, std) over the last 16 speech
embeddings (192-d). None is shipped yet — do not invent one in this tree.

When `wake_word.verifier` points at a compatible npz, LiveKit remains the
primary scorer and the embedding verifier must also pass before the satellite
fires. Mine on the LiveKit score as today — the veto runs after mining, not
before. Omit `verifier` for single-stage LiveKit (generate-first default). If
`verifier` is set but the file is missing, wake detection fails closed.

### living2 + mel verifier (historical Pi operating point)

Frozen Google speech embeddings separate this-mic SaySo from overlapping talk
(AUROC 0.97) but not from the 19 live false wakes. The mel logistic fit on
50 vs 19 achieved mel AUROC 1.0 on that narrow set. Do not dump those FPs
into the classifier.

Host AND-gate (hop-scan at living2 **0.50** / mel verifier **0.445**):

| Set | Result |
| --- | ---: |
| living-room SaySo | **6/8** |
| isolated talk | **0/8** |
| overlapping talk (89) | **0/89** |
| miner party (74) | **0/74** |
| verifier_live_fp (19) | **0/19** |

Pi operating point (living2 **0.28** / mel verifier **0.445**):

| Set | Result |
| --- | ---: |
| living-room SaySo | **7/8** |

Miss on Pi: `live_sayso_05` = 0.179 collides with `live_talk_02` = 0.178
(verifier blesses both). 8/8 is blocked by that collision.

The 19 are verifier-train, not an unbiased FP set. Best unbiased-FP backup
on the Pi is `sayso.onnx.bak-03e612d8`.

## Bootstrap blockers

- `../eval/audio/` has no trusted fixtures. `cases.json` skips every case in
  default mode and fails in `--strict`. Populate with real EMEET (or historical
  Snowball) recordings from the living room before baseline freeze, qualification, or
  enabling `--schedule` on `scripts/wake_train.py`.
- Homophone negatives in `sayso.yaml` remain phonetic inference until the
  mining spool supplies labelled real confusions. Run
  `python scripts/wake_mine_report.py <mine_dir> --inventory` to audit wake
  assets before cleanup.
