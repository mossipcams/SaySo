# Plan: Promote living2 + mel verifier; drop failed mixes

Status: shipped. Pi loaded living2 `b840f51f` + verifier 0.445. Backup
`sayso.onnx.bak-03e612d8`. Recipe: `satellite/models/living2.yaml`.

## Scope

Ship the this-room LiveKit recipe that actually hears the Snowball:

1. Copy host `output-living2/sayso/sayso.onnx` (`b840f51f`) into
   `satellite/models/sayso.onnx`.
2. Check in `satellite/models/living2.yaml` (skip-generate mix + class
   prior) and `sayso-verifier.npz`.
3. Keep verifier wiring (`verifier.py`, LiveKit AND-gate, tests).
4. Slim `docs/HANDOFF_WAKE.md` and `satellite/models/README.md` to this
   operating point.
5. Delete failed-experiment wake-provider plans, `PLAN_LIVEKIT_LIVE*`, `PLAN_LIVEKIT_LIVING`,
   `PLAN_LIVEKIT_RETHINK`, `PLAN_LIVEKIT_EMBED`, `PLAN_WAKE_*`,
   `PLAN_PR81_CI.md`.

Do not commit wavs, npy, or `context.json`. Do not empty
`CUDA_VISIBLE_DEVICES`. Do not train the classifier on holdout, miner, the
89, or `verifier_live_fp`. Phrase **SaySo**.

## How living2 was trained (host `/home/ubuntu/sayso-wakeword`)

Skip `livekit.wakeword generate`. Features come from the living
skip-generate tree (same 50 / 90 / 89 wavs, renamed `clip_NNNNNN.wav`):

| Role | n | Host path under `sayso-wake-data/data/` |
| --- | ---: | --- |
| Train pos | 50 | `positive_recorded/` (no `mine_*`) |
| Train neg | 90 | `negative_living_iso` 23 + `negative_room` 43 + `negative_recorded` 24 |
| Trainer val pos | 10 | copies of the 50 |
| Trainer val neg | 89 | `negative_living_talk` (**val only**) |

Omit: `holdout_living`, `holdout_eval`, `negative_miner_party`,
`verifier_live_fp`, `negative_mined`, `negative_living_up`, generate/`from_list`
TTS.

living1 (`b070d8a9`, pos 16 / ACAV 256 / `max_negative_weight` 3000) was
deaf at 0.50 even on its own 50. living2 changes only the class prior so
0.50 can fire:

- `positive: 96`, `ACAV100M_sample: 64`, `max_negative_weight: 200`
- `conv_attention` / `small`, 20000 steps

```
python -m livekit.wakeword augment living2.yaml
python -m livekit.wakeword train    living2.yaml
python -m livekit.wakeword export   living2.yaml
```

Ignore trainer `optimal_threshold` (~0.05 / FPPH 261). Score at **0.50**.

living2 alone: SaySo 6/8, talk 0/8, overlap 1/89 (`neg_talk_038`), miner
0/74, **1/19** live FP (`163633_904` 0.91). living3 (more ACAV) and blend
(TTS+50) failed those gates. Do not dump FPs into the classifier.

Frozen Google speech embedding separates this-mic SaySo from the 89
(AUROC 0.97) but not from the 19 live FPs (probe on the 89 still fires
13/19; **mel AUROC 1.0**). Second stage is a logistic on frozen-mel
mean+std of the last-16-embedding mel union, fit on 50 recorded SaySo vs
the 19 `verifier_live_fp` windows only. Threshold **0.445**. Miner 74 and the
89 stay out of verifier train.

Fire iff `living2 ≥ 0.50` **and** `verifier ≥ 0.445`. Mine on the LiveKit
score before the veto.

## Verification

```
PYTHONPATH=satellite pytest -q \
  satellite/sayso/wake/test_livekit.py \
  satellite/sayso/wake/test_verifier.py \
  satellite/sayso/test_config.py \
  satellite/sayso/test_launcher.py
```

Confirm `satellite/models/sayso.onnx` md5 `b840f51f312abcd5b205e1fc1e32b2ed`
and `sayso-verifier.npz` md5 `0c632e778ca263e51c92d9ca95f451af`.

Host AND (already passed): SaySo 6/8, talk 0/8, overlap 0/89, miner 0/74,
verifier_live_fp 0/19.

## Result (staged)

| Suite | Result |
| --- | --- |
| living SaySo 8 | **6/8** (04/05 miss on living2 0.29/0.18; verifier still 0.93/0.98) |
| isolated talk 8 | **0/8** |
| overlap 89 | **0/89** (killed `neg_talk_038` without training on the 89) |
| miner 74 | **0/74** (held out of verifier train) |
| verifier_live_fp 19 | **0/19** (fingerprint of the 19; not an unbiased FP set) |

Pi journal 2026-09-20 17:40: Loaded LiveKit `b840f51f` + mel verifier 0.445.
Rollback: `sayso.onnx.bak-03e612d8` (best unbiased-FP backup).
