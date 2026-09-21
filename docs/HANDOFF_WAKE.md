# Handoff: living-room wake word

Phrase is **SaySo**, not “say so”. Wavs are off git. Do not commit wavs,
`context.json`, or passwords.

Satellite is a thin LVA overlay. It does not do STT, NLU, or actions.

## Live

| Machine | Role |
| --- | --- |
| Satellite `192.168.1.54` (`pi`, SSH port **2222**) | Capture, live wake, miner |
| Train host `192.168.1.140` (`ubuntu`) | `/home/ubuntu/sayso-wakeword/` (LiveKit) and `/home/ubuntu/sayso-wake-data/` (Snowball wav data) |
| Worktree | wake-data docs on this branch |

**Loaded now** (Pi):

- `provider: livekit`
- `/opt/sayso-satellite/models/sayso.onnx` living2
  (`b840f51f312abcd5b205e1fc1e32b2ed`)
- `/opt/sayso-satellite/models/sayso-verifier.npz`
  (`0c632e778ca263e51c92d9ca95f451af`) threshold **0.445**
- Fire iff living2 ≥ **0.28** and verifier ≥ **0.445**
- `mine_threshold: 0.25`, refractory **2.0 s**
- miner: `/var/lib/sayso-satellite/wake-mining`
- Rollback backup: `sayso.onnx.bak-03e612d8`
  (`03e612d8671df941bd63c5a982d37ad4`)

Recipe: `satellite/models/living2.yaml`. Ship record:
`docs/PLAN_LIVEKIT_VERIFIER.md`.

## Recipe (living2)

Skip `livekit.wakeword generate`.

| Role | n | Source |
| --- | ---: | --- |
| Train pos | 50 | `positive_recorded/` |
| Train neg | 90 | iso 23 + room 43 + recorded 24 |
| Val neg | 89 | overlapping talk (**val only**) |

Class prior: positive **96**, ACAV **64**, `max_negative_weight` **200**.
Ignore trainer `optimal_threshold` (~0.04–0.07).

**Never train the classifier on:** `holdout_living`, `holdout_eval`, miner
party 74, `verifier_live_fp` 19, or unlabeled spool clips.

**Verifier only:** 50 recorded SaySo vs 19 `verifier_live_fp`. Miner 74 and the
89 stay out of that fit.

Mine on the LiveKit score **before** the verifier veto.

## Gates

Host AND-gate (hop-scan at living2 **0.50** / verifier **0.445**):

| Set | Result |
| --- | ---: |
| living-room SaySo | **6/8** |
| isolated talk | **0/8** |
| overlapping talk (89) | **0/89** |
| miner party (74) | **0/74** |
| verifier_live_fp (19) | **0/19** |

Pi operating point (living2 **0.28** / verifier **0.445**):

| Set | Result |
| --- | ---: |
| living-room SaySo | **7/8** |

Miss: `live_sayso_05` = 0.179 on `live_talk_02` = 0.178 (verifier blesses
both). **8/8** is blocked by that collision.

## Failed levers (do not repeat)

- Live false wakes from the removed third-party provider
- living1 deaf class prior (pos 16 / ACAV 256 / `max_negative_weight` 3000)
- Blend TTS positives from `from_list`
- living3 more ACAV
- living4 +20 quiet speakers (5/8 SaySo + verifier-blessed FPs)
- living5 iso-val / no `max_negative_weight` doubling (swapped 05 vs 02/04)
- living6 +357 WavLM-labelled mine negs (01 0.896→0.258, 05 lifts, miner
  ceiling 0.274 blocks 8/8)
- Verifier v2 probes cannot split 05 / `live_talk_02`
- Unlabeled auto-retrain from the spool
- WavLM on the Pi
- Dumping holdout / 89 / miner 74 / `verifier_live_fp` into the classifier
- Using 89 as val with `target_fpph` 0.02 (doubles `max_negative_weight`)

## What works

- living2 class prior (pos 96 / ACAV 64 / `max_negative_weight` 200)
- AND-gate (LiveKit primary + mel verifier second stage)
- Lower living2 threshold to **0.28** for 7/8 on the Pi
- Human `--label` on the mining spool (`wake_mine_report.py`)
- New this-room SaySo recordings that are **not** in the holdout if 8/8 is
  required
- `wake_train.py --schedule` stays **disabled** (empty eval audio)

## Ops

- Do not empty `CUDA_VISIBLE_DEVICES`.
- Do not stop LFM2 on the train host.
- Do not pipe yaml into `sudo -S`.
- Pi has no `sftp-server` — use `tar` / `cat` for file transfer.
- Do not record more unless asked.
- Do not treat prompted talk 0/8 as “FPs are fixed.”

## Data (host, never git)

**Available, not in living2**

| Set | n | Path |
| --- | ---: | --- |
| `200-positive` | 200 | `sayso-wake-data/data/200-positive/` (16 kHz / 2 s; raw 48 kHz in `raw/`) |

Prompted living-room SaySo from 2026-09-21. Do not merge into the living2 50 until a retrain plan says so.

**Never train on**

| Set | n | Path |
| --- | ---: | --- |
| `holdout_living` | 23 | `sayso-wake-data/data/holdout_living/` |
| `holdout_eval` | 21 | `sayso-wake-data/data/holdout_eval/` |
| Miner party | 74 | `sayso-wake-data/data/negative_miner_party/` |
| Unsure miner | — | Pi spool |

**living2 classifier**

| Role | n | Path |
| --- | ---: | --- |
| Train pos | 50 | `positive_recorded/` |
| Train neg | 90 | `negative_living_iso` + `negative_room` + `negative_recorded` |
| Val neg | 89 | `negative_living_talk` (val only) |

**Verifier only** (not the classifier)

| Set | n | Path |
| --- | ---: | --- |
| `verifier_live_fp` | 19 | `sayso-wake-data/data/verifier_live_fp/` |

LiveKit uses 2 s windows. Isolated `live_talk_*` is not the FP test that
matched the live complaint; overlapping 89 and miner 74 are. The 19 live
FPs are verifier-train only.
