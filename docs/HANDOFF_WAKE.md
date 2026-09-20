# Handoff: living-room wake word

Phrase is **SaySo**, not “say so”. Wavs are off git. Do not train the
classifier on holdout dirs, miner fires, or `nano_live_fp`. Do not empty
`CUDA_VISIBLE_DEVICES`. Do not put passwords in the repo. Do not commit
wavs or `context.json`.

Satellite is a thin LVA overlay. It does not do STT, NLU, or actions.

## Live

| Machine | Role |
| --- | --- |
| Satellite `192.168.1.54` (`pi`, SSH port **2222**) | Capture, live wake, miner |
| Train host `192.168.1.140` (`ubuntu`) | `/home/ubuntu/sayso-wakeword/` (LiveKit) and `/home/ubuntu/sayso-nanowakeword/` (Snowball wav data) |
| Worktree | `ajax-nanowakeword` (branch `ajax/nanowakeword`) |

**Loaded now** (Pi, 2026-09-20 17:40):

- `provider: livekit`
- `/opt/sayso-satellite/models/sayso.onnx` living2
  (`b840f51f312abcd5b205e1fc1e32b2ed`)
- `/opt/sayso-satellite/models/sayso-verifier.npz` threshold **0.445**
- Fire iff living2 ≥ **0.50** and verifier ≥ **0.445**
- `mine_threshold: 0.45`, refractory `2.0 s`
- miner: `/var/lib/sayso-satellite/wake-mining`
- Rollback backup: `sayso.onnx.bak-03e612d8`
  (`03e612d8671df941bd63c5a982d37ad4`)

How we trained it: `satellite/models/living2.yaml` and
`satellite/models/README.md`. Ship record: `docs/PLAN_LIVEKIT_VERIFIER.md`.

## Do not

- Record more unless asked.
- Train the classifier on holdout / miner / `nano_live_fp` / the 89.
- Dump conversational FPs into the main LiveKit mix (living3 / blend).
- Stop LFM2 on the train host. Empty `CUDA_VISIBLE_DEVICES`.
- Commit wavs or `context.json`.
- Treat prompted talk 0/8 as “FPs are fixed.”

## Data (host, never git)

**Never train on**

| Set | n | Path |
| --- | ---: | --- |
| `holdout_living` | 23 | `sayso-nanowakeword/data/holdout_living/` |
| `holdout_eval` | 21 | `sayso-nanowakeword/data/holdout_eval/` |
| Miner party | 74 | `sayso-nanowakeword/data/negative_miner_party/` |
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
| `nano_live_fp` | 19 | `sayso-nanowakeword/data/nano_live_fp/` |

LiveKit uses 2 s windows. Isolated `live_talk_*` is not the FP test that
matched the live complaint; overlapping 89 and miner 74 are. The 19 live
FPs are verifier-train.
