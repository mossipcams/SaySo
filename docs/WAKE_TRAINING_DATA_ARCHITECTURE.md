# Wake training-data architecture

Phrase is **SaySo**. Wavs are **off git**. The canonical trainer is LiveKit
generate-first (`sayso.yaml`). Long-form recording sessions and miner windows
are eval, labeling, and optional later overlay — not the primary positive class.
2 s scored windows are derived artifacts for mining and evaluation.

Collection: `docs/SATELLITE_DATA_COLLECTION.md`.
Labeling rules: `docs/WAKE_WORD_DATA.md`.

## Canonical trainer (LiveKit generate-first)

```text
satellite/models/sayso.yaml  (LiveKit prod scale; Piper default)
  -> scripts/wake_livekit_run.py setup
  -> scripts/wake_livekit_run.py generate   # never skip
  -> scripts/wake_livekit_run.py augment
  -> scripts/wake_livekit_run.py train
  -> scripts/wake_livekit_run.py export
  -> scripts/wake_livekit_run.py eval
```

Or `wake_livekit_run.py run --work-dir <isolated host dir>`. `data_dir` and
`output_dir` rewrite under `--work-dir`; do not write into `output-living2/`.
Real Snowball / miner audio stays eval and optional later overlay — not the
primary positive class. Shipped Pi operating point remains living2 + verifier
until a later qualified export.

## Corpus / snapshot pipeline (secondary — implemented)

```text
long-form WAV session (Pi: ingest as staging, then ship to train VM)
  -> scripts/wake_corpus.py ingest
  -> scripts/wake_corpus.py ship SESSION  # rsync-over-SSH; delete Pi copy on verify
  -> production living2 + verifier replay (16 kHz, 2 s window, 160 ms hop)
  -> candidate events under corpus/events/
  -> human labels via scripts/wake_mine_report.py or wake_corpus.py label
  -> deterministic session-level splits (corpus_splits.json)
  -> scripts/wake_corpus.py snapshot (train + eval examples; holdout excluded)
  -> scripts/wake_train.py train (living2 seed preserved unless --replace-living2)
  -> scripts/wake_corpus.py holdout-eval (continuous FA/hour on holdout sessions)
```

Rules:

- Long-form sessions are canonical; 2 s scored windows are derived artifacts.
- Human labels are authoritative. No automatic positives.
- Split at recording-session level before deriving train/eval windows.
- `corpus_splits.json` is sticky: later `--holdout` / `--corpus-holdout` IDs
  reassign those sessions to holdout instead of being ignored.
- Eval-session labeled events appear in snapshots with `split=eval` and are
  excluded from train. Holdout sessions stay completely outside snapshots.
- Re-replay uses a per-session `.replay_spool/<session_id>/` scratch dir and
  prunes unlabeled prior events for that session before import.
- Holdout FA/hour uses the same hop/lag path as replay (`window_end -
  pending_lag`). Human-labeled positive windows are not counted as false
  activations.
- Trusted living2 seed and eval holdouts remain unless `--replace-living2`.

CLI surface:

| Command | Role |
| --- | --- |
| `wake_corpus.py ingest` | Register a long-form WAV as a named session |
| `wake_corpus.py ship` | rsync one session Pi → train VM; delete Pi copy after remote sha256 verify |
| `wake_corpus.py replay` | Replay one session and import candidate events |
| `wake_corpus.py split` | Write or update session-level splits |
| `wake_corpus.py snapshot` | Deterministic train/eval snapshot manifest |
| `wake_corpus.py holdout-eval` | Continuous holdout replay + FA/hour report |
| `wake_corpus.py label` | Set a human label on one corpus event |
| `wake_mine_report.py --replay-session` | Same replay/import helper on a corpus root |
| `wake_train.py --corpus` | Train from corpus snapshot + living2 seed |

| Host | Tree | Job |
| --- | --- | --- |
| Pi `192.168.1.54` | `/var/lib/sayso-satellite/` | Capture, live ONNX, miner spool |
| Pi `192.168.1.54` | `/var/lib/sayso-satellite/wake-sessions/` | **Staging** for long-form ingest before `ship` |
| Train `192.168.1.140` | `/home/ubuntu/sayso-wake-data/corpus/` | **Canonical** long-form session corpus (replay, splits, snapshots) |
| Train `192.168.1.140` | `/home/ubuntu/sayso-wake-data/data/` | Named wav **sets** (Snowball) |
| Train `192.168.1.140` | `/home/ubuntu/sayso-wakeword/` | LiveKit `data/` (backgrounds, RIRs, ACAV features), `output-living2/`, isolated `runs/` |

`wake_corpus.py ship` defaults to `ubuntu@192.168.1.140` and remote corpus
`/home/ubuntu/sayso-wake-data/corpus` (rsync over SSH). Shipping does
not require stopping LFM2 on the train host.

The git worktree does **not** hold training wavs. `satellite/models/sayso.yaml`
is the primary training recipe (generate-first, 25k/5k samples, 3 augment
rounds, `target_fp_per_hour: 0.1`). `satellite/models/living2.yaml` documents
the **historical shipped** skip-generate recipe (50 clips, 8 augment rounds) —
current Pi operating point, not the next train. Features are 16 kHz mono,
**2.0 s** windows (`clip_duration: 2.0`).

## What the classifier actually sees

LiveKit `predict` is one **32000-sample** (2.0 s) window, hop **2560**.
living2 train originals on disk are **32768** samples (2.048 s). Replay uses
the first 2.0 s only, so the last 48 ms is unused. That is the real positive
class: a **this-mic 2 s frame**, not “a SaySo burst in isolation.”

Shipped model: living2 ONNX + mel verifier. Fire iff LiveKit ≥ **0.28** (Pi)
or **0.50** (host recipe comment) **and** verifier ≥ **0.445**. Mine on
LiveKit **before** the verifier veto.

## Named sets on the host (`sayso-wake-data/data/`)

Counts are wavs in the directory root (2026-09-21). Many folders are leftover
experiments and must not be treated as “more data = better.”

### In living2 (classifier)

| Set | n | Role |
| --- | ---: | --- |
| `positive_recorded/` | 50 | Train positives |
| `negative_living_iso/` | 23 | Train neg |
| `negative_room/` | 43 | Train neg |
| `negative_recorded/` | 24 | Train neg |
| `negative_living_talk/` | 89 | Trainer **val only** |

Val positives are **10 copies** of the 50 (not independent). Backgrounds/RIRs
live under `sayso-wakeword/data/`, not this tree. Class prior: positive **96**,
ACAV **64**, `max_negative_weight` **200**.

### Verifier only (not the classifier)

| Set | n |
| --- | ---: |
| `verifier_live_fp/` | 19 |

Fit: 50 recorded SaySo vs these 19. Do not dump them into living2.

### Never train the classifier on

| Set | n | Why |
| --- | ---: | --- |
| `holdout_living/` | 23 | Living-room gate (8 SaySo + talk/cmd/neg) |
| `holdout_eval/` | 21 | Second holdout |
| `negative_miner_party/` | 74 | Live-complaint-style FPs |
| Pi mining spool | — | Unlabelled until `--label` |

### Available, not in living2

| Set | n | What it actually is |
| --- | ---: | --- |
| `200-positive/` | 200 | 16 kHz **2.0 s** center-padded prompted takes + `SOURCE.txt` — **keep** |
| `200-positive/raw/` | 200 | Native **48 kHz ~1.13 s** ALSA chops from the same session — **keep** |
| `positive_recorded_holdback/` | 74 | Extra recorded positives; not the living2 50 |

On the stored 2 s `200-positive/*.wav` files, living2 AND-gate already fires
**147/200** at 0.28 (median LiveKit ~0.74). They are real SaySos in that
packaging. They are **not** a drop-in replacement for the 50: raw captures
are quieter (~−48 dBFS, peak ~900 vs living2 ~−37 dBFS, peak ~3700) and
only ~0.56 s of active speech vs ~1.2 s on the 50/holdout.

### Leftover / do not casually mix

TTS and scrape piles from older recipes: `positive/` (13700), `positive_val/`
(6000), `negative/` (21500), `negative_phoneme/` (8000), `negative_talk/`,
`negative_iso_tts/`, `negative_tts_small/`, `negative_living_up/`,
`negative_mined/` (1539 unlabelled-style), `negative_mined_labelled/` (32).
`sayso-training.yaml` still points at phonetic inference. Failed levers
(blend, living3 ACAV, living6 +357 mine negs) used this sprawl.

## Isolated train runs (`sayso-wakeword/runs/`)

Each candidate should copy wavs into `output/sayso/{positive,negative}_{train,test}/`
as `clip_NNNNNN.wav` (unaugmented) plus `_r0`…`_r7` after augment. Do **not**
write into `output-living2/`.

Known runs:

| Run | Intent | Outcome |
| --- | --- | --- |
| `labelled-20260921/` | 50 + labelled miner clips | Rejected (recall/FP trade) |
| `pos200-20260921/` | **Replace** 50 with 200, `target_fpph` 0.02 | `max_neg_w` 200→800; 5/8; dry train ~0.08 |
| `pos200-wake-20260921/` | 50 + 200 **+10 dB, room overlay, right-align** | 7/8 but FPs; overlay verifier ~0.03 |
| `pos200-keep-20260921/` | 50 + 200 as stored 2 s | **Aborted** (no export) |

Production Pi still living2.

## Why this architecture is not ideal

1. **Two host trees** (`sayso-wake-data/data` vs `wakeword/output-living2`) with
   duplicated splits. The yaml `data_dir` is backgrounds/RIRs, not the named
   sets. Easy to train the wrong folder.
2. **Directory name = label** with no sidecar on the 50/200. Miner records
   have labels; recorded positives do not. Easy to mix holdout into train.
3. **Val talk (89) is also an eval gate.** `target_fp_per_hour: 0.02` on that
   val set **doubles** `max_negative_weight`. living2.yaml still has 0.02.
   Pinning 1800 is a trainer hack, not a data model.
4. **New positives have no intake spec.** 200 arrived as 1.13 s 48 kHz chops,
   then silence-padded 2 s, then (in one run) room-overlaid. Three different
   objects, one folder name. The classifier window is always 2 s this-mic.
5. **Eval sets and hard-neg piles sit next to train sets** with similar names
   (`negative_mined` vs `negative_miner_party` vs `negative_living_talk`).
6. **Holdback 74, 200-positive, and the 50** are three positive families
   with no documented join rule (union vs replace vs “misses only”).
7. **Git cannot see the data.** Architecture lives in HANDOFF + this file +
   host disk. Drift is normal.

## Intake that would match inference (not implemented)

One positive example = **this Snowball**, **16 kHz**, **2.05 s native** (or
2.0 s with the word in-window), level near the 50 (**~−38 dBFS**), **not** in
holdout. One hard negative = **human-labelled miner window** near 0.25–0.45,
not the 89/74/19 eval sets. Keep the 50. Do not silence-pad 1 s takes or
overlay room beds the verifier rejects.

## Trainer contract (sayso.yaml — next model)

```text
setup → generate (Piper) → augment × 3 → train 100k steps → export → eval
conv_attention / small; target_fp_per_hour 0.1; batch 50/50/1024/50
ignore optimal_threshold (~0.05) until a new export qualifies
```

## Shipped operating point (living2 — unchanged on Pi)

```text
skip generate (historical)
augment × 8 → features (n × 8, 16, 96)
train 20k + optional FP phases
export sayso.onnx
score at 0.28 (Pi) / 0.50 (host notes) AND verifier 0.445
```

`wake_train.py --schedule` stays disabled (`satellite/eval/audio/` empty).
