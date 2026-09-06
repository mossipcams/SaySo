# SaySo training log

What actually ran, what changed, what it scored, and which checkpoint was selected.
Design rules live in [docs/TRAINING_PLAN.md](../docs/TRAINING_PLAN.md); current
commands live in [training/README.md](README.md). Neither belongs here.

## Conventions

- A **run ID** is one training execution. A fresh execution gets a new ID even
  with identical data and settings. Epochs belong to a run; they are not runs.
- A launch that dies before the first optimizer step (failed smoke gate, OOM,
  bad row-count guard) gets a note on the run it was attempting, not an ID.
- Re-scoring a checkpoint produces another **result row**, not another run. Add
  the row; never edit an old one to match a newer scorer.
- Every score records its **eval set revision** and **scorer**, because SaySo has
  two scorers that disagree on the same checkpoint. See "Scorers" below.
- Hashes are the first 16 hex of sha256. Datasets are gitignored, so the hash is
  the only durable identifier.

## Scorers

Neither scorer is in this repository — both live on the training host and are
not version-controlled. That is a known gap: a scorer change cannot currently be
attributed to a commit, so results are pinned by file hash instead.

| Scorer | Path (host) | sha256 | Reads | Notes |
|---|---|---|---|---|
| `structured` | `/srv/training-runs/sayso-eval-quality-llamacpp.py` | `068f412a9be0872b` | `/v1/chat/completions` `tool_calls` | Truncates apostrophe names (`O'Malley's`, `Kids'`). Serving bug, not a label bug. |
| `rawparse` | `/srv/training-runs/eval_gold_raw.py` | `b649076ed217f950` | raw `/completion` text | Apostrophe-safe Python-call parser. Authoritative per TRAINING_PLAN §4. |

`training/scripts/evaluate.py` is **not** a scorer. It wires the offline harness
with `_stub_infer`, which returns a fixed string. No result in this log came
from it.

The same champion checkpoint scores **34/38 structured** and **37/38 rawparse**.
Never compare a structured score to a rawparse score and call it progress.

## Datasets

| File (host) | Rows | sha256 |
|---|---|---|
| `sayso_v2/sayso_train_first_10000_render.jsonl` | 10,000 | `2b1ce63c4cf9db0d` |
| `sayso_v2/sayso_train_10k_plus_supplement_render.jsonl` | 11,701 | `870eaaf44dd4ead9` |
| `sayso_v2/sayso_train_10k_plus_corrective_render.jsonl` | 12,276 | `211de92039b2401e` |
| `sayso_v3/sayso_train_v3_40k_render.jsonl` | 40,000 | `17754a93ae9c397e` |
| `sayso_v2/sayso_quality_eval_recipe_lock.jsonl` (gold) | 38 | `3467874f936887a9` |
| `sayso_v2/sayso_shadow_eval.jsonl` (shadow) | 125 | `b5db74aa3028d9a6` |
| `sayso_v3/sayso_quality_eval_v3_gold.jsonl` | 30 | `6368f5559178abd8` |
| `sayso_v3/sayso_quality_eval_v3_shadow.jsonl` | 100 | `f014c7e874ccbcce` |
| `sayso_test_balanced.jsonl` (held-out) | 2,500 | `af4b44d34b601fa7` |

---

## Run 001: 2,500-row LoRA on the Instruct checkpoint
- **Base:** `/srv/models/LFM2.5-230M` (Instruct, pre-Base era)
- **Data:** `sayso_train_2500_render.jsonl`
- **Config:** `/srv/training-runs/sayso-lfm-2500.yml` (3 epochs, rank 16, lr 2e-4, max_length 2048)
- **Training code:** not recorded
- **Change:** first end-to-end SFT attempt

| Checkpoint | Eval suite / revision | Scorer | Result |
|---|---|---|---|
| final | held-out balanced 2,500 / `af4b44d3` | `eval_heldout_lora2500.py` | exact 0.686, name_ok 0.803, schema_ok 0.9996 |

**Selected:** none — superseded by the Base line.
**Artifact:** output directory `/srv/training-runs/SaySo-LFM2.5-230M-2500` no longer present.

## Run 002: synthetic v2 — no evidence it ran
Config `/srv/training-runs/sayso-lfm-synthetic-v2.yml` exists (Instruct base,
`sayso_train_render.jsonl`, 3 epochs, rank 16) but no output directory and no
eval artifacts. Recorded so the config is not mistaken for a completed run.

## Run 003: first Base run, 10k legacy
- **Base:** `/srv/models/LFM2.5-230M-Base`
- **Data:** `sayso_train_first_10000_render.jsonl` (10,000, `2b1ce63c4cf9db0d`)
- **Config:** `sayso-lfm-base-first.yml` (1 epoch, rank 32, lr 2e-4, max_length 2048)
- **Training code:** not recorded
- **Change:** moved from Instruct to Base; rank 16 → 32
- **Date:** 2026-09-04

| Checkpoint | Eval suite / revision | Scorer | Result |
|---|---|---|---|
| `LFM2.5-230M-Base` (reference) | recipe-lock gold / `3467874f` | `structured` | 2/38 |
| checkpoint-625 (ep1) | recipe-lock gold / `3467874f` | `structured` | 29/38 |
| ep1 merged | held-out balanced 2,500 / `af4b44d3` | `eval_heldout_base.py` | exact 0.048, name_ok 0.696, schema_ok 0.977 |

**Selected:** ep1, carried into Run 004.
**Artifact:** `/srv/training-runs/SaySo-LFM2.5-230M-Base-First/checkpoint-625`
**Note:** a 3-epoch variant was abandoned — `SaySo-LFM2.5-230M-Base-First-3ep-aborted`.

## Run 004: second epoch continued from Run 003
- **Base:** `/srv/models/SaySo-LFM2.5-230M-Base-First-merged` — continued from a merged checkpoint, not from Base
- **Data:** unchanged from Run 003
- **Config:** `sayso-lfm-base-first-ep2.yml` (1 epoch, lr 1e-4)
- **Change:** halved the learning rate for a second pass
- **Date:** 2026-09-05

| Checkpoint | Eval suite / revision | Scorer | Result |
|---|---|---|---|
| checkpoint-625 | recipe-lock gold / `3467874f` | `structured` | 31/38 |
| merged | held-out balanced 2,500 / `af4b44d3` | `eval_heldout_base.py` | exact 0.057, name_ok 0.732, schema_ok 0.981 |

**Selected:** ep2.
**Artifact:** `/srv/training-runs/SaySo-LFM2.5-230M-Base-First-ep2/checkpoint-625`
**Note:** continuing from a merged checkpoint was later ruled out; TRAINING_PLAN now requires training from Base.

## Run 005: 10k-plus supplement
- **Base:** `/srv/models/LFM2.5-230M-Base`
- **Data:** `sayso_train_10k_plus_supplement_render.jsonl` (11,701, `870eaaf44dd4ead9`)
- **Config:** `sayso-lfm-base-10k-plus.yml` (3 epochs, rank 32, lr 2e-4, max_length 2048)
- **Change:** added the supplement rows; trained from Base rather than continuing a merge
- **Date:** 2026-09-05

| Checkpoint | Eval suite / revision | Scorer | Result |
|---|---|---|---|
| checkpoint-1464 (ep2) | recipe-lock gold / `3467874f` | `structured` | 34/38 |
| checkpoint-2196 (ep3) | recipe-lock gold / `3467874f` | `structured` | 33/38 |
| checkpoint-1464 (ep2) | recipe-lock gold / `3467874f` | `rawparse` | 37/38 |
| checkpoint-1464 (ep2) | shadow / `b5db74aa` | `rawparse` | 117/125 |

**Selected:** ep2 (`checkpoint-1464`). Epoch 3 regressed under the same scorer.
**Artifact:** `/srv/models/SaySo-LFM2.5-230M-Base-10k-plus-ep2-Q8_0.gguf`
**Note:** ep2 was re-scored after a parser fix (`eval_quality_recipe_lock_champion_ep2_parserfix.json`); the structured result was unchanged at 34/38.

## Run 006: 10k-plus + corrective — current champion
- **Base:** `/srv/models/LFM2.5-230M-Base`
- **Data:** `sayso_train_10k_plus_corrective_render.jsonl` (12,276, `211de92039b2401e`) — Run 005's data plus 575 corrective rows
- **Config:** `sayso-lfm-base-10k-plus-corrective.yml` (2 epochs, rank 32, lr 2e-4, max_length 2048)
- **Change:** corrective rows weighted at the four known failure classes (light brightness vs fan speed, lock vs unlock, apostrophe names, multi-action retention)
- **Date:** 2026-09-05

| Checkpoint | Eval suite / revision | Scorer | Result |
|---|---|---|---|
| checkpoint-768 (ep1) | recipe-lock gold / `3467874f` | `rawparse` | 38/38 |
| checkpoint-768 (ep1) | shadow / `b5db74aa` | `rawparse` | 119/125 |
| checkpoint-1536 (ep2) | recipe-lock gold / `3467874f` | `rawparse` | 38/38 |
| checkpoint-1536 (ep2) | shadow / `b5db74aa` | `rawparse` | 120/125 |

**Selected:** ep2 (`checkpoint-1536`) — gold tied, shadow moved the right way.
**Promoted:** 2026-09-05T18:46:14Z
**Artifact:** `/srv/models/SaySo-LFM2.5-230M-champion-corrective-ep2-Q8_0.gguf`
**Never scored with `structured`,** so it has no directly comparable number against Runs 003–005.

> **Conflicting champion markers.** `/srv/training-runs/CHAMPION.txt` names this
> run's ep2 (gold 38/38, shadow 120/125). A second marker,
> `/srv/models/SaySo-LFM2.5-230M-Base-10k-plus-CHAMPION.txt`, still names Run 005
> ep2 at 34/38 and was never retired. The second is stale. Its 34/38 also comes
> from a different scorer than the 38/38 above, so the two numbers were never
> comparable to begin with.

## Run 007: v3 40k — abandoned at step 57
- **Base:** `/srv/models/LFM2.5-230M-Base`
- **Data:** v3 40k render, 40,000 rows, sha `2da0a12b269eec67` — scripts modeled as `HassTurnOn`
- **Config:** `sayso-lfm-v3-40k.yml` (2 epochs, rank 32, lr 2e-4, max_length 8192)
- **Training code:** `88f83b5`
- **Date:** 2026-09-06
- **Outcome:** stopped deliberately at step 57/5000 after the script-modeling defect was found. No checkpoint reached, nothing scored.

Earlier launches of this configuration died at the smoke gate and never stepped,
so they carry no run ID: one on a 52,276-row blended mix (stopped on instruction),
and two CUDA OOMs at `max_length` 8192 and 6144 before homes were capped at 64
entities.

## Run 008: v3 40k, corrected script modeling — in progress
- **Base:** `/srv/models/LFM2.5-230M-Base`
- **Data:** `sayso_v3/sayso_train_v3_40k_render.jsonl` (40,000, `17754a93ae9c397e`)
- **Config:** `training/configs/lfm25-230m-synthetic-v3-40k-trl.yml`, host copy `sayso-lfm-v3-40k.yml` (2 epochs, rank 32, lr 2e-4, max_length 8192)
- **Training code:** `1eb94a9`
- **Change vs Run 007:** scripts render as per-script zero-argument tools, are excluded from the entity overview, and their state is marked unavailable
- **Started:** 2026-09-06
- **Expected:** 5,000 steps, ~25 s/step, checkpoints at 2500 (ep1) and 5000 (ep2)

| Checkpoint | Eval suite / revision | Scorer | Result |
|---|---|---|---|
| _pending_ | | | |

**Selected:** not yet.
**Artifact:** `/srv/training-runs/SaySo-LFM2.5-230M-v3-40k`

> This run is the first trained on the Home Assistant 2026.8.3 prompt format
> (YAML overview, no per-entity state, 8 offered tools). Runs 001–006 trained on
> a JSON context that included entity state, which the integration never sends.
> Its gold score is therefore **not** comparable to Run 006's 38/38, in either
> scorer. The v3 gold and v3 shadow suites have never been scored against any
> checkpoint.
