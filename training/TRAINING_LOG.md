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
| `rawparse` | `/srv/training-runs/eval_gold_raw.py` | `b649076ed217f950` | raw `/completion` text | Apostrophe-safe Python-call parser. Authoritative per TRAINING_PLAN §4. Hangs on untuned-base output — see below. |
| `repoparse` | `training/scripts/eval_v3_rawparse.py` | pinned by commit | raw `/completion` text | Same method as `rawparse`, in-repo. Calibrated equal to it (below). |

`repoparse` exists because `rawparse` cannot score the base model: its
`parse_call` loop does not advance the cursor on prose that contains `(`, so an
untuned completion spins forever. Observed 2026-09-06 on v3 gold — 29 of 30 rows
served, then the process pinned a core at 8.3 GB RSS until killed. `repoparse`
reuses `training/evals/lfm_python_parse.py`, which raises instead of looping.

**Calibration.** `repoparse` reproduced both recorded `rawparse` results for Run
008 checkpoint-2500 exactly — 16/30 on v3 gold `530234a0` and 9/38 on the
v3-format recipe lock `47d9ca1c`. The two scorers are therefore comparable on
these suites, unlike `structured` and `rawparse`. Re-calibrate if either changes.

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
| `sayso_v3/sayso_train_v3_40k_render.jsonl` (Run 008) | 40,000 | `17754a93ae9c397e` |
| `sayso_v3/sayso_train_v3_40k_render.jsonl` (Run 009) | 40,000 | `a6babc7345fc097c` |
| `sayso_v2/sayso_quality_eval_recipe_lock.jsonl` (gold, v2 format) | 38 | `3467874f936887a9` |
| `sayso_v2/sayso_shadow_eval.jsonl` (shadow) | 125 | `b5db74aa3028d9a6` |
| `sayso_v2/sayso_quality_eval_recipe_lock.jsonl` (gold, v3 format) | 38 | `47d9ca1cc52d935b` |
| `sayso_v3/sayso_quality_eval_v3_gold.jsonl` | 35 | `79c90d4cd4bc9615` |
| `sayso_v3/sayso_quality_eval_v3_shadow.jsonl` | 100 | `32ab38ea06932344` |
| `sayso_test_balanced.jsonl` (held-out) | 2,500 | `af4b44d34b601fa7` |

Superseded eval revisions, kept so old result rows stay resolvable:
`6368f5559178abd8` and `f014c7e874ccbcce` (v3 gold/shadow, pre-dedup),
`6377977c71354b04` (v3 shadow, stale script rows — see Run 008), and
`530234a03f0bb46f` / `5a98b56297099211` (v3 gold/shadow, before light and fan
coverage existed).

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
| checkpoint-2500 (ep1) | recipe-lock gold, v3 format / `47d9ca1cc52d935b` | `rawparse` | 9/38 |
| checkpoint-2500 (ep1) | v3 gold / `530234a03f0bb46f` | `rawparse` | 16/30 |
| checkpoint-2500 (ep1) | v3 shadow / `6377977c71354b04` | `rawparse` | 25/100 |
| `LFM2.5-230M-Base` (reference) | recipe-lock gold, v3 format / `47d9ca1cc52d935b` | `repoparse` | 4/38 |
| `LFM2.5-230M-Base` (reference) | v3 gold / `530234a03f0bb46f` | `repoparse` | 6/30 |
| `LFM2.5-230M-Base` (reference) | v3 shadow / `5a98b56297099211` | `repoparse` | 15/100 |
| checkpoint-2500 (ep1) | recipe-lock gold, v3 format / `47d9ca1cc52d935b` | `repoparse` | 9/38 |
| checkpoint-2500 (ep1) | v3 gold / `530234a03f0bb46f` | `repoparse` | 16/30 |
| checkpoint-2500 (ep1) | v3 shadow / `5a98b56297099211` | `repoparse` | 25/100 |
| checkpoint-2500 (ep1) | v3 gold, light/fan coverage / `79c90d4cd4bc9615` | `repoparse` | 20/35 |
| checkpoint-2500 (ep1) | v3 shadow, light/fan coverage / `32ab38ea06932344` | `repoparse` | 22/100 |

**Selected:** not yet — epoch 1 only.

The light and fan rows paid for themselves on the first run. Gold scores 4/5 on
them and the one failure is the class the coverage was added for: the model
emitted `HassLightSet(name='Sunroom Accent Light', percentage=25)` where
`brightness=25` was wanted — the same wrong-argument slip as the `temperature=64`
case, now caught by a suite instead of by reading completions. Shadow scores
**0/10**, all ten the blanket refusal, which is the memorization failure below
rather than anything specific to lights and fans.

The base rows are the first baseline any v3 suite has had. Scored 2026-09-06
against `LFM2.5-230M-Base-Q8_0.gguf` on a llama.cpp server, tokenizer
`/srv/models/LFM2.5-230M-Base`. Epoch 1 beats base on all three suites
(+10 gold, +10 shadow, +5 recipe-lock), but the aggregate hides two regressions:

- **Refusal discipline inverted.** On the six v3 gold rows that want no tool
  call, base scores 3/6 and ep1 scores **0/6** — ep1 emits a call every time,
  inventing tools and entities (`HassMediaStart(...)`, `add_area(...)`,
  `name='The Courment'`). Two of base's three are genuine prose refusals; the
  third passes only because its completion did not parse, which both scorers
  treat as "made no call".
- **Scripts regressed below base.** On the six corrected shadow `script_run`
  rows base scores 4/6 by copying the offered tool name (`[cellar_away_script()]`),
  and ep1 scores **0/6**, answering every one with the same refusal: "I can't do
  that with the available Home Assistant tools." The per-script tools carry very
  thin supervision — ~190 distinct names over 40k rows, 1 to 37 occurrences each,
  none above 0.1% — and the script is excluded from the overview, so the fix in
  `1eb94a9` appears to have taught refusal rather than execution.

Base's passes are mostly not real capability: 10 of its 15 shadow passes and 4 of
its 4 recipe-lock passes are rows that wanted no call at all.

Re-scoring ep1 on the corrected shadow `5a98b56297099211` returned 25/100 again,
the same total as the stale revision. The six repaired script rows moved from
unwinnable to winnable and ep1 still fails all six, so the totals coincide.

These were taken mid-schedule and are **not** a verdict on the run. The cosine
schedule is sized for 2 epochs, so at step 2500 the learning rate is still
1.37e-4 of 2e-4 and the weights are far from annealed. Every historical score in
this log comes from an end-of-run checkpoint; no previous run recorded a
mid-training eval, so there is nothing to compare these against.

Training metrics at the same point: loss 0.004, `mean_token_accuracy` 0.995,
entropy 0.011. The model fits the training data almost perfectly while scoring
poorly on held-out prompts, which is what the gold and shadow sets exist to
detect.

Failure modes are argument precision, not tool selection — the model picks the
right tool and emits valid call syntax:

- `temperature=64` where `brightness=64` was expected (the known light/fan class)
- `name='Patio Blinds'` for `Patio South Blinds`; `name='Up'` for `Upstairs Robot Vacuum` — dropped words when copying names
- `device_class=['door']` where `['garage']` was expected
- occasional spurious refusal on a supported action
- repetition loops in long names (`'Sunroom Robotics Robotics Robotics…'`)

### Diagnosis: the model memorized entity names

Run 008 ep1 fits its training data and does not generalize, and the variable
responsible is the entity-name vocabulary. Isolated 2026-09-06 by mutating one
thing at a time and re-scoring with `repoparse` against the ep1 server:

| Probe | Change from the row as trained | Score |
|---|---|---|
| 100 training rows, unmodified | none — replayed through the eval harness | **99/100** |
| 100 training rows, entities renamed | novel name tokens, everything else identical | **38/100** |
| v3 shadow, as built | held-out rows | 25/100 |
| v3 shadow, homes padded to 32 entities | home size matched to training | 21/100 |
| v3 shadow, targets renamed to training names | target name only | 19/100 |

The 99/100 rules out the harness, the chat-template render, and the scorer: the
model answers its own training rows correctly through the exact path the eval
uses. Renaming the entities in those same rows — same homes, same home sizes,
same utterance templates, same tools — drops it 61 points. Home size is not the
cause (padding made it worse), and neither is name novelty at the target alone.

The reason is that names are barely varied. Across all 40,000 rows the overview
uses **2,124 distinct entity names built from 52 distinct words**, filling
1,245,259 name slots — a median of **693 appearances per name**. At that
repetition the cheapest thing for a 230M model to learn is the finished string,
not the rule that produces it. So it never learned to copy a name out of the
context and slug it, and on an unseen name it either garbles the slug —
`athematic_quill_script`, `boxroom_quill_script`, `cell_room_quill_script` for
`Solarium Quill Script`-style targets — or refuses outright. On the renamed
probe that is `tool_name_mismatch` 52, `missing_tool_call` 10.

This also explains why base beats ep1 on scripts, 4/6 to 0/6: base has no
memorized lookup, so it does the obvious thing and copies the offered tool name.

Fixing this is a generation change, not a training or eval change: the name
vocabulary has to be large enough that no name is seen often enough to memorize.
Until then a longer run or a second epoch will drive training loss lower without
moving these suites.

**Fixed in `generators/homes.py`** (not yet regenerated or trained). Names are now
drawn from 56 areas, 48 placements, 26 neutral words, 24 owner possessives and 13
nouns across four shapes, with a per-home guard so no two names slug alike. Two
constants that had been injected into every home of size ≥ 16 — `Kitchen Ceiling
Lamp` and `Bedroom TV` — are drawn too; they alone had appeared in roughly 35k
rows. Measured against the real pipeline:

| | Run 008 data | after the change |
|---|---|---|
| distinct entity names | 2,124 | ~291,000 at 40k rows |
| distinct words in names | 52 | 175 |
| median repeats per name | 693 | 1 |
| distinct target names | — | 1,538 across 1,561 calls (2k sample) |
| rows accepted / attempted | 69% | 69% |
| names carrying an apostrophe | rare, via a `% 7` branch | 15.9% |

`tests/test_homes.py::test_entity_names_are_too_varied_to_memorize` pins the
property so it cannot regress quietly.

The brightness failure was not visible to the v3 suites. `HassLightSet` and
`HassFanSetSpeed` are 12.3% of training calls (4,472 of 36,381) and were 0% of
both v3 gold and v3 shadow, so the light/fan class only showed up in the
v3-format recipe-lock score. **Closed 2026-09-06**: gold gained 5 rows (35
total, 3 `HassLightSet` and 2 `HassFanSetSpeed`) and shadow gained 10 (5 and 5),
half of each pool placing the other device in the home so a row fails if the
model reaches for brightness on a fan or speed on a light. The shadow slot
rebalance took one row each from ten categories to hold the count at 100; every
category still has at least four. Shadow totals before and after that change are
not comparable.

One further gap, recorded because it bounds what these numbers mean: every v3
eval home holds one or two entities while training homes hold 7 to 64 (median
32). That is deliberate — minimal homes isolate the behavior — and it is not the
cause of the low scores; padding the homes to 32 entities scored *worse*, 21/100
against 25/100. See the diagnosis below.

Scored twice: once against eval sets whose entity names were duplicated by a
context bug, and again after the fix. Both runs returned 9/38, 16/30, 25/100, so
the duplication was not the cause. Only the corrected revisions are recorded
above.

> **The shadow 25/100 is capped at 94.** Shadow revision `6377977c71354b04`
> carried six `script_run` rows built before `1eb94a9`: they expect
> `HassTurnOn(name=…, domain=['script'])`, the modeling this run exists to
> replace. Four of the six name a script that appears nowhere in the prompt —
> scripts are excluded from the overview and those rows were not offered a
> script tool — so they are unanswerable. The other two offer the correct
> per-script tool while still expecting `HassTurnOn`, marking a correctly
> trained model wrong. Gold was regenerated by `1eb94a9`; `build_shadow_specs`
> was missed. Fixed and shadow regenerated as `5a98b56297099211`. The 25/100 is
> not comparable to any score taken against the new revision. Re-scored against
> the corrected revision above.
**Artifact:** `/srv/training-runs/SaySo-LFM2.5-230M-v3-40k`

## Run 009: v3 40k, realistic and varied entity names — cancelled
- **Base:** `/srv/models/LFM2.5-230M-Base`
- **Data:** `sayso_v3/sayso_train_v3_40k_render.jsonl` (40,000, `a6babc7345fc097c`)
- **Config:** unchanged from Run 008 — `sayso-lfm-v3-40k.yml`, 2 epochs, rank 32,
  lr 2e-4, `max_length` 8192
- **Training code:** `d4c9eee`
- **Change vs Run 008:** data only. Entity names are drawn from a vocabulary wide
  enough that copying a name out of the context is cheaper than memorizing it,
  and modelled on real Home Assistant homes rather than invented words.
- **Started:** 2026-09-07
- **Expected:** 5,000 steps at ~25.4 s/step, checkpoints at 2500 (ep1) and 5000 (ep2)

| | Run 008 | Run 009 |
|---|---|---|
| distinct context names | 2,124 | 145,682 |
| distinct words in names | 52 | 478 |
| median repeats per name | 693 | 3 |
| distinct target names | — | 19,959 |
| targets seen exactly once | — | 76% |
| rows accepted / attempted | 69% | 70% |
| sequence tokens, median / max | 2,401 / 3,692 | 2,440 / 3,787 |

Names follow acon96/Home-Assistant-Requests-V2 in shape and casing — "Pantry
Under Cabinet Light", "Chamberlain Blinds", "EV charger outlet", "Foyer Main Door
Lock" — but not in scale: that pile has 995 names at a median of 401 repeats,
which is Run 008's failure mode. Verified before launch: zero eval entity names
and zero eval prompts appear in the corpus, and no sequence exceeds `max_length`.

Run 008's output directory was deleted to launch this. Its merged model and GGUFs
are kept at `/srv/models/SaySo-LFM2.5-230M-v3-40k-ep1-*`, so every Run 008 result
above is still reproducible. The Run 008 dataset is kept beside the new one as
`sayso_train_v3_40k_render.jsonl.run008-17754a93`.

**The scores to beat are the base reference, not Run 008's.** Run 008 trained on
a different corpus; only base is common to both. Base scores 6/30 on the old v3
gold and 15/100 on the old shadow, and has not been scored on the light/fan
revisions — score it on `79c90d4c` and `32ab38ea` before reading ep1.

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

Run 009 was stopped and its training output deleted at the user’s request on
2026-09-08. Its dataset and the earlier exported Run 008 models were retained.

## Run 010: realistic 40k — stopped for semantic label defects

The replacement corpus (`e088b3a6066f7fba`, generator `52b89a4`) began training
from Base on 2026-09-08 in `/srv/training-runs/sayso-realistic-20260908`.
An independent full-corpus audit subsequently found 52 timer absence claims,
including 15 actionable refusals with the required tool offered; 79 status
examples answered “Done.”; and 302 calls used canonical names that also aliased
another same-domain entity. Another 280 positive ambiguity examples used
canonical wording, and all 2,784 floor calls also specified a single area.

Training was paused when the timer defect was confirmed, then its exact launcher
and worker processes were cancelled before replacement. Its artifacts are
retained for diagnosis; its optimizer and adapter must not seed a replacement.
The corrected run starts again from `LFM2.5-230M-Base`.

## Additional Base baseline: realistic 120-case eval

The new realistic 40k corpus (`e088b3a6066f7fba`, source `52b89a4`) has a
separate 120-case diagnostic eval. The [frozen fixture and report](fixtures/realistic_eval_20260908_v2.md)
include the complete case list and commands to reconstruct the exact evaluated
JSONL. Eval sha256: `35e6be66fd42e73ca`. The five households and every normalized
request were checked against the actual training file: no identical contexts
or normalized prompt overlaps. Existing locked eval files are unchanged.

| Model | Evaluation / scorer | Result |
|---|---|---:|
| LFM2.5-230M-Base Q8_0 | realistic 120 / `repoparse` | 19/120 |
| Same saved generations | Python-literal syntax diagnostic, same labels | 14/120 |

Both readings have **0/100 exact action matches**. The original parser accepts
only single-quoted string arguments and hides five double-quoted tool attempts
in no-call cases. The diagnostic counts those attempts, leaving **14/20
abstentions**. Neither score judges whether clarification prose is useful.
All 120 requests completed without transport errors.

The isolated CPU server used `tokenizer.ggml.add_bos_token=bool:false` because
the HF-rendered prompt already includes BOS. All 120 server token sequences
matched the Base tokenizer exactly; prompts span 1,610–2,325 tokens, with no
truncation. This new set and serving configuration differ from earlier locked
baselines; compare future checkpoints using this same frozen fixture and setup.

The host bundle `/srv/training-runs/sayso-realistic-20260908` retains the raw
outputs, diagnostic, and provenance. The isolated eval server was stopped after
scoring; the fresh Base training run continued. No checkpoint is promoted by
this baseline.

## Run 011: v3 semantic repair, fresh Base

- **Generator:** `6fdd46b`, seed `20260908`, 40,000 rows.
- **Canonical SHA256:** `1a18bbd807bdb7b89b5c47acbe15b4af59fc05308814071ffe714bb4f3838c26`.
- **TRL render SHA256:** `58e78c74c72928ac305f4375faa57315aee9eeda799db91afc6b42719908f04f`.
- **Data:** `/srv/datasets/sayso_semantic_20260908/`.
- **Bundle:** `/srv/training-runs/sayso-semantic-20260908/`.
- **Output:** `/srv/training-runs/SaySo-LFM2.5-230M-v3-semantic-20260908`.
- **Recipe:** fresh `/srv/models/LFM2.5-230M-Base`, two epochs, rank/alpha 32,
  rsLoRA, FP16/SDPA, learning rate 2e-4, batch 1 / accumulation 16, assistant-only
  loss, no packing, maximum 8,192 tokens. No Run 010 or smoke adapter is resumed.

The built-in and independent full-corpus audits found zero actionable timer
refusals, zero timer inventory absence claims, zero status queries labeled
“Done.”, and zero called-name/alias collisions. All 138 timer refusals withhold
the required tool. All 2,857 floor calls are floor-wide. There are 282 positive
generic room/device requests, 2,887 multi-call rows, 428 exclusions, and 659
actual spoken-alias requests. No normalized realistic-eval prompt or exact
eval context overlaps the new corpus; existing locked prompts also pass the
generator exclusion gate. The frozen 120-case eval is unchanged.

The corpus contains 3,470 distinct canonical context names, 185 words in those
names, and 24,522 distinct requests ignoring case. It deliberately repeats
ordinary household names. STT corruption affects 5,596 rows (13.99%); call and
no-call casing are balanced independently. This is deterministic synthetic
coverage, not a measurement of real-user task success.

Local validation: **268 tests passed** across generator, training tests, evals,
and scripts. PR CI also passed all automated checks. The bundle retains both
independent audit scripts, the old and rebuilt reports, source snapshot,
configs, and launch/evaluation scripts.

Base reference uses Q8_0 and `tokenizer.ggml.add_bos_token=bool:false`, matching
the checkpoint evaluator. Each request completed without transport errors.

| Suite | Raw-parser Base result |
|---|---:|
| Recipe-lock (38) | 3/38 |
| V3 gold (35) | 8/35 |
| V3 shadow (100) | 12/100 |
| Frozen realistic (120) | 19/120 |

These references supersede the duplicate-BOS serving setup for this run. The
existing raw parser’s quote-handling limitation remains; do not interpret
no-call accuracy as clarification quality. Both epoch checkpoints are scheduled
for all four suites, with no automatic promotion.

The full HF template/tokenizer audit matched the rendered-file hash: sequence
lengths are 1,485 minimum, 2,454 median, and 3,675 maximum; no truncation and no
empty assistant-loss masks. The frozen realistic eval hash remains
`35e6be66fd42e73cad015836e09f584209e6f395846a6476338acae14758ed43`.

**Started:** 2026-09-08 21:33:47 UTC, launcher PID `102630`. The 12-step GPU
smoke on the longest row passed: finite losses, finite gradients after initial
FP16 loss-scaler backoff, finite saved tensors, and nonzero learned LoRA B
weights. The separate full run then loaded Base and its fresh output directory.
Status and logs live in the bundle; epoch results remain pending.

## Run 012: early checkpoint feedback — restart authorized

Run 011 was cancelled at the user’s request to enable earlier evaluation. Its
logs and artifacts remain in place. The running trainer could not reload the
epoch-only save schedule and had no saved checkpoint to resume.

The replacement uses the same audited 40k corpus (`58e78c74c72928ac`), source
`6fdd46b`, Base, and training hyperparameters. Its only training-config changes
are a fresh output path and saving every 250 steps, retaining all 20 checkpoints
so both epoch checkpoints remain available. A background watcher validates the
first checkpoint’s trainer state and adapter before invoking the existing
CPU merge/Q8 export and all four frozen eval suites. It writes a separate
`early-eval-status` and reports under `results/step-250-*.json`; training
continues during evaluation. The first result is expected about two hours after
restart at the measured throughput. No automatic model promotion or early stop.

- **Bundle:** `/srv/training-runs/sayso-semantic-early-20260908`.
- **Output:** `/srv/training-runs/SaySo-LFM2.5-230M-v3-semantic-early-20260908`.
- **Started:** 2026-09-08 22:10:46 UTC, after the fresh 12-step GPU smoke passed.
- **Launcher PID:** `105178`.
- **Validation:** the early watcher regression failed before implementation,
  then passed by evaluating step 250 while no epoch checkpoint existed. Shell
  syntax checks passed. The config diff preserves the data and training recipe.

### Quick gradient-checkpointing memory benchmark

Both attempts used the longest 3,675-token example, batch 1, accumulation 1,
FP16, and the same rank-32 rsLoRA recipe. CUDA-synchronized timings exclude
the first four steps. The normal full run still accumulates 16 microbatches;
these per-example timings are not its optimizer-step timings.

| Setting | Result | Peak total GPU memory |
|---|---|---:|
| Checkpointing on | 12 steps passed; median 2.581 seconds per step | 4,961 MiB |
| Checkpointing off | CUDA OOM in the first step; requested another 826 MiB | 7,523 MiB sampled before failure |

The on benchmark’s saved tensors are finite and LoRA B weights changed, proving
an optimizer update. The inference server remained running during both trials.
**Keep gradient checkpointing enabled:** disabling it does not fit this GPU
workload, so the earlier estimated six-hour saving is not available with this
configuration. The bundle retains `benchmark.py`, both configs/logs, sampled
GPU usage, the passing adapter, and `memory-benchmark.json`.
## Run 013: 40k grounded gauntlet v4, fresh Base

- **Generator:** `6fdd46b`, seed `20260911`, 40,000 rows, v3 pipeline.
- **Canonical SHA256:** `c181ca96eb1a584142ea9605f56819cc7ef8731f86b71234dacd73b7abf47a15`.
- **TRL render SHA256:** `8fde3c75faf28b0c79c7c85f61ebfa8a5601f948c418e2b68b1f59682a4f9ccc`.
- **Manifest SHA256:** `bdb08d8c1d3065d29e448d538601713df7e34c5af2a6350082b36e6592307a04`.
- **Data:** `/srv/datasets/sayso_40k_grounded_gauntlet_20260911/`.
- **Bundle:** `/srv/training-runs/sayso_40k_grounded_gauntlet_20260911-source/`.
- **Output:** `/srv/training-runs/SaySo-LFM2.5-230M-40k-grounded-gauntlet`.
- **Recipe:** fresh `/srv/models/LFM2.5-230M-Base`, two epochs, rank/alpha 32,
  rsLoRA, FP16/SDPA, learning rate 2e-4 cosine, batch 1 / accumulation 16,
  assistant-only loss, no packing, maximum 8,192 tokens. Saving every 250
  steps, retaining 20 checkpoints. No Run 011 or Run 012 adapter is resumed.

The 40k corpus regenerated cleanly. Both files contain exactly 40,000 rows.
Manifest gates pass: no quota shortfall; `uncovered_operations` empty;
real-home rate requested 10.0% achieved 10.0%; 16 grounding families present
(88 rows total, requested 3% achieved 0.22%); 35,642 tool calls positive,
4,871 negative. The held-out gauntlet eval (15 rows), recipe-lock (35 rows),
v3 gold (35 rows), and v3 shadow (100 rows) show zero overlap with the
v4 corpus.

Token audit confirms 40,000 rows; min 1,552 / median 2,476 / max 3,795
tokens; zero truncated rows; zero empty assistant masks; assistant content
and tool-call names verified inside decoded target on every row. The 12-step
GPU smoke on the longest 3,795-token row passed: finite losses (1.362 to
0.001071), finite gradients after initial FP16 loss-scaler backoff,
finite saved tensors, nonzero learned LoRA B weights. Per-optimizer-step
time ~25 s on the GTX 1070; peak GPU memory 4,687 MiB / 8 GiB.

The full run then loaded Base and its fresh output directory.

- **Started:** 2026-09-11 18:47:45 UTC, after the fresh 12-step GPU smoke passed.
- **Launcher PID:** `162739`.
- **Status:** training in progress; epoch results remain pending. Both epoch
  checkpoints (step 2,500 and step 5,000) will be evaluated against the four
  frozen eval suites; no automatic model promotion or early stop.