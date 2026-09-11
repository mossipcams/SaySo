# SaySo training pipeline

Current commands, the active config, and what each eval set is for.

- Stable design and constraints: [docs/TRAINING_PLAN.md](../docs/TRAINING_PLAN.md)
- What actually ran and what it scored: [TRAINING_LOG.md](TRAINING_LOG.md)

This directory holds operational scripts for `LFM2.5-230M-Base` TRL rsLoRA.
Axolotl and FunctionGemma YAML under `training/configs/` are leftovers.

## Quick start

```bash
python -m venv training/.venv
source training/.venv/bin/activate
pip install -r training/requirements.txt
pip install pyyaml   # config validation tests
python -m pytest training/tests training/evals training/scripts training/generators -q
```

## Active config

| | |
|---|---|
| Checked-in recipe | `training/configs/lfm25-230m-synthetic-v3-40k-trl.yml` |
| Host copy | `/srv/training-runs/sayso-lfm-v3-40k.yml` |
| Launcher | `/srv/training-runs/run-v3-40k-train.sh` |
| Settings | from Base, rsLoRA rank 32 / alpha 32, `all-linear`, FP16 (no BF16, no flash-attn), microbatch 1, accum 16, lr 2e-4 cosine, assistant-only loss, 2 epochs, `save_strategy: epoch` |

The launcher guards on an exact train-row count, runs a `--max_steps 1` smoke
step, then `nohup`s the real run. Update the row count when the dataset changes.

## Pipeline commands

| Step | Command |
|------|---------|
| Build synthetic v3 train (40k, deterministic) | `python training/scripts/build_synthetic_dataset.py --pipeline v3 --count 40000 --out-dir training/datasets/synthetic_v3_train.jsonl --render-out training/datasets/sayso_train_v3_40k_render.jsonl` |
| Generate v3 quality eval (gold + shadow) | `python training/scripts/generate_v3_quality_eval.py` |
| Generate recipe-lock eval + deterministic 10k train | `python training/scripts/generate_recipe_lock_eval.py` |
| Generate corrective SFT + shadow eval | `python training/scripts/generate_training_supplement.py` |
| Generate balanced held-out test set | `python training/scripts/generate_balanced_test_data.py` |
| Build synthetic train (legacy 10k) | `python training/scripts/build_synthetic_dataset.py --generator-model ... --judge-model ...` |
| Split 80/10/10 | `python training/scripts/split_dataset.py INPUT.jsonl --out-dir training/datasets` |
| Fetch the real Home Assistant home | `HA_URL=... HA_TOKEN=... python training/scripts/fetch_ha_home.py --out training/fixtures/real_home.json` |
| Detect GPU | `python training/scripts/detect_gpu.py` |
| Export GGUF | `python training/scripts/export_gguf.py --checkpoint PATH --dry-run` |
| Verify llama.cpp | `python training/scripts/verify_llamacpp.py --dry-run` |

The v3 build writes the canonical JSONL and the TRL render in one pass and
records `render_rows` in the manifest; it must equal `accepted`. Never hand-filter
the render — dropping rows there shrinks the train set silently.

## Mixing in a real home

`training/scripts/fetch_ha_home.py` pulls a live Home Assistant instance into the
same dict `generators.homes.generate_home` returns: two REST calls, `/api/states`
for entities and `/api/template` for the area/floor map, plus two websocket
commands (`scripts/ha_websocket.py`) for the two facts REST does not carry —
which entities Assist actually exposes, and their aliases. It strips email
addresses from names and entity ids (voice bridges name entities after the linked
account); possessives stay, since they are the apostrophe failure class the
recipe-lock gate tests.

Home Assistant's exposure list is authoritative. An entity it does not expose to
the conversation agent never enters the corpus, whatever its domain, and the
export records which rule it was built under:

| `exposure_source` | Meaning |
|---|---|
| `assist_exposure` | Filtered by Home Assistant's own Assist exposure list. The only valid home-recipe input. |
| `domain_filter` | Websocket unreachable, `--allow-unexposed` used. Marked, and rejected by the recipe. |
| `synthetic_fixture` | `synthetic_reference_home.json`, for tests only. |

`generators.real_home.require_exposure_source` enforces that, so a stale snapshot
fails at build time rather than quietly training on entities the assistant cannot
see. `--allow-stale-home` overrides it deliberately.

Supported actions come from `supported_features`, not from the domain: an Echo
Dot reports no `TURN_ON`, and `media_player.living_room_tv` reports no
`VOLUME_SET`. Generation targets only entities that can actually perform the
operation (`capability_registry.entity_supports`), so the corpus never labels a
call Home Assistant would refuse.

```bash
# refresh the snapshot (required before the home recipe)
HA_URL=http://homeassistant.local:8123 HA_TOKEN=... \
  python training/scripts/fetch_ha_home.py --out training/fixtures/real_home.json \
  --require-entity media_player.living_room_tv

# the home-specific recipe: real-home mixing on, at an explicit nonzero rate
python training/scripts/build_synthetic_dataset.py --pipeline v3 --count 40000 \
  --real-home training/fixtures/real_home.json --home-recipe

# the explicit opt-out
python training/scripts/build_synthetic_dataset.py --pipeline v3 --count 40000 \
  --synthetic-only
```

`--home-recipe` defaults `--real-home-rate` to 0.10; `--real-home-rate` still
overrides it. `--synthetic-only` is the only way to turn mixing off on purpose —
forgetting the flag is not the same decision, so the two are mutually exclusive.

`generators.real_home` holds every fifth entity of each capability out of
training (`split="holdout"`, a capability with one entity stays in train). Those
names never enter the corpus, so scoring on them separates "learned this home"
from "learned homes" — the distinction the synthetic suites cannot make.

One home is a few dozen names, so real rows repeat a small vocabulary. Measured at 10% of
a 10k run: 487 real target labels over 162 distinct names, the most frequent at
1.8x its fair share. `--real-home-entity-cap` bounds how often one entity may be
the target; 0 derives it as four times the fair share
(`real_home.derive_entity_cap`), which is a safety net rather than an active
constraint at these settings. A capped row is rejected and retried, and the retry
re-rolls the real/synthetic draw, so capping redistributes rows instead of
shrinking the corpus. The manifest records the cap, the row count, and the five
most-targeted entities under `real_home`.

The cap matters most for a capability the real home has only one of -- climate,
fan, scene -- where every row of that capability lands on the same name. Set it
explicitly to bind:

```bash
--real-home-entity-cap 100
```

The manifest separates requested from achieved mixing (`real_home.requested_rate`
vs `real_home.achieved_rate`) and counts rows, not targets: a row naming three
entities is one real-home row. `real_home.target_counts` holds the per-entity
label counts, and only real entities — an entity injected for a capability the
home lacks is synthetic and does not inflate them.

## Coverage gates

Quotas are accounted on what a row *teaches*, not on its metadata.
`generators/coverage.py` reads the rendered row and reports the outcome
(`action`, `status`, `clarify`, `absence`, `unsupported`), the tool, the domain
and the targeting mode. A row counts as positive supervision for its operation
only when it calls that operation's tool, on that capability's domain, against a
real target. A refusal tagged `media_players/turn_on` fills that bucket's
*negative* allowance and never its positive floor, and merely offering a tool in
the schema is not coverage.

Every bucket therefore carries three numbers: total rows, a positive floor
(`1 - negative_rate` of the bucket), and a negative allowance. Operations Home
Assistant supplies no tool for get an explicit refusal quota instead of a
positive one. Impossible configurations — a count below 1, a negative rate of 1,
tier proportions that do not sum to 1, or a bucket needing positive rows for an
operation with no tool mapping — raise before the generation loop starts rather
than after exhausting `max_attempts`.

`generators/audit.py` then re-derives all of it from the accepted rows and fails
generation when a required operation or tool has no positive row, or when absence
answers exceed `--max-absence-rate` (default 10%). The manifest records
`positive_by_tool`, `positive_by_operation`, `positive_by_domain_targeting`,
`by_outcome`, `negatives_by_reason` and `absence_rate`, so an absence count is a
distribution to look at rather than a verdict on its own.

`GetDateTime` is the one pinned-contract tool the corpus does not teach
(`capability_registry.TRAINING_COVERAGE_EXCLUDED`): it answers from no entity and
no home state, so an entity-graph generator has no scenario for it. It is
excluded from distractor sampling too — offering a tool no row ever calls teaches
"never call this", which is worse than never having seen it. Home Assistant still
supplies it at runtime; this set bounds only what the dataset claims.

## Grounding families

`generators/grounding.py` holds paired scenarios where the request is fixed and
the entity graph decides the answer — one eligible media player in the area, the
same device renamed, the device moved out of the area, and the same device with
lights, switches and another player added as distractors. Nothing is hard-coded:
each variant is handed to `build_scenario`/`gold_from_scenario`, which derive the
label from the graph and the pinned contract before any wording is applied.
Families also vary aliases, domains and supported actions, and cover individual
and area targeting, genuine ambiguity, and presence/absence pairs.

`--grounding-rate` (default 0.03) sets the share; a run large enough to fit every
required family fails closed if one is missing. `training/evals/grounding_eval.py`
owns the held-out side — the exact Living Room + `media_player.living_room_tv` →
`HassTurnOn(name="TV", domain=["media_player"])` regression plus variations with
different names, ids, areas and distractors. Its prompts join
`excluded_train_prompts()`, so neither the eval rows nor near-duplicate variants
can be trained on.

## Eval sets and what each is for

| Set | Rows | Role |
|---|---|---|
| `sayso_quality_eval_recipe_lock.jsonl` | 38 | Locked release gate. Recipes 1–8, thermostat omitted. Never trained on. |
| `sayso_shadow_eval.jsonl` | 125 | Overfitting check for the recipe-lock gate: same concepts, different homes and phrasing. |
| `sayso_quality_eval_v3_gold.jsonl` | 30 | Expanded gate for v3 domains — climate setpoint, media, timers, vacuum, scene, script, plus ambiguity and no-call. Not yet scored against any checkpoint. |
| `sayso_quality_eval_v3_shadow.jsonl` | 100 | Overfitting check for the v3 gold set. |
| `sayso_test_balanced.jsonl` | 2,500 | Broad held-out regression sample (exact / name_ok / args_ok / schema_ok rates). Not the promotion gate. |
| `training/evals/adversarial.jsonl` | 3 | Held out from training and checkpoint selection. |
| `evals/cases/` | 1 file | Integration-level offline eval for the Home Assistant runtime path, not model selection. |

Promotion needs gold and shadow to move the right way together. Gold alone
improving means the benchmark is being overfit.

## Scoring

**`training/scripts/evaluate.py` is a harness wiring stub.** It calls
`_stub_infer`, which returns a fixed string; it does not score a model. No result
in the training log came from it.

Real scoring runs on the training host against a llama.cpp server:

```bash
# structured tool_calls, drives the merge + serve + score sequence
/srv/training-runs/run_quality_eval_three.sh

# one eval against an already-running server
/srv/training-runs/.venv/bin/python /srv/training-runs/sayso-eval-quality-llamacpp.py \
  --eval-set /srv/datasets/sayso_v2/sayso_quality_eval_recipe_lock.jsonl \
  --out /srv/training-runs/eval_quality_recipe_lock_<checkpoint>.json

# apostrophe-safe raw /completion parse (authoritative)
EVAL_JSONL=/srv/datasets/sayso_v2/sayso_quality_eval_recipe_lock.jsonl \
EVAL_TOK=/srv/models/<merged-checkpoint> \
EVAL_OUT=/srv/training-runs/eval_gold_<checkpoint>_rawparse.json \
  /srv/training-runs/.venv/bin/python /srv/training-runs/eval_gold_raw.py
```

The two scorers disagree on the same checkpoint — the current champion scores
34/38 structured and 37/38 rawparse — because llama.cpp truncates apostrophe
names in structured `tool_calls`. Record which scorer produced a result and never
compare across them. Neither scorer is version-controlled; see the log's
"Scorers" table for file hashes.

### Baselining a checkpoint on the v3 suites

`training/scripts/eval_v3_rawparse.py` is a third scorer (`repoparse`) that does
what `rawparse` does — raw `/completion` text through the apostrophe-safe Python
parser — but lives in this repository, so a change to it is attributable to a
commit. Run it against a llama.cpp server hosting the checkpoint:

```bash
# 1. calibrate: reproduce a recorded rawparse number before trusting a new one
python3 training/scripts/eval_v3_rawparse.py \
  --eval-set training/datasets/sayso_quality_eval_v3_gold.jsonl \
  --tokenizer /srv/models/<run-008-ep1-merged> \
  --out /srv/training-runs/eval_v3_gold_ep1_repoparse.json      # expect 16/30

# 2. the missing baseline
python3 training/scripts/eval_v3_rawparse.py \
  --eval-set training/datasets/sayso_quality_eval_v3_gold.jsonl \
  --tokenizer /srv/models/LFM2.5-230M-Base \
  --out /srv/training-runs/eval_v3_gold_base.json
```

If step 1 does not return 16/30, `repoparse` is not equivalent to `rawparse` and
its numbers belong in their own column — do not compare them to the recorded
rawparse results.

## Dataset views

- **canonical**: OpenAI-compatible envelope with JSON-string `function.arguments`
- **TRL render**: dict `function.arguments` for `apply_chat_template` only

## Layout

```
training/
  adapters/          Schema validation and LFM helpers
  artifacts/         Checkpoints, eval outputs (gitignored)
  configs/           Trainer YAML (TRL recipe is the live path)
  datasets/          Generated JSONL (gitignored)
  evals/             Metrics, harness, recipe lock, v3 quality, adversarial set
  fixtures/          Test fixtures
  generators/        v3 synthetic generation
  scripts/           Pipeline operations
  tests/             Unit tests (no model downloads)
```

## GPU notes (GTX 1070 / Pascal)

Run `python training/scripts/detect_gpu.py` before training. Disable BF16 and
flash-attn. FP16 buys memory, not speed: Pascal has no tensor cores. The card
OOMs above roughly 5k tokens per row at this vocabulary size, which is why homes
are capped at 64 entities and rows offer 8 tools rather than the full catalog.
TRL drops over-length rows silently rather than truncating them, so a `max_length`
below the longest row shrinks the train set without reporting it.
