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
python -m pytest training/tests training/evals training/scripts -q
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
| Detect GPU | `python training/scripts/detect_gpu.py` |
| Export GGUF | `python training/scripts/export_gguf.py --checkpoint PATH --dry-run` |
| Verify llama.cpp | `python training/scripts/verify_llamacpp.py --dry-run` |

The v3 build writes the canonical JSONL and the TRL render in one pass and
records `render_rows` in the manifest; it must equal `accepted`. Never hand-filter
the render — dropping rows there shrinks the train set silently.

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
