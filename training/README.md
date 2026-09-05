# SaySo training pipeline

**Training design:** [docs/TRAINING_PLAN.md](../docs/TRAINING_PLAN.md) is the
only training-plan document. Read it before changing generators, adapters, or
configs.

This directory holds operational scripts for `LFM2.5-230M-Base` TRL rsLoRA.
Axolotl and FunctionGemma YAML under `training/configs/` are leftovers.

## Quick start

```bash
python -m venv training/.venv
source training/.venv/bin/activate
pip install -r training/requirements.txt
pip install pyyaml   # config validation tests
python -m pytest training/tests training/evals -q
```

## Pipeline commands

| Step | Command |
|------|---------|
| Generate recipe-lock quality eval + deterministic 10k train | `python training/scripts/generate_recipe_lock_eval.py` |
| Generate corrective SFT + shadow eval | `python training/scripts/generate_training_supplement.py` |
| Generate balanced held-out test set | `python training/scripts/generate_balanced_test_data.py` |
| Adapt only | `python training/scripts/adapt_dataset.py INPUT.jsonl OUTPUT.jsonl --seed 42` |
| Split 80/10/10 | `python training/scripts/split_dataset.py INPUT.jsonl --out-dir training/datasets` |
| Detect GPU | `python training/scripts/detect_gpu.py` |
| Evaluate | `python training/scripts/evaluate.py training/evals/adversarial.jsonl` |
| Export GGUF | `python training/scripts/export_gguf.py --checkpoint PATH --dry-run` |
| Verify llama.cpp | `python training/scripts/verify_llamacpp.py --dry-run` |

Host TRL runs copy `training/configs/lfm25-230m-synthetic-v2-trl.yml` and point
`data_files` / `output_dir` at the current mix. Train from Base, not from a
champion checkpoint. Defaults: rsLoRA rank 32, FP16, accum 16, `2e-4`, cosine,
assistant-only loss.

## Dataset views

- **canonical**: OpenAI-compatible envelope with JSON-string `function.arguments`
- **TRL render**: dict `function.arguments` for `apply_chat_template` only

See [TRAINING_PLAN.md](../docs/TRAINING_PLAN.md) for the on-disk contract and
the apostrophe-safe eval parser.

## Layout

```
training/
  adapters/          Example conversion and schema validation
  artifacts/         Checkpoints, eval outputs (gitignored)
  configs/           Trainer YAML (TRL recipe is the live path)
  datasets/          Generated JSONL (gitignored)
  evals/             Metrics, harness, recipe lock, adversarial set
  fixtures/          Test fixtures
  generators/        Label-first example generation
  scripts/           Pipeline operations
  tests/             Unit tests (no model downloads)
```

## GPU notes (GTX 1070 / Pascal)

Run `python training/scripts/detect_gpu.py` before training. Disable BF16 and
flash-attn. The live path is FP16 rsLoRA on TRL. If no GPU is present, `train.py`
exits 0 with a documented skip message.

## Adversarial eval

`evals/adversarial.jsonl` is held out from training and checkpoint selection.
The working quality gate is the 38 recipe-lock cases plus the shadow eval.
