# SaySo training pipeline

**Training design:** [docs/TRAINING_PLAN.md](../docs/TRAINING_PLAN.md) is the
single source of truth for dataset design, tool contracts, example format,
curriculum, and hyperparameters. Read that document before changing generators,
adapters, or configs.

This directory holds **operational scripts** that implement the plan. Legacy paths
(LFM defaults, Home-LLM vendored piles, locked v1 catalog filtering) exist for
migration and comparison only; they are not the target architecture.

## Quick start

```bash
python -m venv training/.venv
source training/.venv/bin/activate
pip install -r training/requirements.txt
pip install pyyaml   # config validation tests
python -m pytest training/tests -q
```

## Pipeline commands

| Step | Command |
|------|---------|
| Pin upstream (legacy Home-LLM ref) | `python training/scripts/pin_upstream.py` |
| Generate + adapt | `python training/scripts/generate_dataset.py --language english --small` |
| Adapt only | `python training/scripts/adapt_dataset.py INPUT.jsonl OUTPUT.jsonl --seed 42` |
| Split 80/10/10 | `python training/scripts/split_dataset.py INPUT.jsonl --out-dir training/datasets` |
| Detect GPU | `python training/scripts/detect_gpu.py` |
| FunctionGemma train | `python training/scripts/train.py --backend functiongemma --config smoke` |
| Evaluate | `python training/scripts/evaluate.py training/evals/adversarial.jsonl` |
| Export GGUF | `python training/scripts/export_gguf.py --checkpoint PATH --dry-run` |
| Verify llama.cpp | `python training/scripts/verify_llamacpp.py --dry-run` |

Target model and tuning defaults are defined in
[TRAINING_PLAN.md](../docs/TRAINING_PLAN.md) (FunctionGemma 270M full SFT,
~5e-5 LR, FP16, no BF16, no flash-attn on Pascal).

## Dataset views

The adapter can emit JSONL views from converted examples:

- **sayso**: OpenAI-compatible envelope (runtime / llama.cpp transport shape).
- **axolotl**: `function.arguments` parsed to dicts for the FunctionGemma Jinja
  template; `train_on_turn` marks assistant tool-call and TTS turns.

See [TRAINING_PLAN.md §3](../docs/TRAINING_PLAN.md#3-functiongemma-native-format)
for the canonical on-disk format and template tokens.

## Layout

```
training/
  adapters/          Example conversion and schema validation
  artifacts/         Checkpoints, eval outputs (gitignored)
  configs/           Axolotl YAML
  datasets/          Generated JSONL (gitignored)
  evals/             Metrics, harness, adversarial set
  fixtures/          Test fixtures
  generators/        Label-first example generation (per TRAINING_PLAN)
  scripts/           Pipeline operations
  tests/             Unit tests (no model downloads)
```

## GPU notes (GTX 1070 / Pascal)

Run `python training/scripts/detect_gpu.py` before training. Use FP16; disable
BF16 and flash-attn. If no GPU is present, `train.py` exits 0 with a documented
skip message.

## Adversarial eval

`evals/adversarial.jsonl` is held out from training and checkpoint selection.
