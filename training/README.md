# SaySo training pipeline

**Training design:** [docs/TRAINING_PLAN.md](../docs/TRAINING_PLAN.md) is the
source of truth for the MVP dataset contract, example format, LFM target, and
safety boundary. Read that document before changing generators, adapters, or
configs.

This directory holds **operational scripts** that implement the plan. The LFM
path is the MVP target; alternate backends and legacy Home-LLM compatibility
paths exist for migration and comparison only.

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
| LFM train | `python training/scripts/train_lfm.py --config smoke` |
| Evaluate | `python training/scripts/evaluate.py training/evals/adversarial.jsonl` |
| Export GGUF | `python training/scripts/export_gguf.py --checkpoint PATH --dry-run` |
| Verify llama.cpp | `python training/scripts/verify_llamacpp.py --dry-run` |

Target model and tuning defaults are defined in
[TRAINING_PLAN.md](../docs/TRAINING_PLAN.md) (LFM2.5-230M full SFT, 3 production
epochs, `2e-4` learning rate, FP16, no BF16, no Flash Attention).

## Dataset views

The adapter can emit JSONL views from converted examples:

- **lfm**: OpenAI-compatible envelope with JSON-string `function.arguments` for
  LFM tokenizer training; `train_on_turn` marks assistant tool-call and TTS
  turns.
- **sayso**: Alias for the runtime envelope.
- **axolotl**: Compatibility view for alternate experiments; not the MVP path.

See [TRAINING_PLAN.md §3](../docs/TRAINING_PLAN.md#3-lfm-format-and-training-configuration)
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
