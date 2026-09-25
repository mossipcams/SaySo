# SaySo local training preflight

## Scope

Add one fail-fast local workflow that builds the existing v5 corpus, validates
it against a small checked-in policy and the pinned SaySo schema, runs a
250-step Unsloth canary with the production training path, scores that
checkpoint through the existing eval runner, and starts full training only if
all earlier stages pass. Keep baseline updates explicit.

## Files

- `train.sh` — generation, preflight, rendered-view export/copy, canary, eval,
  full training through the VM's GPU lock and Unsloth container.
- `scripts/preflight.py` and `training/configs/preflight.yaml` — JSONL checks,
  capability/fixture requirements, duplicate and drift thresholds, summary.
- `training/configs/training_baseline.json` — accepted dataset profile and
  accepted eval metrics with regression tolerances.
- `training/scripts/train_unsloth_full.py` — `--canary` mode using the exact
  production model, tokenizer, encoding, masks, optimizer, batching, and other
  settings; only steps and output directory differ.
- `training/scripts/eval_checkpoint_cpu.py` — invoke the existing smoke,
  recipe-lock, gold, and gauntlet cases through `evals.runner` and the shared
  scorer, then compare metrics with the accepted baseline.
- `training/tests/test_preflight.py` — synthetic pass/fail cases for the
  requested dataset invariants.
- `training/README.md` — normal workflow commands and explicit baseline update.

## Verification

- Run `python scripts/preflight.py` against the current generated candidate.
- Run `training/.venv/bin/python -m pytest training/tests/test_preflight.py -q`
  and the focused existing schema/eval tests.
- Inspect the canary CLI and compare its `TrainingArguments` with production;
  verify the canary does not slice or reorder the training dataset.
- Run shell syntax validation for `train.sh` and Python compile checks.
- Do not launch GPU training as part of repository verification.
