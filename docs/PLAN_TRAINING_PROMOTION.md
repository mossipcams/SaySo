# Plan: Local training promotion lifecycle

## Scope

Wrap the existing generator, static preflight, rendered-dataset exporter, Unsloth trainer, and SaySo evaluator in a small local `./sayso` command. Keep every generated corpus as an unpromoted candidate; require recorded validation and canary success before dataset promotion; require final eval success before model promotion. Store local run records under ignored training artifacts and never commit automatically.

## Files to touch

- `sayso` — local command dispatcher and minimal per-run/promotion state transitions.
- `train.sh` — retain a safe compatibility entry point that delegates to guarded full training.
- `training/generators/cli.py` — allow the existing generator to write into a run-specific candidate path.
- `training/scripts/train_unsloth_full.py` — require a dataset promotion record for full training, while preserving canary/production configuration parity.
- `training/scripts/eval_checkpoint_cpu.py` — apply the existing suite gates to final promotion evals.
- `training/README.md` — document the local commands and lifecycle.
- `.gitignore` — ignore local lifecycle metadata/artifacts if existing training artifact rules do not cover them.

## Verification

- Inspect command help and statically validate the existing accepted corpus without starting GPU work.
- Run Python compilation and shell syntax checks for changed entry points.
- Exercise state transition guards with local temporary metadata and confirm full training cannot proceed without an explicit dataset promotion; do not start Unsloth training or alter accepted promotion pointers during verification.
