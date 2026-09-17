# Plan: delete leftover dual generator entry points

**Status:** implement this increment of `docs/PLAN_GENERATOR_REFACTOR.md` §3
(delete leftover dual names after the move).

Previous increment thinned `pipeline.py` to orchestration (~441 lines). Dual
files named in the original plan are still on disk.

## Scope

Delete leftover dual names with no wrappers, aliases, or fallbacks. Update
callers to the canonical CLI/config. Keep useful deterministic coverage by
moving it first if it is not already in `scenarios/` / `validation.py` /
`deduplication.py`.

Delete (if still present and unused after caller updates):

- `training/generators/area_scenarios.py` (replaced by `scenarios/area.py`)
- `training/generators/duplicates.py` (replaced by `deduplication.py`)
- `training/generators/validate.py` (replaced by `validation.py`)
- `training/generators/validator.py`
- colocated `training/generators/test_*.py` (tests live in `training/tests/`)
- `training/scripts/build_synthetic_dataset.py` v1/v2 LLM path and related
  `v2_scenarios.py`, `llm_curation.py`, `scripts/rendering.py`
- argparse-only second CLI; `--allow-rate-shortfall`; mixed-contract flags
- `generate_training_supplement.py` if it is only a second generator
- `generate_balanced_test_data.py` must call canonical `generators.cli` /
  config if it still needs to exist

Update `training/README.md` commands to `python -m generators.cli --config ...`.
`docs/TRAINING_PLAN.md` only if command/design-map paths changed.

## Do not

- Train, regenerate 40k, edit `evals/`, or change `custom_components/sayso`
- Rewrite allocation math or smoke counts unless a leftover script was the
  only thing keeping a test alive

## Verification

1. `PYTHONPATH=training python3 -m pytest training/tests -q --ignore=training/tests/test_lfm_python_parse.py --ignore=training/tests/test_llamacpp_parse.py`
2. `cd training && PYTHONPATH=. python3 -m generators.cli --config configs/generation/smoke.yaml --dry-run`
3. `git diff -- evals/` empty
4. No remaining imports of deleted module names
5. One CLI: `python -m generators.cli`
