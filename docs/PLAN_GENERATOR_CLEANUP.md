# Generator cleanup plan

## Scope

- Make `training/generators/` the only synthetic training-row implementation.
- Remove the retired model verbalizer/judge pipeline and its v1/v2 scenario and
  rendering modules.
- Migrate active eval and held-out-data entry points to the deterministic
  generator modules.
- Remove documentation and training-config references to retired generator
  paths. Keep the runtime, locked evals, shadow evals, and held-out test
  generation behavior covered by the current generator.
- Rename the active generator home field from `sayso_entity_area` to
  `satellite_area`; do not retain a fallback alias.

## Files to touch

- Delete `training/scripts/build_synthetic_dataset.py`, `llm_curation.py`,
  `v2_scenarios.py`, `rendering.py`, and `generate_training_supplement.py`.
- Update `training/evals/recipe_lock.py`, `training/evals/v3_quality.py`, and
  generator scripts to import canonical render/write helpers.
- Replace the balanced-test script's v1/v2 spec dependency with the current
  pipeline.
- Update active fixtures and eval builders to use `satellite_area` directly.
- Remove obsolete v2 training configuration and update training documentation.
- Retire tests that solely exercise deleted paths and keep the equivalent
  current-generator coverage.

## Verification

- `python3 -m pytest training/generators`
- `python3 -m pytest training/tests`
- Run the active eval and held-out generators with small deterministic outputs.
- `python3 -m compileall -q custom_components/sayso training/generators training/scripts`
- `git diff --check` and targeted Ruff checks.
- Confirm no active production or user-facing documentation references to the
  removed generator modules remain.
