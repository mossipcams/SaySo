# Plan: thin generator pipeline to orchestration only

**Status:** implement this increment of `docs/PLAN_GENERATOR_REFACTOR.md` §3.

Parent-owned plan. Do not train, do not regenerate a 40k corpus, do not change
runtime or the locked eval suite.

## Scope

`training/generators/pipeline.py` is still a god file (~1147 lines) with
`generate_row`, scenario-specific wording, grounding/discrimination slot
picking, uniqueness, and the accept loop. Target analogue is `evals/runner.py`
(~400 lines of orchestration).

Keep one generation loop. Recipe `FamilyTracker` remains the production
planner. `QuotaTracker` may remain only as the planner when the recipe has
empty `allocations:` (existing tests). Row construction must not branch on
planner type except for slot selection.

## Files to touch

- `training/generators/pipeline.py` — orchestration only (`run_generation`,
  `_run_generation`, rate gates, `run_build`, writes)
- `training/generators/scenarios/` — facts / area / uniqueness / unavailable
  request templates that currently live in pipeline
- `training/generators/` helpers as needed (`generation.py` only if no existing
  module fits; prefer `scenarios/`, `grounding.py`, `utterances.py`,
  `manifest.py`)
- `training/tests/` — update imports if helpers move; add no new framework
- `training/configs/generation/smoke.yaml` — only if family-vs-outcome
  clarify/area rounding must change to keep smoke feasible

Do not touch: `evals/`, `custom_components/sayso/` runtime, full production
YAML counts unless a gate is currently unsatisfiable at smoke size.

## Out of this increment

- Deleting leftover training scripts that are not in the generator loop
- TRAINING_PLAN allocation rewrite
- Production 40k regen

## Verification

1. `PYTHONPATH=training python3 -m pytest training/tests -q --ignore=training/tests/test_lfm_python_parse.py --ignore=training/tests/test_llamacpp_parse.py`
2. `cd training && PYTHONPATH=. python3 -m generators.cli --config configs/generation/smoke.yaml --dry-run`
3. `wc -l training/generators/pipeline.py` should be in the same order as
   `evals/runner.py` (orchestration), not 1000+ lines of row construction
4. `git diff -- evals/` empty
5. No leftover `_run_family_generation` / `_run_quota_generation` names
6. Area rows still go through the same `generate_row` → validate → duplicate
   path (no post-loop append)
