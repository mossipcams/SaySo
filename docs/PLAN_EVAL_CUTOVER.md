# Eval cutover

Implemented. Canonical evals live under `evals/` with one runner (`evals.cli`),
one scorer, and no leftover execution paths.

## Scope

- One case schema (`evals/cases/*.jsonl` + `evals/homes/`).
- One execution pipeline (`evals/runner.py`) and one scorer (`evals/scorer.py`).
- Model adapters only invoke the model (`in_memory`, `endpoint`).
- Suites are ID lists (`smoke.yaml`, `promotion.yaml`).
- Archive the frozen realistic v2 fixture and report byte-for-byte.
- Migrate unique recipe-lock, quality gold/shadow, grounding, and verified v1
  coverage; then delete the superseded modules.

Out of scope: retraining, regenerating train JSONL, rewriting the locked 120
utterances, changing production parse/validate/area policy.

## Files to add

- `evals/cases.py`, `outcomes.py`, `runner.py`, `scorer.py`, `cli.py`, `README.md`
- `evals/adapters/{__init__,in_memory,endpoint}.py`
- `evals/cases/{realistic_v3,regressions}.jsonl`
- `evals/homes/*.json`
- `evals/suites/{smoke,promotion}.yaml`
- `evals/config/gates.yaml`
- `evals/archive/realistic_eval_20260908_v2.{json,md}`
- `evals/results/.gitkeep`, `evals/baselines/.gitkeep`
- `evals/tests/{test_cases,test_runner,test_scorer,test_migration,test_cli}.py`

## Files to delete (after coverage verification)

- `evals/{contract,realistic_v3,migrate_realistic_v2,live,compare,metrics}.py`
- `evals/cases/{realistic_v3.json,v1.json}`
- `evals/config/promotion_gates_v1.json`
- `training/evals/{recipe_lock,v3_quality,grounding_eval,specs,harness,eval_contract,metrics,llamacpp,lfm_python_parse,adversarial.jsonl,__init__.py}`
- `training/evals/test_*.py`
- `training/scripts/{eval_v3_rawparse,generate_v3_quality_eval,generate_recipe_lock_eval}.py` and their tests
- `training/fixtures/realistic_eval_20260908_v2.{json,md}` (moved to archive)

## Callers to update

- `training/generators/{pipeline,homes}.py` holdout sets
- `training/scripts/generate_training_supplement.py` recipe-lock overlap
- `training/scripts/verify_llamacpp.py` production parser
- `training/tests/{test_harness,test_lfm_python_parse,test_llamacpp_parse,test_v3_quality,test_homes,test_generator_pipeline,test_generate_training_supplement}.py`
- `training/generators/{test_coverage_gates,test_grounding_v2}.py`
- `tests/{test_eval,test_realistic_v3}.py`
- `AGENTS.md`, `ARCHITECTURE.md`, `training/README.md`, `docs/TRAINING_PLAN.md`,
  `pyproject.toml`, `.gitignore`, CI live-llama job

## Verification

- Archive SHA-256 matches the pre-move v2 files.
- Promotion is exactly the original 120 IDs, 10 per category.
- Suite IDs resolve uniquely; smoke covers the required behaviors.
- Migrated cases keep expected behavior; labels validate on production schemas.
- Eval rendering matches production for identical inputs.
- Malformed ≠ abstention; refusal ≠ clarification; status mutations, excluded
  targets, and incomplete multi-action fail.
- Both adapters score through `evals.scorer`.
- No remaining references to deleted runners.
- `pytest tests evals/tests` and training tests without model downloads.
