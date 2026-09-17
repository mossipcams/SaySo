# Plan: Fix HA 2026.8.3 `jsonschema` CI failure

## Failure

- Check: **HA 2026.8.3**
- Job: https://github.com/mossipcams/SaySo/actions/runs/35245652891/job/105285049223
- Head: `98ae012cb2bcb0d280d123969166742413e8830e`
- Step: Run compatibility tests (`scripts/compat_matrix.py`)
- Result: `1 failed, 170 passed`
- Failed test: `tests/test_eval.py::test_shared_runner_scores_a_perfect_smoke_case`
- Error: `ModuleNotFoundError: No module named 'jsonschema'` at `training/adapters/schema.py:11`

Import chain from the job log:

```text
tests/test_eval.py
  → evals/runner.py:evaluate/render_case
  → generators.context
  → training/generators → training/adapters/schema.py
  → import jsonschema
```

Other PR #74 checks at this head passed (Conventional PR title, Satellite unit tests). Live llama.cpp latency was skipped.

## Cause

This is caused by `ajax/evals`. The shared eval runner now pulls the training generator package, which imports `jsonschema`. The HA compatibility venv does not install that package.

## Scope

- `evals/runner.py` and related eval helpers used by `tests/test_eval.py`
- `tests/test_eval.py`
- dependency/extra declarations actually installed by HA compat CI (`pyproject.toml`, `scripts/compat_matrix.py`)
- `training/adapters/schema.py` only if a lazy/optional import is the smallest correct fix

Do not change satellite runtime, HA conversation wiring, or unrelated tests.

## Approach

Use the smallest change that makes the HA compat pytest set pass:

1. Prefer keeping the HA test venv free of training-only deps: stop `evals/runner.py` from importing the training package tree just to render a case, if a local/eval-owned path already exists.
2. If `jsonschema` is a real requirement of the shared runner in this CI job, add it to the extras/deps that `scripts/compat_matrix.py` installs — not as a Home Assistant runtime integration dependency unless runtime already needs it.
3. Do not weaken fail-closed eval scoring or skip the failing test.

## Verification

- `pytest tests/test_eval.py`
- Repository local verification gate (the same pytest subset CI runs if the HA extra is available; otherwise the closest local equivalent)
- Confirm `jsonschema` is no longer required at import time for `test_shared_runner_scores_a_perfect_smoke_case`, or is installed by the HA compat extra
