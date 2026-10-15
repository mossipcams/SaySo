# Plan: eval runner matches production prompt, context gate gets real failures

## Why

Production (HA 2026.10.1, pipeline "SaySo", local intent handling off) failed
simple on/off commands. The eval runner must render what production sends and
the `context` gate must include the exact failing commands.

## Scope

- `evals/runner.py`: `DYNAMIC_CONTEXT_PROMPT` renders the production Assist
  paragraph verbatim (same line breaks). It says "retrieving live context" and
  "use the provided tool to retrieve live context" and no longer names
  `GetLiveContext`. Nothing else in the prompt changes.
- `evals/cases/context_lookalikes.jsonl`: append ids 12-16 verbatim from
  `/tmp/sayso_slices/context_cases_live.jsonl` (household `live_home_1`, tags
  `regression`, `context`, `production`).
- `evals/suites/context.yaml`: list ids 12-16.
- `evals/tests/test_context_suite.py`: suite size 11 -> 16; cases tagged
  `production` are exactly ids 12-16; every case scores clean against its own
  expected calls; prompt wording asserted (both phrases present, no
  `GetLiveContext`).

## Constraints

- No code comments or docstrings.
- `evals/**` is excluded from ruff format; do not reformat unrelated lines.
- Repo is public: only redacted names, no person names in added content.
- Do not modify other tests or weaken assertions.

## Verification

- `pytest evals/tests -q`
- `ruff check .` and `ruff format --check .`
- `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`
- `pytest -q satellite/sayso`

## Delivery

Commit, push, open PR against main, report URL to ajax-model-training-failure-a4.
ajax-optimization bumps its SaySo submodule to the merged commit before
generating the v6 corpus (training excludes eval utterances by exact match).
