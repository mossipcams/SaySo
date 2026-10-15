# Plan: live-like static context in the eval gate

## Problem
Production SaySo-LFM-v5b sent HassTurnOn/HassTurnOff with `domain: sensor` for the TV
(media_player) and Living room lights (switch), and once targeted a person entity. Home
Assistant's Assist static context lists non-controllable entities (sensor, binary_sensor,
person, sun, select, ...) next to devices, e.g. `TV` followed by `TV Active app` (sensor).
The training corpus never contained such entries, and neither does the eval gate:
`evals/runner.py::exposed_entities` renders only `home["entities"]`, no eval home has
context-only entries, a wrong domain is only one failed case against a 70-80% pass rate,
and nothing gates on targeting a non-controllable entity. The gate cannot see the failure.

## Scope
1. Runner: `exposed_entities` also renders `home["context_entities"]` (name, domain,
   optional area) sorted by name together with the entities, the same rule the
   ajax-optimization generator uses. Homes without the key render exactly as before, so
   existing suites and baselines are unchanged.
2. Eval home `evals/homes/live_home_1.json`: the redacted real home (21 devices + 27
   context-only entries; persons redacted to Person One/Person Two). A test asserts it
   renders the captured live Assist static-context block byte-for-byte
   (`evals/tests/fixtures/live_home_1_static_context.yaml`).
3. Scorer: zero-tolerance flags `wrong_domain` (an action/status call naming an entity of
   the case home whose `domain` argument excludes that entity's domain) and
   `context_entity_targeted` (an action call whose name matches a context-only entry).
   Added to the smoke and promotion gates and to the new suite.
4. Cases `evals/cases/context_lookalikes.jsonl` (tags `regression`, `context`; never
   `gold`/`shadow`, so training does not reserve these entity names), registered in
   `_CASE_FILES`, and suite `evals/suites/context.yaml` with gate `min_overall_pass_rate:
   1.0` plus the zero-tolerance flags. The locked 120-case promotion suite is unchanged.

## Files
evals/runner.py, evals/scorer.py, evals/cases.py, evals/config/gates.yaml,
evals/homes/live_home_1.json, evals/cases/context_lookalikes.jsonl,
evals/suites/context.yaml, evals/tests/ (new tests + fixture), this plan.

## Verification
- `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`, `ruff check .`,
  `ruff format --check .`
- live_home_1 renders the captured live block exactly.
- Scorer flags fire on a synthetic wrong-domain / context-target turn and stay silent on
  correct ones; existing suites' case lists and fingerprints are unchanged apart from the
  new cases.

## Follow-up outside this repo
ajax-optimization runs and gates the `context` suite in `./sayso canary` and
`./sayso eval`, and bumps its SaySo submodule to the merged commit.
