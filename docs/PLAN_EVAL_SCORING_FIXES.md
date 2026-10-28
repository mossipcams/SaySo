# Plan: eval scoring and case-integrity fixes

## Why

Reading all 91 v12 canary failures (211-case set) showed about 13 scorer
artifacts. Six came from `GetLiveContext` domain str vs list: main already
fixes that, but the VM `ajax-optimization` eval copy doesn't have the fix, so
it is out of scope here. Five came from running a script via `HassTurnOn`, and
two from alias names. The rest are real model errors. Separately, five
`HassTimerStatus` cases are labelled `action`, the Porch Speaker pair
contradicts itself (capabilities never reach the prompt or tools), and run
metadata does not hash what the model actually saw.

## Scope

Slice A, the scorer:

- `evals/scorer.py`
- `evals/tests/test_scorer_equivalence.py` (new)

Slice B, cases, fingerprint, and invariant:

- `evals/cases/regressions.jsonl`
- `evals/runner.py`
- `evals/tests/test_case_invariants.py` (new)

Out of scope: VM `ajax-optimization` sync, grounding expansion, gate
thresholds, refusal/clarification regex, `migration_map.json`, and any
existing test.

## Changes

Slice A:

1. An actual action call's `name` matches the expected `name` when both
   resolve, case-insensitively, through name or aliases to the same single
   entity in the case household. If a name resolves to more than one entity,
   it does not match.
2. An expected script tool call (`<object_id>` with `{}`) is matched by
   `intent__HassTurnOn` whose `name` resolves to that script entity, with
   `domain` absent, `"script"`, or `["script"]`.
3. Everything else stays strict: tool name, other arguments, domain, flags.

Slice B:

1. Relabel `v3_quality_gold_timer_status` and
   `v3_shadow_timer_status_0031`..`0034` from `response_type: action` to
   `status`.
2. `sem_e997d4555dc4c991` (Porch Speaker, capability hidden) expects the same
   `HassSetVolume` action as `sem_cbb85a396361684e`, because the model cannot
   see capabilities and the integration's capability barrier handles the
   refusal.
3. `write_run` metadata gains `input_hash`: sha256 over each case's rendered
   messages and tool catalog. `case_hash` is unchanged so historical runs stay
   comparable.
4. Invariant test: cases with identical rendered messages and tools must have
   identical `(response_type, calls)` expectations.

## Verification

- `ruff check .` and `ruff format --check .`
- `pytest tests evals/tests --ignore=tests/test_realistic_v3.py -q`
- Parent re-scores the v12 canary raw outputs with the new scorer and
  confirms that only the expected cases flip.
