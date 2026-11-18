# Plan: scale the realistic promotion suite to 500-1000 cases

## Goal

Grow the promotion suite from 78 to between 500 and 1000 cases. The added
cases keep the rules of `PLAN_EVAL_NATURAL_LANGUAGE.md`: natural,
STT-shaped commands, accurate expected answers, a grader faithful to HA,
minimal device-state checks, and no exclusion.

## Approach

- Rows are produced by a deterministic, seeded generator checked into
  `evals/`. No LLM writes or judges rows.
- The generator reads the household fixtures and crosses phrasing
  templates with each household's devices. Expected calls are derived
  from the fixture, so every expected answer comes from the home it runs
  in.
- The 78 hand-reviewed cases stay unchanged and remain in promotion.
- Generated cases go to a separate file with their own id prefix and
  `provenance.source` pointing at the generator.

## Reference forms (every device type)

- **Full name:** "Turn off the Hallway Ceiling Light."
- **Exclusive alias:** "Turn on the reading lamp."
- **Room plus type:** "Close the bedroom blinds." Used only when exactly
  one device of that type is in the room.
- **Satellite-room short reference:** "Turn the fan on." Used only when
  exactly one device of that type is in the satellite area. The expected
  call follows the existing conventions in `realistic_v3.jsonl`.
- **Room plus domain group:** "Turn off the kitchen lights." The expected
  call is an area+domain call, as HA executes it.

## Commands per domain

Only intents in the production catalog (`docs/HA_2026_11_TOOLS.md`):

| Domain | Commands |
|---|---|
| light | on, off, brightness %, color |
| fan | on, off, speed % |
| cover | open, close, position |
| lock | lock, unlock |
| climate | set temperature |
| media_player | pause, resume, next, volume set, volume up/down, mute |
| switch | on, off |
| vacuum | start, return to base, clean area |
| scene, script | activate or run |
| todo | add item |
| timers | start, cancel, query, with and without a label |

Multi-action cases combine two single commands in one sentence. Status
cases use the existing `GetLiveContext` rule.

## Phrasing

- Small per-command pools of everyday openers ("turn on", "switch on",
  "can you", "please", a bare imperative), numbers spoken as STT writes
  them ("35%", "68", "12 minute").
- No slang stacks, no invented entity words, no exclusion phrasing.
- Exact-duplicate utterances within one household are rejected.

## Composition target (about 600 total, 78 reviewed included)

- **Action categories:** about 85%, spread across ordinary,
  light_fan_settings, climate, media, timers, routines_vacuum, aliases
  and multi_action.
- **Ambiguity:** about 5%. Clarify only where at least 2 devices really
  match the reference; the candidates come from the fixture.
- **Unavailable:** about 5%. device_absent or script_absent only; the
  absent name must not appear in that home.
- **Status:** no more than 4%.
- **Exclusion:** none.

The exact counts are pinned once generated.

## Households

- Add households 6 to 10, modeled on the existing schema. Each gets a
  different mix of rooms and a satellite area, so short references are
  not always the same room.
- New names and aliases must not collide with training vocabulary. Run
  the training eval-name guard and `excluded_train_utterances()` against
  the new homes.

## Validation (fails the generator, not the run)

Each generated case must pass these checks:

- Every expected call validates against the production tool schemas.
- Every `name` resolves to exactly one entity (by name or alias) in its
  household, and every `area` exists there.
- Clarify cases name at least 2 real candidates; unavailable cases name
  nothing that resolves.
- Regenerating gives a byte-identical file.

## Slices

1. **Generator over the existing 5 homes.** Files:
   `evals/generate_realistic.py`, `evals/cases/realistic_generated.jsonl`,
   `evals/tests/test_generated_cases.py`. The tests cover determinism,
   the validation checks and the composition bounds. Not yet in
   promotion.
2. **Households 6 to 10.** Five new
   `evals/homes/realistic_eval_household_{6..10}.json` files, plus the
   regenerated cases file. Also any registry or test that enumerates
   households (found by the delegate inside the eval tree).
3. **Wire up the suite.** `evals/suites/promotion.yaml` (78 plus
   generated), `evals/config/gates.yaml` if needed. Update the exact pins
   in `tests/test_eval.py`, `tests/test_realistic_v3.py` and
   `evals/tests/test_migration.py`; no assertion is weakened. Smoke stays
   at 24.

## Not touched

`custom_components/`, `training/`, the scorer rules (except a bug the
validator exposes, reported first), `ARCHITECTURE.md`.

## Verification

- `ruff check .`, `ruff format --check .`
- `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`
- `pytest -q tests/test_realistic_v3.py` (one known `jsonschema`
  failure in the read-only venv)
- `python -m evals.generate_realistic --check` shows the file is up to
  date.
- Afterwards, as a sanity check: score v17 and v11 on the new suite.
