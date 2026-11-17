# Plan: eval tool-call validity and case fixes

Source: `docs/EVAL_AUDIT.md` (2026-11-17). Correction to that audit: 10 of the 12
`GetLiveContext` "shape" findings are false positives because
`evals/scorer.py:_canonical` already turns a string `domain` into a list. Only
`recipe_lock_06_a` and `recipe_lock_06_c` (name only, no domain) remain.

## Scope

1. Existence: every expected call names a tool in the production catalog
   for its household (the 27 schema tools in `schemas/sayso-tool-schema-v2.json`,
   namespaced as the runner does, plus per-home scripts).
2. Accuracy: every expected call's argument keys exist in that tool's schema,
   values match schema types and enums, `name` resolves to an entity name or
   alias in the household, `area`/`floor` exist in the household, and
   `domain`/`device_class` match the resolved entity.
3. Fix the wrong expected answers (audit Section 1, issue 1), in place.

Out of scope: prompt parity (issue 2), new coverage cases (issue 3), training
generator changes, scorer changes.

## Slices

### Slice 1: validator test (one new file)

- `evals/tests/test_case_tool_calls.py`: loads every case in `evals/cases/*.jsonl`
  with its home via the existing loaders in `evals/cases.py` / `evals/runner.py`,
  and asserts points 1 and 2 above. Reuse existing loaders and the runner's
  catalog builder; do not import `training/`.
- Expected to fail on current data. The delegate reports the failing case IDs
  and reasons; the test is not weakened to pass.

### Slice 2: fix cases

Decision rules (in order):

- **A-feature (24 cases):** for per-case homes (`v3_shadow_*`, `v3_gold_*`)
  where the category names the action (media_mute, media_pause, ...), add the
  missing feature to the entity's `capabilities` in the home. Allowed only if
  `capabilities` is not rendered into the eval prompt or catalog (check the
  runner); otherwise stop and report. For homes meant to be incapable
  (`grounding_*_incapable`), confirm intent from
  `git show 98ae012^:training/evals/grounding_eval.py`; if the intent was a
  refusal, change the expected answer to the refusal shape used by sibling
  unsupported cases.
- **B (8 realistic TV cases) and C (7 lock cases):** remove `device_class` from
  the expected arguments; leave homes unchanged so past outputs can be rescored.
- **D (named timers):** no change; names are valid HA arguments. This is a
  training gap, recorded only.
- **E (11 timer cases) and F (1):** make the same phrasing produce the same
  arguments. When the utterance names an area that exists in the household and
  the timer tool's schema accepts `area`, expect `{"area": ...}` (matching
  `v3_quality_gold_timer_cancel_named` and `training/generators/gold.py`).
  For F, use the area that the utterance names and the household defines, or
  drop `area` if neither applies.
- **G (2 cases):** add the entity's `domain` to the `GetLiveContext` call.
- Any further failures from the slice 1 test are fixed by the same rules, or
  listed as unresolved if no rule applies.

Files: `evals/cases/realistic_v3.jsonl`, `evals/cases/regressions.jsonl`,
`evals/cases/context_lookalikes.jsonl` (only if the test flags it),
`evals/homes/v3_*.json` and `evals/homes/grounding_*.json` (capability fixes only).

Stop if a test pins case content (hash, fingerprint, lock file) and the
change would require editing that test.

## Verification

- `pytest evals/tests` passes, including the new slice 1 test.
- `pytest tests --ignore=tests/test_realistic_v3.py` passes.
- `ruff check .` and `ruff format --check .` pass.
- A diff summary lists every changed case ID and its old and new expected call.

## Results

Slice 2 applied with `/tmp/slice2/apply.py` (outside the repo). Every change was
asserted against the pre-edit state before it was written. Diff check: each
changed case differs from HEAD only in `expected.calls` (or `expected` for
`sem_`), and only the listed home files changed.

### Validator (`evals/tests/test_case_tool_calls.py`)

- Added `_is_timer_tool(name)`: `intent__` prefix and `Timer` in the name. The
  8 timer intents in the schema (Start, Pause, Unpause, TimerStatus, Cancel,
  CancelAll, Increase, Decrease) are the only matches. The `name` argument skips
  entity resolution for them. All other checks are unchanged.

### Case changes

| Rule | Case id | File | Old expected call | New expected call |
|---|---|---|---|---|
| B | realistic_eval_20260908_media_01 | realistic_v3.jsonl | `{"device_class": ["tv"], "domain": ["media_player"], "name": "Living Room TV"}` | `{"domain": ["media_player"], "name": "Living Room TV"}` |
| B | realistic_eval_20260908_media_02 | realistic_v3.jsonl | `{"device_class": ["tv"], "domain": ["media_player"], "name": "Lounge Television"}` | `{"domain": ["media_player"], "name": "Lounge Television"}` |
| B | realistic_eval_20260908_media_03 | realistic_v3.jsonl | `{"device_class": ["tv"], "domain": ["media_player"], "name": "Living Room Television", "volume_level": 30}` | `{"domain": ["media_player"], "name": "Living Room Television", "volume_level": 30}` |
| B | realistic_eval_20260908_media_04 | realistic_v3.jsonl | `{"device_class": ["tv"], "domain": ["media_player"], "name": "Sofa TV"}` | `{"domain": ["media_player"], "name": "Sofa TV"}` |
| B | realistic_eval_20260908_media_06 | realistic_v3.jsonl | `{"device_class": ["tv"], "domain": ["media_player"], "name": "Living Room TV"}` | `{"domain": ["media_player"], "name": "Living Room TV"}` |
| B | realistic_eval_20260908_media_07 | realistic_v3.jsonl | `{"device_class": ["tv"], "domain": ["media_player"], "name": "Lounge Television"}` | `{"domain": ["media_player"], "name": "Lounge Television"}` |
| B | realistic_eval_20260908_media_08 | realistic_v3.jsonl | `{"device_class": ["tv"], "domain": ["media_player"], "name": "Living Room Television", "volume_level": 55}` | `{"domain": ["media_player"], "name": "Living Room Television", "volume_level": 55}` |
| B | realistic_eval_20260908_media_09 | realistic_v3.jsonl | `{"device_class": ["tv"], "domain": ["media_player"], "name": "Sofa TV"}` | `{"domain": ["media_player"], "name": "Sofa TV"}` |
| C | recipe_lock_03_b | regressions.jsonl | `{"device_class": ["door"], "name": "Patio Side Door Lock"}` | `{"name": "Patio Side Door Lock"}` |
| C | recipe_lock_03_d | regressions.jsonl | `{"device_class": ["door"], "name": "Joe's Guest Room Door Lock"}` | `{"name": "Joe's Guest Room Door Lock"}` |
| C | recipe_lock_04_b | regressions.jsonl | lock call `{"device_class": ["door"], "name": "Patio Side Door Lock"}` | `{"name": "Patio Side Door Lock"}`; cover call unchanged (`device_class: ["blind"]`) |
| C | recipe_lock_04_d | regressions.jsonl | lock call `{"device_class": ["door"], "name": "Patio Side Door Lock"}` | `{"name": "Patio Side Door Lock"}`; cover and garage calls unchanged |
| C | recipe_lock_05_a | regressions.jsonl | `{"device_class": ["door"], "name": "Basement South Door Lock"}` | `{"name": "Basement South Door Lock"}` |
| C | recipe_lock_05_d | regressions.jsonl | `{"device_class": ["door"], "name": "Joe's Guest Room Door Lock"}` | `{"name": "Joe's Guest Room Door Lock"}` |
| C | recipe_lock_07_h | regressions.jsonl | `{"device_class": ["door"], "name": "Kitchen Back Door Lock"}` | `{"name": "Kitchen Back Door Lock"}` |
| E | v3_shadow_timer_status_0033 | regressions.jsonl | `{}` | `{"area": "Porch"}` ("Porch" is defined in the household) |
| F | v3_shadow_timer_cancel_0038 | regressions.jsonl | `{"area": "Courtyard"}` | `{}` ("Courtyard" is not defined in the household) |
| G | recipe_lock_06_a | regressions.jsonl | `{"name": "Patio South Blinds"}` | `{"domain": ["cover"], "name": "Patio South Blinds"}` |
| G | recipe_lock_06_c | regressions.jsonl | `{"name": "Joe's Guest Room Door Lock"}` | `{"domain": ["lock"], "name": "Joe's Guest Room Door Lock"}` |

### Home changes (A-feature, 23 homes)

`capabilities` gains the one feature the expected call needs, per
`capability_registry.py`. The capabilities field is not rendered into the prompt
or catalog (`evals/runner.py` `exposed_entities` emits only names, domain and
areas, and `production_catalog` is built from the schema and domains).

| Feature | Expected tool | Homes (`evals/homes/`) |
|---|---|---|
| `play` | HassMediaUnpause | v3_gold_media_play, v3_shadow_media_play_0004..0007 |
| `pause` | HassMediaPause | v3_gold_media_pause, v3_shadow_media_pause_0008..0011 |
| `volume` | HassSetVolume | v3_gold_media_volume, v3_shadow_media_volume_0012, 0014, 0016 |
| `volume_step` | HassSetVolumeRelative | v3_gold_media_volume_rel, v3_shadow_media_volume_0013, 0015 |
| `mute` | HassMediaPlayerMute | v3_gold_media_mute, v3_shadow_media_mute_0017..0021 |

Each home changes `capabilities: ["on", "off"]` to `["on", "off", <feature>]`.

### Not changed (decision rules applied, no edit)

- **D (named timers, 8 realistic cases):** no change, per the plan.
- **E (other timer cases):** the `area` rule applies only when the utterance
  names a household area. The households define only `Porch` (and `Sunroom` for
  the gold homes). These cases name a word that is not defined, so their
  expected arguments stay as they were: `v3_shadow_timer_pause_0027..0030`
  (Terrace, Balcony, Cellar, Attic), `v3_shadow_timer_status_0031, 0032, 0034`
  (Cellar, Attic, Veranda), and `v3_shadow_timer_cancel_0035, 0037, 0039`
  (Attic, Veranda, Annex). `v3_shadow_timer_cancel_0036` already expects
  `{"area": "Porch"}`.
- **G, 10 cases flagged by the audit:** no change. `domain` is already a list,
  and `_canonical` treats a string domain as a list, so these are false
  positives. `recipe_lock_06_b`, `06_d`, `v3_quality_gold_status_media`,
  `v3_quality_gold_status_climate`, `v3_shadow_status_0085..0089`, and
  `context_lookalikes_11`.
- **`context_lookalikes.jsonl`:** no change. The test does not flag it.

### Unresolved

- **`sem_e997d4555dc4c991`** (`regressions.jsonl`, home
  `grounding_porch_speaker_incapable`): **not changed**. Expected answer is still
  `HassSetVolume` on Porch Speaker.
  - Intent is refusal. In `git show 98ae012^:training/evals/grounding_eval.py`,
    the case comes from `grounding.supported_action_family`'s `incapable` variant.
    It has `inject_missing=False` and no `volume` feature. For an area-targeted,
    present-but-unsupported entity, `training/generators/gold.py` routes to
    `expected_device_unsupported` (`no_action`, refusal). The refusal shape is
    `response_type: refusal`, `calls: []`, `forbidden_entities: []`, as in
    `sem_4f2ddd1dbb8fd215`. I could not run the generator directly because
    `jsonschema` is missing from the venv, so this rests on the static trace.
  - The refusal change was applied, then reverted. `capabilities` is not rendered,
    so the prompt is identical to `sem_cbb85a396361684e`
    (`grounding_porch_speaker_capable`), which expects the same action.
    `test_identical_rendered_inputs_share_one_expectation` then fails. Resolving
    that needs a change to the capable sibling or a weaker invariant, and both are
    out of scope.
  - The capability fix was not applied. The A-feature rule covers per-case homes
    in action-named categories. Adding `volume` to a home named `incapable` would
    reverse its meaning.
  - Decision needed: keep the action answer (matches the sibling and is the only
    answer the prompt supports), or change both `sem_` cases to refusal with the
    prompt changed to expose capabilities (a separate slice).

### Verification (this change)

- `pytest evals/tests -q`: 419 passed (before: 17 failed, 402 passed).
- `pytest tests -q --ignore=tests/test_realistic_v3.py`: blocked. `tests/conftest.py`
  imports `aiohttp`, which the venv lacks. `--noconftest` fails fixture collection
  (14 errors). `tests/test_eval.py` alone passes with `--noconftest` (4 passed).
  The eval case edits don't touch anything `tests/` reads except through
  `test_eval.py`, which passes.
- `ruff check .`: passed. `ruff format --check .`: passed.

### Results: timer names and cancel areas

Decision from the user. Cancel-all cases add the area to their household, so the
case tests area-scoped cancel. Pause and status cases use the place word as the
area, not the timer name: each household's `satellite_area` is the named place,
matching sibling `v3_shadow_timer_status_0033` (`{"area": "Porch"}`). Only
`arguments` changed. `expected` for every case keeps its `response_type` and
tool. `unavailable` stays `null`.

Pause and status (`evals/cases/regressions.jsonl`, `area` = place word in the utterance):

- `v3_shadow_timer_pause_0027` ("Hold the Terrace countdown timer"): `{}` -> `{"area": "Terrace"}`
- `v3_shadow_timer_pause_0028` ("Hold the Balcony countdown timer"): `{}` -> `{"area": "Balcony"}`
- `v3_shadow_timer_pause_0029` ("Hold the Cellar countdown timer"): `{}` -> `{"area": "Cellar"}`
- `v3_shadow_timer_pause_0030` ("Hold the Attic countdown timer"): `{}` -> `{"area": "Attic"}`
- `v3_shadow_timer_status_0031` ("How much time is left on the Cellar timer?"): `{}` -> `{"area": "Cellar"}`
- `v3_shadow_timer_status_0032` ("How much time is left on the Attic timer?"): `{}` -> `{"area": "Attic"}`
- `v3_shadow_timer_status_0034` ("How much time is left on the Veranda timer?"): `{}` -> `{"area": "Veranda"}`

Home files changed for pause and status (each used only by its own case). Each
adds the place as `satellite_area` with `[]` to `areas` and `area_aliases`, after
the existing `Porch` entry:

- `evals/homes/v3_shadow_timer_pause_0027.json` (Terrace)
- `evals/homes/v3_shadow_timer_pause_0028.json` (Balcony)
- `evals/homes/v3_shadow_timer_pause_0029.json` (Cellar)
- `evals/homes/v3_shadow_timer_pause_0030.json` (Attic)
- `evals/homes/v3_shadow_timer_status_0031.json` (Cellar)
- `evals/homes/v3_shadow_timer_status_0032.json` (Attic)
- `evals/homes/v3_shadow_timer_status_0034.json` (Veranda)

Cancel-all (`area` = area named in the utterance, matching sibling `v3_shadow_timer_cancel_0036`):

- `v3_shadow_timer_cancel_0035` ("Stop every running timer in the Attic"): `{}` -> `{"area": "Attic"}`
- `v3_shadow_timer_cancel_0037` ("Stop every running timer in the Veranda"): `{}` -> `{"area": "Veranda"}`
- `v3_shadow_timer_cancel_0038` ("Clear all timers in the Courtyard"): `{}` -> `{"area": "Courtyard"}`
- `v3_shadow_timer_cancel_0039` ("Stop every running timer in the Annex"): `{}` -> `{"area": "Annex"}`

Home files changed (each used only by its own case, checked across `evals/cases/*.jsonl`).
Each adds the area with `[]` to `areas` and `area_aliases`, after the existing `Porch` entry:

- `evals/homes/v3_shadow_timer_cancel_0035.json` (Attic)
- `evals/homes/v3_shadow_timer_cancel_0037.json` (Veranda)
- `evals/homes/v3_shadow_timer_cancel_0038.json` (Courtyard)
- `evals/homes/v3_shadow_timer_cancel_0039.json` (Annex)

Utterance labels matched the list exactly, so no differences needed reporting.

Verification: `pytest evals/tests -q` 419 passed. `ruff check .` passed.
`ruff format --check .` passed. The test `test_expected_tool_calls_exist_and_are_accurate`
accepts `area` on the pause, status and cancel tools, and checks each `area` against the home.

### Results: duplicate removed

`sem_e997d4555dc4c991` was removed from `evals/cases/regressions.jsonl` because its
rendered prompt is identical to capable sibling `sem_cbb85a396361684e`, while
capabilities are not rendered, so it could never test a refusal.
`evals/homes/grounding_porch_speaker_incapable.json` was deleted: no remaining case
in `evals/cases/*.jsonl` uses that household.
