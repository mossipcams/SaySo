# Plan: natural-language promotion/smoke evals, no exclusion

## Problem

The 120-case promotion suite over-weights edge behavior and uses
template language nobody says to a voice assistant:

- 78/120 utterances say the full Title Case HA entity name
  ("Have Downstairs Robot Vacuum start cleaning, please.").
- Filler from templates: "for me", "please", "now", "for cooking",
  "when you can", "before I leave".
- 40/120 are edge behavior: exclusion 10, ambiguity 10 (9 are two
  prompts reworded), unavailable 10 (6 are synthetic `tool_removed`
  catalogs), multi_action 10 pairing unrelated devices.
- Implicit-area commands ("turn off the lights" from the Kitchen
  satellite) are the most common real request and barely covered.

## Scope

Evals only. Training generator exclusion scenarios and the scorer's
`forbidden_entities` / `excluded_entity_affected` machinery are out of
scope (separate follow-up; the scorer change touches `tests/`).

### Changes

1. Remove the exclusion category entirely from eval cases and suites:
   - Delete the 10 `realistic_eval_20260908_exclusion_*` cases.
   - Delete the 4 `recipe_lock_04_*` `multi_action_exclusion` regressions.
   - Remove `exclusion` from `evals/config/gates.yaml`.
   - Mark the corresponding `evals/tests/migration_map.json` rows as no
     longer retained (status other than `retained`) so the map stays
     consistent.
2. Trim edge categories in promotion:
   - ambiguity 10 -> 4: keep distinct prompts only; drop
     `ambiguity_10` ("run the routine for me"), duplicate rewordings,
     and cases whose candidates are not plausible matches for the
     phrase (e.g. "reading light" -> TV Backlight).
   - unavailable 10 -> 3: keep only `device_absent` / `script_absent`
     (garage heater, driveway gate, Holiday Mode). Drop all
     `tool_removed` cases and `unavailable_02` (expects an action).
   - multi_action 10 -> 6: keep/rewrite only natural pairings (same
     room, bedtime/leaving routines, e.g. "turn off the TV and the
     floor lamp", "lock the front door and turn off the lamp").
3. Rewrite remaining utterances as natural speech. Expected calls stay
   identical (same tool, same full entity name, same arguments):
   - lowercase device references with articles: "turn on the kitchen
     counter lights".
   - use the short form people say when it is unambiguous in that
     case's household (`evals/homes/<household>.json`): "start the
     vacuum", "set the thermostat to 67", "is the front door locked?",
     "pause the TV". If the short form matches more than one entity of
     that domain in the household, keep a distinguishing word.
   - remove template filler; keep "please"/"can you" on at most ~1 in 5.
   - fix the odd phrasings: "for Hallway Heating, use 72 degrees",
     "run Bedroom Tower Fan at a quarter of its full speed", "put Living
     Room TV back on play", "set the white temperature of ... to 2700
     kelvin", "Please hold the pasta timer".
   - aliases category: keep using the alias phrases; just drop filler.
4. Add ~7 implicit/area cases to `ordinary` (new ids
   `realistic_eval_20261118_area_NN`, tags `promotion`, provenance noting
   they were authored 2026-11-18): "turn off the lights", "turn on the
   lights" resolved via the household `satellite_area`, and explicit-area
   forms like "turn off the bedroom lights". Follow the existing
   expected-call shape already used by cases whose
   `target_area_source` is `satellite`/area-based; do not invent a new
   expected format.
5. Smoke stays 24 cases, all drawn from promotion: replace the 2
   exclusion cases, `ambiguity_10`, and `unavailable_01` with area cases
   and a `device_absent` unavailable case. Smoke must still include
   ordinary, aliases, multi_action, status, ambiguity, unavailable, and
   both explicit and satellite area sources.
6. Update `evals/tests/test_migration.py` to the new composition
   (exact new promotion count and per-category counts, no `exclusion`
   in smoke's required set). This is a direct consequence of removing
   the category, not a weakening: assertions stay exact.

Resulting promotion: ordinary 17, light_fan_settings 10, climate 10,
media 10, timers 10, routines_vacuum 10, status 10, aliases 10,
multi_action 6, ambiguity 4, unavailable 3 = 100. Edge behavior is
7/100 instead of 40/120.

Scores against this suite are not comparable to v7/v11/v12 promotion
history; git history keeps the old suite.

## Files to touch

- `evals/cases/realistic_v3.jsonl`
- `evals/cases/regressions.jsonl`
- `evals/suites/promotion.yaml`
- `evals/suites/smoke.yaml`
- `evals/config/gates.yaml`
- `evals/tests/migration_map.json`
- `evals/tests/test_migration.py`

Not touched: `evals/baselines/`, `evals/results/`, `evals/archive/`,
scorer/runner code, `tests/`, `training/`.

## Verification

- `ruff check .` and `ruff format --check .`
- `pytest evals/tests`
- `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`
- `rg -n -i exclusion evals/cases evals/suites evals/config` is empty.
- Every promotion case's expected entity names exist in its household
  file; every short-form utterance resolves to exactly one entity of the
  expected domain in that household (one-off script, not committed).
- Parent reads every changed utterance in the delta.

# Phase 2: accurate homes, HA-faithful scoring, fewer state checks

Slices 1-3 above are applied (uncommitted). An audit found the cases
are still inaccurate and partly unnatural.

## Findings

- Household fixtures are one template copied five times with aliases
  pasted regardless of device: "sofa lamp" -> Living Room Ceiling Light
  while a Sofa Reading Lamp exists (household 2), "coffee plug" ->
  Toaster Plug / Kettle Plug, "reading light"/"sofa lamp" -> TV
  Backlight, "my bedside light" pinned to one person although the
  assistant cannot know the speaker. Ambiguity exists only because the
  template gives two lamps the same alias. Satellite is always Kitchen.
- The scorer requires exact arguments where Home Assistant accepts
  more: `GetLiveContext` args are all optional; an omitted `area` on a
  satellite-area command resolves to the satellite area; relative
  volume accepts an integer step; a spoken clarification names the
  distinguishing word ("Jamie's or Robin's?"), not full entity names.
- Wording leftovers: kelvin values, "turn the bedroom bedside lamp
  blue", "turn the television volume to 55 percent", near-duplicate
  cases (10 thermostat setpoints, paired "pause the television" /
  "can you pause the television?", "mute the TV" / "can you mute the
  TV?", "turn off the lights" / "turn the lights off").
- Device-state questions (status, 10 cases) are over-weighted; they
  should be minimal.

## Slice 4: scorer accepts what HA accepts

Files: `evals/scorer.py`, a test module under `evals/tests/`.

- Status: an expected STATUS case passes when the turn makes only query
  calls, at least one is `homeassistant__GetLiveContext`, and every
  argument it supplies is consistent with the expected entity (name
  equals the entity name or one of its aliases, domain contains the
  entity domain, area equals the entity area). No-argument
  `GetLiveContext` passes.
- Area: when the case's target area came from the satellite, an action
  call that omits `area` matches an expected call whose `area` equals
  that satellite area (all other arguments equal). A wrong explicit
  area still fails and still flags `wrong_target_area`.
- Relative volume: expected `volume_step` `"down"`/`"up"` also matches
  a negative/positive integer step.
- Clarification: passes when the reply is a clarification and, for
  every candidate, the reply contains the full name, or one of that
  candidate's aliases not shared with another candidate, or a name word
  that no other candidate has (e.g. "jamie's").
- Nothing else gets more lenient: wrong entity, wrong tool, action on a
  status or ambiguity case, malformed output and unknown tools still
  fail with the same flags.

## Slice 5: accurate homes and natural cases

Files: `evals/homes/realistic_eval_household_{1..5}.json`,
`evals/cases/realistic_v3.jsonl`, `evals/suites/promotion.yaml`,
`evals/suites/smoke.yaml`, `evals/tests/test_migration.py`,
`tests/test_eval.py`, `tests/test_realistic_v3.py`,
`evals/tests/test_live_home_context.py` if it pins fixture content.

- Aliases: only aliases a person would give that device. Remove
  "my bedside light" everywhere. An alias is shared by two devices only
  where that is realistic (two bedside lamps both called "bedside
  lamp").
- Vary satellite areas across homes (Kitchen, Living Room, Bedroom at
  least). Where a home gains a second device of a type (e.g. a Bedroom
  TV), "turn off the TV" said in the room with one TV resolves to that
  TV.
- Every case re-derived from its fixed home: the expected entity is the
  one a person would mean; ambiguity cases only where the home really
  has two equally good matches.
- Composition target (exact counts pinned in tests):
  ordinary 20 (7 area/satellite commands, 2 "add X to the shopping
  list" via `todo__HassListAddItem`), light_fan_settings 8 (no kelvin),
  climate 5, media 8, timers 8, routines_vacuum 8, status 3, aliases 6,
  multi_action 6, ambiguity 3, unavailable 3 = 78.
- Remove near-duplicates first; each case is a distinct command.
- Only add commands whose correct call shape is documented in
  `docs/HA_2026_11_TOOLS.md`. "turn off all the lights" (whole house)
  is skipped: its domain-only semantics with a satellite area are not
  documented.
- Smoke: 24 promotion cases, at most 1 status, still covering ordinary,
  aliases, multi_action, status, ambiguity, unavailable and both
  explicit and satellite area sources.

## Verification

- CI gate: `ruff check .`, `ruff format --check .`,
  `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`,
  `pytest -q satellite/sayso`.
- `pytest -q tests/test_realistic_v3.py` (perfect run passes every
  case and gate).
- One-off script: every expected entity exists in its home with the
  expected domain; every referenced alias resolves; every ambiguity
  case's candidates are exactly the home's matches for the phrase.
- Parent reads every scorer hunk and every case.
- Follow-up outside this plan: rerun the training eval-name guard
  against the changed homes (new names can leak as substrings), and run
  a real checkpoint on the VM to confirm the scorer changes.

# Phase 3: grader correctness, STT-shaped input, real usage

Deeper audit after Phase 2 found:

- The slice 4 satellite-area rule is wrong for area-only calls. HA's
  preferred area only breaks ties (`docs/HA_2026_11_TOOLS.md:192`), so
  `HassTurnOff(domain=light)` with no area and no name targets every
  light in the home, yet scores VALID on "turn off the lights".
- Natural refusals ("I couldn't find a garage heater.", "Sorry, there
  isn't a garage heater.") and clarifications ("Jamie's or Robin's
  bedside lamp?", "The floor lamp or the table lamp?") fail the cue
  regexes.
- `tests/test_realistic_v3.py` strips id prefixes, so the four
  `20261118_ordinary_*` ids overwrite `20260908_ordinary_01..04` in its
  `CASES` map; the format tests at lines 298-309 now run on a different
  case.
- Status has 3 cases with a 0.7 gate, so one miss fails promotion.
- Case text is lowercase, unpunctuated and spells numbers, while live
  faster-whisper output is "Turn off librarian TV." (capitalized,
  trailing punctuation, digits, real mishearings).
- Out of scope (runtime, separate plan): `area_context.py:84` reports
  the satellite area as `target_area` for named devices in other rooms
  (31/78 cases), in production too.

## Slice 6: grader fixes (evals/scorer.py,
evals/tests/test_scorer_ha_equivalence.py, tests/test_realistic_v3.py,
evals/config/gates.yaml)

- Omitted-area relaxation applies only when the actual call has a
  `name`. An area-only expected call still requires the area.
- Clarification case passes when the reply contains "?" and names every
  candidate (existing full-name / exclusive-alias / unique-name-word
  rule); cue words are no longer required for that path.
- Refusal cues also cover "couldn't find", "could not find", "can't
  find", "cannot find", "don't see", "do not see", "there isn't",
  "there aren't", "don't know of", "isn't a", "no device", without
  matching affirmative confirmations.
- `CASES` in tests/test_realistic_v3.py keys are unique and the format
  tests use the original `20260908_ordinary_01` case again.
- Status gate 0.7 -> 0.6 (2/3).

## Slice 7: STT-shaped case text (evals/cases/realistic_v3.jsonl)

- Utterances written as faster-whisper writes them: sentence case,
  trailing "." or "?", digits for numbers. Expected calls unchanged.
- Replace 3-4 existing utterances in place with real mishearings taken
  from the live transcripts (task below) or the logged pool in
  `training/generators/stt_noise.py`, keeping the same expected call.

## Real usage pull (read-only, user-approved 2026-11-18)

- Read recent assist/conversation transcripts from Home Assistant
  read-only (no service calls, no config changes) and write them to a
  scratch file outside the repo. Compare against the promotion suite:
  which real commands are missing, which suite commands never occur.
