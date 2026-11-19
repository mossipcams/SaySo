# Plan: Fix 11 (grounding) and Fix 12 (tool annotations), adapted to Fix 13

Parent: `docs/PLAN_review_fixes.md` Fix 11 (line 298) and Fix 12 (line 325).
Depends on: `docs/PLAN_fix13_post_model_evals.md` (landed: `core/post_model.py`
calls `is_grounded` at line 274, and evals run the same core path through
`evals/runner.py`). Scope: plan only, no code changes. Line numbers are against
`b2b2c96`.

## Goal

Stop grounding from asking pointless clarifying questions on correct calls, and
keep genuine ambiguity clarifying. Fix 13 measured the cost on the stub adapter:
29 unique promotion cases (33 result rows, because 4 of them also appear in the
smoke suite) went from `valid_tool_call` to `missing_tool_call` with
`grounding_clarify`. Source: `evals/results/fix13_transitions.md`.

Constraint (parent decision): the rules live in the HA-free
`custom_components/sayso/core/grounding.py`, so production and evals run the same
code. `intent.async_match_targets` is not proposed inside core. Production still
builds `GroundingHome` from HA registries (`conversation.py:288-323`), so HA stays
the source of the entity list.

## Verified facts that change the review's plan

1. Fix 11's "reuse HA's matcher" is out of scope under the constraint above. The
   plan below stays with the English-token resolver and adds rules to it.
2. Fix 12's read-only list in the review is wrong for HA 2026.11.x.
   `components/intent/llm.py:57-73` (`INTENT_ANNOTATIONS`) marks only
   `INTENT_TIMER_STATUS` as `_READ_ONLY` (`:55`, `:72`). `HassGetState`,
   `HassGetCurrentDate`, `HassGetCurrentTime`, and `HassGetWeather` are not in that
   table. `components/todo/llm.py:59` sets `read_only=True` for `todo__get_items`.
   `GetLiveContext` and `GetDateTime` carry no annotation (`helpers/llm.py:216`
   is the default `ToolAnnotations()`, which is `read_only=False`).
3. Fix 12 changes no grounding outcome today. The four read-only tools already
   skip the name and area checks:
   - `intent__HassTimerStatus`: the `"Timer" in bare` check (`grounding.py:250`).
   - `homeassistant__GetLiveContext`, `llm__GetDateTime`: `_ALWAYS_GROUNDED`
     (`grounding.py:40`, `:250`).
   - `todo__get_items`: has no `name` or `area` argument, so `is_grounded` returns
     True (`grounding.py:272`).
   The read-only plumbing is therefore future-proofing for HA 2026.11 (see Slice
   3). It is in the plan because the parent asked for it. Parent to confirm.
4. `docs/HA_2026_11_TOOLS.md` already exists (tag 2026.11.1, no annotation
   coverage). Slice 5 appends to it. It does not create a new file.
5. `tests/test_realistic_v3.py::test_a_perfect_run_passes_every_case_and_every_gate`
   (line 211) fails today with `KeyError` at line 219. Its `ask()` looks up the
   prompt in `by_prompt` (built from first-pass messages only), so it cannot answer
   the follow-up call that the post-model path now makes after execution. It is
   excluded from CI (`.github/workflows/ci.yml:75`). Pending approval; this plan
   does not depend on it.

## Root causes (29 unique cases)

Classes are based on the rule that rejects each case today. Old rule cites:

- **A. Bare device noun, sole device of its domain, no room spoken (15).** Name
  match exists, but support is 0 (`grounding.py:82-93`; `:83` drops NOUN and
  GENERIC tokens from the utterance). Then `_area_noun_match` is False because
  no room is spoken (`:229-231`), and `is_grounded` returns False (`:260`).
- **C. Partial or fuzzy name (4).** Support is the average over label tokens
  (`:93`). One of three tokens matched, so 0.33 < 0.5 (`:258`). `television` is
  not in NOUN (`:30-38`), and `vacuum` is not in NOUN either.
- **R. Room spoken as the action area argument, device elsewhere (5).** The
  name path rejects (`:258`, `:229-231`). `drop_unspoken_area` keeps the area
  because it is spoken (`:147-150`).
- **B. Two or more same-domain devices, the satellite room holds exactly one
  (3).** `is_grounded` never credits `satellite_area` on the name path. The
  satellite is used only for area arguments (`_area_uttered`, `:122-127`).
- **N. "lamp" with many lights, satellite room holds two (2).** Same as B, plus
  the lamp noun needs a name match, not only a room.

| # | Case (suffix after `realistic_`) | Utterance | Model call (name, area) | Same-domain entities (household) | Satellite | Class | Rule |
|---|---|---|---|---|---|---|---|
| 1 | eval_20260908_media_01 | Resume the TV. | MediaUnpause: Living Room TV | H1 media: Living Room TV | Kitchen | A | G1 |
| 2 | eval_20260908_routines_vacuum_01 | Start the vacuum. | VacuumStart: Downstairs Robot Vacuum | H1 vacuum: Downstairs Robot Vacuum (Kitchen) | Kitchen | C | G1 |
| 3 | eval_20260908_routines_vacuum_02 | Send the vacuum back to the dock. | ReturnToBase: Kitchen Robot Vacuum | H2 vacuum: Kitchen Robot Vacuum (Kitchen) | Living Room | C | G1 |
| 4 | eval_20260908_ordinary_03 | Turn on the floor lamp. | TurnOn: Living Room Floor Lamp | H3 light: 8 lights, only Living Room Floor Lamp has "floor" | Bedroom | C | G3 |
| 5 | eval_20260908_media_03 | Set the volume on the television to 30%. | SetVolume: Living Room Television | H3 media: Living Room Television | Bedroom | C | G1 |
| 6 | eval_20260908_media_04 | Mute the TV. | Mute: Sofa TV | H4 media: Sofa TV (Living Room) | Kitchen | A | G1 |
| 7 | eval_20260908_media_05 | Turn the TV down a bit. | VolumeRelative: Family TV | H5 media: Family TV (Living Room) | Living Room | A | G1 |
| 8 | eval_20260908_media_06 | Pause the TV. | MediaPause: Living Room TV | H1 media: Living Room TV | Kitchen | A | G1 |
| 9 | eval_20260908_light_fan_settings_08 | Set the fan to 25%. | SetSpeed: Bedroom Tower Fan | H3 fan: Bedroom Tower Fan (Bedroom), Living Room Fan | Bedroom | B | G2 |
| 10 | eval_20260908_media_08 | Set the TV volume to 55. | SetVolume: Living Room Television | H3 media: Living Room Television | Bedroom | A | G1 |
| 11 | eval_20260908_ordinary_09 | Turn on the fan. | TurnOn: Bedroom Fan | H4 fan: Bedroom Fan (Bedroom) | Kitchen | A | G1 |
| 12 | eval_20260908_light_fan_settings_09 | Set the fan to 65%. | SetSpeed: Bedroom Fan | H4 fan: Bedroom Fan | Kitchen | A | G1 |
| 13 | eval_20260908_light_fan_settings_10 | Turn the fan down to 40%. | SetSpeed: Bedroom Standing Fan | H5 fan: Bedroom Standing Fan (Bedroom) | Living Room | A | G1 |
| 14 | eval_20260908_media_10 | Turn the TV up a little. | VolumeRelative: Family TV | H5 media: Family TV | Living Room | A | G1 |
| 15 | eval_20261118_ordinary_03 | Turn off librarian TV. | TurnOff: Lounge Television, area Living Room | H2 media: Bedroom TV (Bedroom), Lounge Television (Living Room) | Living Room | B | G2 blocked by modifier guard (residual, see Decision 1) |
| 16 | eval_20261118_ordinary_04 | Switch the fan on. | TurnOn: Bedroom Tower Fan, area Bedroom | H3 fan (as #9) | Bedroom | B | G2 |
| 17 | gen_1_light_fan_settings_014 | Turn the fan up to 100%. | SetSpeed: Bedroom Ceiling Fan | H1 fan: Bedroom Ceiling Fan (Bedroom) | Kitchen | A | G1 |
| 18 | gen_1_media_006 | Set the TV volume to 35%. | SetVolume: Living Room TV | H1 media | Kitchen | A | G1 |
| 19 | gen_1_media_007 | Turn the volume up on the TV. | VolumeRelative: Living Room TV | H1 media | Kitchen | A | G1 |
| 20 | gen_1_routines_vacuum_004 | Have the vacuum clean the hallway. | CleanArea: Downstairs Robot Vacuum, area Hallway | H1 vacuum (Kitchen) | Kitchen | R | G1 + room exemption |
| 21 | gen_1_routines_vacuum_005 | Vacuum the bathroom. | CleanArea: Downstairs Robot Vacuum, area Bathroom | H1 vacuum (Kitchen) | Kitchen | R | G1 + room exemption |
| 22 | gen_2_ordinary_021 | Switch off the lamp. | TurnOff: Sofa Reading Lamp, area Living Room | H2 light (Living Room): Living Room Ceiling Light, Sofa Reading Lamp [reading light, sofa lamp] | Living Room | N | G2 (lamp narrowing) |
| 23 | gen_2_routines_vacuum_003 | Vacuum the hallway. | CleanArea: Kitchen Robot Vacuum, area Hallway | H2 vacuum (Kitchen) | Living Room | R | G1 + room exemption |
| 24 | gen_2_routines_vacuum_004 | Have the vacuum clean the bedroom. | CleanArea: Kitchen Robot Vacuum, area Bedroom | H2 vacuum (Kitchen) | Living Room | R | G1 + room exemption |
| 25 | gen_3_routines_vacuum_003 | Dock the vacuum. | ReturnToBase: Roomba | H3 vacuum: Roomba (Kitchen) | Bedroom | A | G1 |
| 26 | gen_3_routines_vacuum_005 | Have the vacuum clean the hallway. | CleanArea: Roomba, area Hallway | H3 vacuum (Kitchen) | Bedroom | R | G1 + room exemption |
| 27 | gen_5_ordinary_004 | Switch on the fan. | TurnOn: Bedroom Standing Fan | H5 fan (Bedroom) | Living Room | A | G1 |
| 28 | gen_5_ordinary_009 | Switch off the TV. | TurnOff: Family TV, area Living Room | H5 media: Family TV | Living Room | A | G1 |
| 29 | gen_5_ordinary_016 | Switch on the lamp. | TurnOn: Living Room Corner Lamp, area Living Room | H5 light (Living Room): Living Room Corner Lamp [reading light], Living Room TV Backlight | Living Room | N | G2 (lamp narrowing) |

Rows 1, 2, 5, and 15 are also in the smoke suite (`fix13_transitions.md`, smoke
table). "Same-domain" means `_matches_dom` (`grounding.py:168-176`) for the call's
domain. Household fixtures are `evals/homes/realistic_eval_household_{1..5}.json`.
Vacuum and fan cases in H1 to H5 each have one vacuum or one fan unless the table
says otherwise. The lamp and light lists come from `_grounding_home` on each
fixture (`evals/runner.py:361-378`).

Counts: A 15, C 4, R 5, B 3, N 2 (total 29).

## Proposed rules (in `core/grounding.py`)

Keep the existing name check first (`:254-260`). Add the new branch only when
the name matches an entity E and the existing check returns False. Replace
`return _area_noun_match(first, utterance, home)` (`:260`) with
`if _area_noun_match(...): return True`, then `return _rule_grounded(...)`.
Leave the area-argument branch (`:269-272`) and `clarify` (`:275-308`) untouched.

New constant: `DOMAIN_WORDS = {**NOUN, "television": "media_player",
"vacuum": "vacuum", "vacuums": "vacuum"}`. Keep `NOUN` unchanged, because
`support` (`:83`) and `clarify` (`:278`) read it.

Pool: same-domain exposed entities, excluding the satellite device, as `_pool`
does (`:181-182`). Production and evals both get the same pool, because
`_grounding_home` filters by exposure (`conversation.py:296-297`).

- **G1, unique domain (fixes 23).** The utterance has domain word D with
  `_matches_dom(E, D)`, and the pool for D has exactly one entity (E).
- **G2, satellite-local (fixes 4).** E.area equals the satellite area, and E is the
  only D-pool entity in that area. If there are several, narrow to entities whose
  name tokens contain a spoken domain word (singular and plural folded: "lamp",
  "lamps"). Narrowing must leave only E. Guards below apply.
- **G3, partial name (fixes 1).** E matches D. Best `support(label)` for E is at
  least 1/3, and no other D-pool entity reaches 1/3 (`:93` scale). The 0.5
  threshold stays for the existing check.

Guards, applied to G1, G2, and G3:

- **Room guard.** If `mentioned_area` (`:96-108`) names a room that is not E.area,
  reject. Exception: when the room equals the call's own `area` argument (the room is
  the action target, not the device's location). This is why the five R rows pass.
- **Modifier guard.** For each domain word in the utterance, the adjacent tokens
  must each be a stop word, a digit, a domain word, or a token of E's label. The
  stop set is in the next block. This blocks "attic fan", "librarian lights", and
  "TV time".
- **Possessive and other-candidate guard (G2 only).** Reject if the utterance has an
  apostrophe, or if any utterance token matches a label token of another D-pool
  entity (for example "reading" in "reading light" for Sofa Reading Lamp).

Stop set (enumerate in code as a `frozenset`, no comments):
`the a an my our your this that its on off up down to in at of for and bit little
back is it can you please turn switch set dim mute pause unpause resume start send
dock have run clean brighten make lower raise open close lock unlock play stop go
return put change volume percent degrees some all any`

Sketch (names are suggestions):

```python
def _rule_grounded(entity, utterance, home, args, tool_domain) -> bool:
    if not _room_ok(entity, utterance, home, args) or _modifier_spoken(entity, utterance):
        return False
    for dom in _utterance_domains(utterance):
        if not _matches_dom(entity, dom):
            continue
        pool = _pool(utterance, home, dom, None)
        if len(pool) == 1:
            return True
        if G2 conditions:
            return True
        if G3 conditions:
            return True
    return False
```

No `_pool` change; reuse it with `room=None`. Do not change `_matches_dom`.

### Coverage of the 29 rows (prototype results, see Verification)

- G1 fixes 23: rows 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 23,
  24, 25, 26, 27, 28.
- G2 fixes 4: rows 9, 16, 22, 29.
- G3 fixes 1: row 4.
- Row 15 is blocked by the modifier guard ("librarian"). It is the one residual.

### Effect of the modifier guard

Prototype outcomes (scratch, not in repo; see Verification):

- Without the modifier guard, 29 of 29 rows flip. But 5 non-target calls become
  grounded: "Turn off the librarian lights" (Living Room Ceiling Light), "Turn on
  the attic fan" (Bedroom Ceiling Fan), "Turn on the attic fan" (Bedroom Standing
  Fan), "Turn off the patio lights" (Living Room Ceiling Light), and "Turn on TV
  time" (Family TV). Three of them are `unavailable` cases in the promotion suite, and one is a `routines_vacuum` case.
- With the modifier guard, 28 of 29 rows flip, and zero non-target calls ground.

## Decisions for the parent

1. **Row 15 (`eval_20261118_ordinary_03`, "Turn off librarian TV.").** This is the
   trade-off. Option A (plan default): keep the modifier guard. Row 15 stays a
   clarification, and the stub run is one case below the pre-fix baseline (smoke
   16/24 vs 17/24; promotion 441/530 vs 442/530). Option B: drop the modifier
   guard. All 29 rows pass (stub numbers return to the baseline exactly), but five
   non-target calls become grounded, three of them unavailable devices. The 33-row
   acceptance target is reachable only with Option B.
2. **Fix 12 is behavior-neutral today** (Verified fact 3). Keep Slice 3-4, or
   drop them and record the HA 2026.11 hook as future work. Parent to pick. The plan
   keeps them because the parent asked for them.
3. **Five R rows have expected calls whose area contradicts the entity's area**
   (for example "Downstairs Robot Vacuum" in Kitchen cleaning the Hallway). The room
   exemption grounds them by design. The grader accepts the expected call as-is.
   Not a grounding defect; noted so it is not mistaken for one later.

## Parent decisions (binding for the slices)

1. **Option A.** Keep the modifier guard. Row 15 stays a clarification, because
   the same rule without the guard grounds "the attic fan" and "the librarian
   lights" (unavailable devices) onto real entities, and AGENTS.md keeps
   fail-closed barriers. Stub targets: smoke 16/24, promotion 441/530.
2. **Keep Fix 12** (Slices 3-5). HA's annotations become the authority for
   read-only tools when present; the fallback set covers HA 2026.8.3.
3. **Narrow the room-guard exemption.** It applies only when the call's bare tool
   name contains `CleanArea` (the area is the cleaning target). For every other
   tool a spoken room that differs from E.area rejects. Slice 1 adds the negative
   test: "Turn on the kitchen fan." with `TurnOn` name `Bedroom Fan`, area
   `Kitchen`, one fan in the home (Bedroom) clarifies.
4. Slices 1+2 may ship as one dispatch, and Slices 3-5 as another, if the router
   allows the file count.

## Ambiguity and clarification cases that must stay unchanged

- All 14 `category: ambiguity` cases (`response_type: clarification`, `calls: []`):
  `realistic_eval_20260908_ambiguity_01..03`, `realistic_gen_1_ambiguity_001..003`,
  `realistic_gen_2_ambiguity_001`, `realistic_gen_3_ambiguity_001`,
  `realistic_gen_4_ambiguity_001..004`, `realistic_gen_5_ambiguity_001..002`.
  Under the stub they fail before and after (0.0 in `fix13_transitions.md`).
  The rules add no grounding for any same-domain candidate call on those
  utterances (checked, Verification). A new test in Slice 2 pins this.
- Clarify speech for grounding (`clarify`, `grounding.py:275-308`) is unchanged.

Existing tests the rules must not break (all pass with the prototype patched in):

- `tests/test_grounding.py`:
  - `test_is_grounded_true` (`:70-108`, 11 cases).
  - `test_not_grounded_and_clarify_speech` (`:114-156`, 7 cases). In particular
    "turn off the liver and lights" with Counter Switch-1234 (must clarify
    "Did you mean the Living room lights?"), and "unpause the tv" with
    `name: SaySo` (must clarify "Did you mean the TV?").
  - `test_satellite_never_offered` (`:164-169`) and `test_not_grounded_unnamed_entity`
    (`:172-183`).
  - Area-drop tests (`:211-291`), which pin `drop_unspoken_area`. Rules do not
    change that function.
- `tests/test_grounding_flow.py` (`:180-308`, 8 tests). In particular
  `test_voice_ungrounded_clarifies` ("cornet lamp" vs Shelf lamp, satellite Living
  Room with two lamps: G2 narrowing leaves two, so it still clarifies to "Corner
  lamp") and `test_two_option_clarification_named_choice_executes`.
- Grounding-adjacent: `tests/test_post_model_outcome.py`, `tests/test_area_arg_drop.py`,
  `tests/test_held_helpers.py`, `tests/test_match_failure_speech.py`,
  `evals/tests/test_post_model_eval.py`, `evals/tests/test_clarification_candidates.py`.

## Read-only plumbing (Fix 12)

Facts: see Verified facts 2 and 3.

- `core/tool_contract.py`: add `READ_ONLY_FALLBACK = frozenset({"intent__HassTimerStatus",
  "todo__get_items", "homeassistant__GetLiveContext", "llm__GetDateTime"})`. These are
  the names HA marks read-only (`intent/llm.py:72`, `todo/llm.py:59`) plus the two
  eval query tools (`evals/outcomes.py:15-18`). `GetLiveContext` and `GetDateTime`
  are not marked by HA 2026.11.x, so they are in the set because the eval already
  treats them as queries (`schema.py:124-135`).
- `core/post_model.py`: `TurnInput` gets `read_only: frozenset[str] = frozenset()`
  (after `confirmed`, `:72`). In `tool_loop`, split `validated` into read-only and
  other calls. Read-only calls skip the area drop (`:259-272`), grounding, and held
  clarify (`:273-287`). Everything else unchanged.
- `boundary.py`: `tool_is_read_only(tool) -> bool | None` returns
  `getattr(getattr(tool, "annotations", None), "read_only", None)`. On `None`
  (HA 2026.8.3, per `pyproject.toml:11`), the caller uses `READ_ONLY_FALLBACK`.
- `conversation.py` (`TurnInput` wiring near `:504-515`): for each `name, tool` in
  `tool_map`, `flag = tool_is_read_only(tool)`; the name is read-only when
  `flag` is True, or when `flag` is None and `name in READ_ONLY_FALLBACK`. Collect
  the names into a `frozenset`.
- `evals/runner.py` (`TurnInput` wiring near `:465-478`): `read_only = frozenset(names & READ_ONLY_FALLBACK)`.
  Evals never see HA annotations, so they use the fallback set only.

## Slices (each <= 4 files, each commits separately, nothing committed in this task)

Each slice ends with the CI gate:
`ruff check .`, `ruff format --check .`,
`pytest tests evals/tests --ignore=tests/test_realistic_v3.py -q`,
`pytest -q satellite/sayso`, and `python -m evals.generate_realistic --check`.
New code has no comments or docstrings (AGENTS.md).

**Slice 1. Core rules.** Files: `custom_components/sayso/core/grounding.py`,
`tests/test_grounding_rules.py` (new; the existing grounding test files stay
untouched). Tests: one parametrized case per row class A, B, C, N, R (all 28
rows, using the eval household's entity list rebuilt as a `GroundingHome`), the
row-15 residual (asserts clarify), and negative cases: the 14 ambiguity utterances
with same-domain candidate names; "Turn off the librarian lights", "Turn on the attic
fan", "Turn on TV time", "Turn on Jamie's bedside lamp" (possessive guard); and the
room guard (a spoken room that differs from the entity's area stays ungrounded).
Verify: `pytest tests/test_grounding.py tests/test_grounding_flow.py tests/test_grounding_rules.py -q`
and `pytest evals/tests/test_post_model_eval.py -q`.

**Slice 2. Eval-target regression.** Files: `evals/tests/test_grounding_targets.py`
(new). Loads the 29 unique case ids from `evals.cases.select_cases` and
`evals.runner._grounding_home(load_home(...))`, asserts `is_grounded` is True for
the 28 expected calls and False for row 15 (the one residual, which is pinned so
Option B would fail it loudly). Also asserts the 14 ambiguity cases ground no
same-domain candidate. Verify: `pytest evals/tests/test_grounding_targets.py -q`.

**Slice 3. Read-only in core.** Files: `custom_components/sayso/core/tool_contract.py`
(`READ_ONLY_FALLBACK`), `custom_components/sayso/core/post_model.py` (`TurnInput.read_only`,
skip rule), `tests/test_read_only_grounding.py` (new). Tests: a read-only call with a
name that would fail grounding still executes; a non-read-only call with the same
args clarifies; `TurnInput.read_only` defaults to empty. Verify:
`pytest tests/test_read_only_grounding.py tests/test_post_model_outcome.py tests/test_grounding_flow.py -q`.

**Slice 4. Read-only in production and evals.** Files:
`custom_components/sayso/boundary.py` (`tool_is_read_only`),
`custom_components/sayso/conversation.py` (wire `read_only`), `evals/runner.py` (wire
`read_only` from `READ_ONLY_FALLBACK`), `tests/test_read_only_grounding.py` (add
`tool_is_read_only` tests for `read_only=True`, `False`, and missing attribute; this is
our own new file from Slice 3). Verify:
`pytest tests/test_read_only_grounding.py tests/test_grounding_flow.py tests/test_schema_identity.py -q`
and `pytest evals/tests -q`.

**Slice 5. Docs.** Files: `docs/HA_2026_11_TOOLS.md` only. Append a section
"Tool annotations (Fix 12)": the `INTENT_ANNOTATIONS` table (`intent/llm.py:52-72`),
`todo/llm.py:59`, and the note that `GetLiveContext` and `GetDateTime` are not
annotated in 2026.11.x. Verify: `grep -n "Tool annotations" docs/HA_2026_11_TOOLS.md`.
Remove the fallback set when `pyproject.toml` raises the minimum to 2026.11.

**Slice 6. Stub re-run (verification only).** No repo file. Run
`PYTHONPATH=. python evals/results/fix13_stub_eval.py grounding` (label `grounding`
so `fix13_after_*.json` is not overwritten; both are gitignored, `.gitignore:46`).
Expected under Option A: smoke 16/24, promotion 441/530, `grounding_clarify`
flags 1 per suite (row 15). Under Option B: smoke 17/24, promotion 442/530,
`grounding_clarify` 0. The 28 (Option A) or 29 (Option B) rows return to
`valid_tool_call`; no other transitions. Any other change is a finding, not a gate change.

## Risks

- **Real-model accuracy is not measured here.** The stub returns the expected call;
  it checks the pipeline and the grounding rules, not the model. The llm VM run is
  out of scope.
- **Exposure parity.** The rules count only exposed entities, because
  `_grounding_home` filters by exposure (`conversation.py:296-297`). An unexposed
  TV does not block grounding. This matches production.
- **Narrowing depends on name tokens.** G2 narrowing uses entity names, not
  aliases. A device whose only satellite-room "lamp" is an alias will clarify.
  Accepted; it is the conservative direction.
- **Modifier stop set is a closed list.** A new verb or particle that is not in the
  list makes the guard reject, which clarifies (safe). A new device-name modifier that
  is in neither the label nor the stop set also clarifies (safe).
- **test_realistic_v3 (pending approval).** Its `ask()` (line 218-226) has no branch
  for follow-up messages, so the perfect-run test fails with `KeyError` at line 219.
  Fixing it means editing a file under `tests/`, which needs the user's approval
  (global CLAUDE.md test rules). Not in any slice.

## Verification done for this plan (scratch only, nothing in the repo)

Prototype rules and harness live in `/tmp/fix11/` (outside the repo). They were
patched in through a pytest plugin (`-p proto_plugin`) and through
`sayso_contract.post_model.is_grounded` for the stub run.

- Flip check over all 530 cases' expected calls: 28 of the 29 unique rows flip
  (row 15 does not); no other case flips; row 15 is the one residual.
- Negative check: 0 same-domain candidate calls newly grounded across the 14
  ambiguity utterances; 0 non-target same-domain candidate calls newly grounded
  across the 530 cases (non-ambiguity). Without the modifier guard, 5 non-target
  calls ground (see Decisions 1).
- Unit tests: `tests/test_grounding.py` + `tests/test_grounding_flow.py`: 36 passed
  with the prototype patched. Grounding-adjacent set (`tests/test_grounding*.py`,
  `tests/test_post_model_outcome.py`, `tests/test_area_arg_drop.py`,
  `tests/test_held_helpers.py`, `tests/test_match_failure_speech.py`,
  `evals/tests/test_post_model_eval.py`, `evals/tests/test_clarification_candidates.py`):
  83 passed, both with and without the prototype.
- Stub run through core with the prototype: smoke 16/24, promotion 441/530. Current
  HEAD: smoke 13/24, promotion 413/530. Pre-fix baseline: smoke 17/24, promotion
  442/530.
- Ran `tests/test_realistic_v3.py::test_a_perfect_run_passes_every_case_and_every_gate`
  on HEAD: fails with `KeyError` at line 219 (Verified fact 5).

## Pending approval (not in any slice)

- `tests/test_realistic_v3.py::test_a_perfect_run_passes_every_case_and_every_gate`:
  update its stub to answer tool-role follow-ups (`ask()` at lines 218-226). Needs
  the user's explicit approval because it edits `tests/`.
