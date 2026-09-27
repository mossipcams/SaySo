# v6 generator: fix the v5b eval failure modes (plan)

Date: 2026-09-26. Scope: generator-only changes under `training/generators` plus
tests under `training/tests`. Recipe shares, eval, runtime, and training scripts
are unchanged. No 40k generate, no training launch in this increment.

## Background: v5b promotion evidence (73/120, 47 failures)

Result dir on sayso-llm:
`/srv/llm/data/sayso/evals/results/20260925T174759Z-5ceb6471`.

Outcome mix: 37 unexpected_tool_call, 6 missing_tool_call, 3 invalid_arguments,
1 wrong_response_type.

Failed gates:

| gate | value | limit |
|---|---|---|
| false_action_rate | 0.6 | 0.2 |
| clarification_accuracy | 0.2 | 0.6 (target) |
| refusal/unavailable | 0.5 | — |
| aliases | 0.2 | — |
| exclusion | 0.4 | — |

Four failure modes drive the fixes below:

1. **Wrong-entity selection (aliases 7/8 failures):** model picks the near-miss
   sibling. Examples: 'prep lights' → Kitchen Ceiling Light (expected Kitchen
   Counter Lights); 'sofa lamp' → Sofa Reading Lamp (expected Sofa TV); 'my
   bedside light' → GetLiveContext on Bedroom Bedside Lamp (expected Morgan's
   Bedside Lamp). Corpus aliases are ~98% word-overlap; eval aliases are
   nicknames; distractors are not near-miss siblings.
2. **Act-instead-of-clarify (ambiguity 7/8, action_on_ambiguity x7):**
   multi-candidate settings requests ('Dim the reading light in the living room
   to 30 percent', 'set the living room reading light to warm white') get a
   direct `HassLightSet` call instead of a clarification. Generator
   clarify/follow-up rows only draw on/off/status operations
   (`planning.py`: `family == "clarify"` draws `turn_on`/`turn_off`/`query_state`,
   `family == "follow_up"` draws `turn_on`/`turn_off`).
3. **Tool substitution on withheld catalogs (unavailable 4/5,
   tool_call_on_unavailable x5):** when the requested tool is not offered, the
   model calls the closest available tool: `HassSetVolume` on a light,
   `HassLightSet` for fan speed and for 'Main Thermostat at 69 degrees',
   `HassTurnOn` with device_class door on a gate. v5's catalog-shape fix removed
   the refusal shortcut but did not stop substitution.
4. **Excluded entity affected (exclusion 6/10, excluded_entity_affected x2 +
   wrong target sets):** 'switch off the kitchen lights, but leave X alone' → X
   gets switched or the target set is wrong. v5 scoped 60% of exclusion rows to
   an area (`scenarios/core.py` `build_scenario`, `exclusion_scope`), but the
   excluded entity is not consistently a same-area, similarly-named sibling of
   the target set.

Plus dataset hygiene from the v5b training log: 2,980 invalid rows, 4.3%
duplicates (limit 1%), 0.42% conflicting examples (limit 0.1%), missing pinned
core tools. The corpus failed its own validation gates.

## Review adjustments (2026-09-26)

- **Baseline:** `PYTHONPATH=training python3 -m pytest training/tests
  training/scripts -q --ignore training/tests/test_lfm_python_parse.py` = 330
  passed, 7 pre-existing failures (`training/tests/test_llamafactory_view.py`
  etc., environment-related); `test_lfm_python_parse.py` fails collection (no
  `homeassistant` module).
- **Step 0:** needs the sayso-llm host and the v5b model, so it can't run from
  the dev worktree. It now gates the 40k run and the training launch, not these
  generator changes (which do nothing until a run).
- **Execution order:** Fix 5 (hygiene gate) first, then Fixes 1, 2, 3, 4; one
  bounded change each, tests in `training/tests/test_v6_eval_fixes.py` (new
  file; never modify other test files).
- **Fix 1:** aliases draw lights/fans/switches/media_players ×
  turn_on/turn_off; the nickname pool in `homes.py` is keyed by role and never
  uses the eval nicknames ("prep lights", "sofa lamp", "my bedside light");
  alias targets prefer entities with a usable alias, and a near-miss same-area
  sibling is injected when missing; the audit counts `nickname_alias_rows`.
- **Fix 2:** clarify/follow_up draw 50% from `AMBIGUOUS_SETTINGS` (lights
  `set_brightness`/`set_color`/`set_color_temperature`, media_players
  `volume_set`, fans `set_speed`, climate `set_temperature`). Follow-up gold
  reuses the value spoken in the ambiguous first turn. Note:
  `planning.SETTINGS_OPERATIONS` lists "set_volume", but the registry op is
  "volume_set" (a pre-existing mismatch, out of scope).
- **Fix 3:** no new contrast family: settings/ordinary rows already call these
  tools with them offered, and `rendering.DECOY_WITHHOLD_RATE=0.3` already
  decouples catalog size. The real gap is that size-16 synthetic homes have no
  `media_player`, so `HassSetVolume` is never offered as a temptation;
  unavailable rows now inject a light and a `media_player`. Unavailable draws
  50% from substitution-prone ops. The audit reports `by_withheld_tool`. Test
  the invariant that non-refusal rows always offer the requested tool.
- **Fix 4:** (a) the excluded pick is biased to the highest name-token overlap
  with in-area siblings; (b) keep the named-list branch as a deliberate
  contrast, add `exclusion_scope` to row metadata, and the audit reports
  `exclusion_same_area` vs `exclusion_named_list`; (c) the phrasing already
  matches the eval wording ("but leave X alone") — deferred to the ablation.
- **Fix 5:** the gate lives in `pipeline.run_generation` (`--dry-run` does not
  call `run_build`) and runs `scripts/preflight.inspect_dataset` over the rows
  with limits from `training/configs/preflight.yaml`; `stats["hygiene"]`
  reports the metrics. The dedup source is `DuplicateTracker` allowing
  `near_duplicate_limit` exact (utterance, home) repeats; exact repeats are now
  rejected after the first.

## Fix 1: near-miss sibling alias rows

Evidence: failure mode 1.

Changes:

- `training/generators/planning.py`: widen the `aliases` family draw (the
  `family == "aliases"` branch of the capability/operation picker, currently
  `rng.choice(["lights", "fans", "switches"]), "turn_on", 1`) to cover the
  shapes in the v5b alias failures: `media_players` targets (Sofa TV, twice)
  and `turn_off` operations. New nickname rows must include those shapes, not
  just lights × turn_on.
- `training/generators/homes.py`: alias generation in `generate_home` adds a
  nickname vocabulary pool (owner nicknames, short informal names) alongside the
  existing word-overlap aliases (`{area} {noun}`, light/lamp synonym swap,
  `_collision_aliases`); a share of alias rows uses nickname aliases. The pool
  is keyed by role and never uses the eval nicknames ("prep lights", "sofa
  lamp", "my bedside light").
- Alias targets prefer entities that already have a usable alias; when the
  chosen target has none, a near-miss same-area sibling is injected so the row
  is well-formed. The audit counts `nickname_alias_rows`.
- Alias/distractor selection (`training/generators/scenarios/discrimination.py`
  — `sibling_entities`, `pick_discriminating_description` — and the `aliases`
  family path in `training/generators/scenarios/core.py`, robustness
  `alias_distractor`): when a row's target has same-domain siblings in the same
  area, prefer those as the competing candidates so the model must resolve the
  alias against near-miss names rather than unrelated entities.
- Gold stays the canonical entity name; the utterance may use the nickname.

Files: `training/generators/planning.py`, `training/generators/homes.py`,
`training/generators/utterances.py`,
`training/generators/scenarios/discrimination.py`,
`training/generators/scenarios/core.py`, tests in `training/tests` (new failing
checks first: nickname alias row resolves to canonical name; near-miss sibling
present in context).

## Fix 2: settings-request ambiguity

Evidence: failure mode 2.

Changes:

- Clarify and follow-up scenario selection draws the requested operation from
  settings ops (`HassLightSet` brightness/color/temperature, `HassSetVolume`,
  climate set) when the request matches multiple candidates, not only
  on/off/status. The operation draw lives in
  `training/generators/planning.py` (the `family == "clarify"` /
  `family == "follow_up"` branches); the ambiguity graph shaping lives in
  `training/generators/scenarios/core.py` (`configure_family_scenario`), and the
  follow-up turn assembly in `training/generators/row_generation.py`.
  Concretely: 50% of clarify/follow_up rows draw from `AMBIGUOUS_SETTINGS`
  (lights `set_brightness`/`set_color`/`set_color_temperature`, media_players
  `volume_set`, fans `set_speed`, climate `set_temperature`). Note:
  `planning.SETTINGS_OPERATIONS` lists "set_volume", but the registry op is
  "volume_set" — a pre-existing mismatch, out of scope for this fix.
- Gold remains a clarification turn (no tool call); the clarification names the
  real candidates. Follow-up gold reuses the value spoken in the ambiguous
  first turn.
- Keep the existing `clarify_target_named` validation
  (`training/generators/validation.py`): an exact multi-word name match must not
  clarify.

Files: `training/generators/planning.py`,
`training/generators/scenarios/core.py`, `training/generators/row_generation.py`,
`training/generators/utterances.py`, `training/generators/gold.py` if needed,
tests.

## Fix 3: withheld-tool refusal contrast

Evidence: failure mode 3.

Changes:

- Unavailable-family rows: withhold exactly the tool of the requested operation
  from the offered catalog (`training/generators/rendering.py`
  `_offered_catalog`, driven by `removed_tools` set in
  `training/generators/scenarios/core.py` `configure_family_scenario`) with gold
  refusal; ensure the sibling tools the model substituted to
  (`HassSetVolume`, `HassLightSet`, `HassTurnOn`) are offered in those rows so
  refusal is the only correct move.
- No new contrast family is needed: settings/ordinary rows already call these
  tools with them offered, and `rendering.DECOY_WITHHOLD_RATE=0.3` already
  decouples catalog size, so tool identity — not catalog size — is what flips
  the decision (the v5 catalog-shape lesson is already covered).
- The real gap: size-16 synthetic homes have no `media_player`, so
  `HassSetVolume` is never offered as a temptation. Unavailable rows now inject
  a light and a `media_player` so the substituted-to tool is present, and
  unavailable draws 50% from substitution-prone ops.
- Audit (`training/generators/audit.py` `audit_rows`) reports the withheld-tool
  distribution (`by_withheld_tool`) so substitution-prone pairs (light settings
  vs volume, climate vs light set, cover vs turn-on) are covered.
- Test the invariant: non-refusal rows always offer the requested tool.
- Invariant this fix depends on: `rendering.py` withholding must never withhold
  the requested operation's tool on non-unavailable rows (v5 behavior —
  `removed_tools` is set only by the unavailable path in
  `scenarios/core.py`, with decoy removals otherwise). The refusal contrast
  only works while that invariant holds.

Files: `training/generators/rendering.py`,
`training/generators/scenarios/unavailable.py`,
`training/generators/scenarios/core.py`, `training/generators/audit.py`, tests.

## Fix 4: same-area exclusion with named sibling

Evidence: failure mode 4.

Scope note (re-scoped): the 60% same-area branch in
`training/generators/scenarios/core.py` `build_scenario`
(`robustness == "exclusion"`, `len(in_area) >= 3 and rng.random() < 0.6`)
already guarantees the excluded entity is same-area, same-capability, a member
of the naive target set, and named exactly in the utterance (the audit enforces
the 'leave X' wording and that gold calls never touch `excluded_names`). The
gap is narrower than it first looks:

Changes:

- (a) Prefer similar-name siblings: the branch currently picks the excluded
  entity uniformly (`kept = rng.choice(in_area)`); bias the pick to the
  highest name-token overlap with in-area siblings so the model must resolve
  the named sibling against the area-wide target set.
- (b) Keep the named-list branch (`target_entities = cap_entities[:2]`,
  excluded = `cap_entities[2]`) deliberately as a contrast; add
  `exclusion_scope` to row metadata and have the audit report
  `exclusion_same_area` vs `exclusion_named_list`.
- (c) The phrasing already matches the eval wording ("but leave X alone"), so
  phrasing-level work in `training/generators/utterances.py` is deferred to the
  Step 3 ablation: the ablation must measure before/after scenario surgery
  alone.

Files: `training/generators/scenarios/core.py`,
`training/generators/utterances.py`, tests (excluded entity must be in the naive
set; gold calls must not touch it).

## Fix 5: dataset hygiene gates (fail-closed)

Evidence: v5b corpus validation failures (2,980 invalid rows, 4.3% dup, 0.42%
conflict, missing pinned core tools).

Changes:

- The gate lives in `training/generators/pipeline.py` `run_generation` (not
  `run_build` — `--dry-run` does not call `run_build`), alongside the existing
  `enforce_rate_gate` calls. It runs `scripts/preflight.inspect_dataset` over
  the generated rows with limits from `training/configs/preflight.yaml`
  (`max_duplicate_rate: 0.01`, `max_conflict_rate: 0.001`, schema validation
  against the pinned SaySo tool schemas, and the eval-overlap check) and fails
  the run (non-zero exit, no manifest) on any error: duplicate rate > 1%,
  conflicting-example rate > 0.1%, any invalid row, or a pinned core tool with
  zero positive rows.
- `training/generators/deduplication.py` (`DuplicateTracker`): the dup source is
  `DuplicateTracker` allowing `near_duplicate_limit` exact (utterance, home)
  repeats; exact repeats are now rejected after the first so the 1% limit is
  achievable at 40k.
- `training/generators/audit.py` (`audit_rows`) / run summary: `stats["hygiene"]`
  reports all four hygiene metrics plus the family × robustness counts (row
  metadata carries both `family` and `category`/robustness) in the run summary.

Files: `training/generators/pipeline.py`, `training/generators/manifest.py`,
`training/generators/deduplication.py`, `training/generators/audit.py`,
`scripts/preflight.py` (read-only reference / pipeline integration point),
tests.

## Verification

- `PYTHONPATH=training python3 -m pytest training/tests training/scripts -q
  --ignore training/tests/test_lfm_python_parse.py`
  (baseline per the 2026-09-26 review: 330 passed, 7 pre-existing
  environment-related failures in `test_llamafactory_view.py` etc.; new tests
  must fail before their fix)
- `cd training && PYTHONPATH=. python3 -m generators.cli --config configs/generation/smoke.yaml --dry-run`
  — the check is against the smoke run's audit output (the manifest
  `stats.quality_audit` printed by the dry run), which reports family ×
  robustness counts; require at least one row per new shape: aliases ×
  alias_distractor (nickname), clarify/follow_up × ambiguity (settings),
  unavailable × unavailable (withheld-tool refusal), exclusion × exclusion
  (same-area)
- Audit output on the smoke run reports the four hygiene metrics
- `git diff -- evals/` is empty; no eval case IDs or utterances copied into
  training labels
- No training launch, no 40k generate in this increment

### Results (2026-09-26)

- Implemented in order: Fix 5 (5, 5a, 5b, 5c), Fix 1 (1a, 1b), Fix 2, Fix 3,
  Fix 4. New tests: `training/tests/test_v6_eval_fixes.py`.
- Full suite: 354 passed, 7 failed — exactly the pre-existing baseline
  failures (`test_llamafactory_view.py` x5, `test_defect_v4_corpus.py`
  token-count x2); `test_lfm_python_parse.py` still needs `homeassistant`.
- Smoke dry-run: passes. Hygiene: 0 invalid rows, 0 invalid calls,
  duplicate_rate 0.0, conflict_rate 0.0, 0 contaminated. Audit:
  alias_rows 5, nickname_alias_rows 2, exclusion_rows 8 (same_area 4,
  named_list 4), by_withheld_tool {HassCancelAllTimers 4,
  HassClimateSetTemperature 1, HassFanSetSpeed 1, HassSetVolume 1}.
- Over 5 smoke seeds: 20 clarify + 11 follow_up rows with a settings
  operation; withheld tools spread across HassLightSet 9, HassFanSetSpeed 7,
  HassSetVolume 4, HassClimateSetTemperature 3, and assorted media/timer tools
  (the single-seed timer skew was noise).
- `git diff -- evals/` is empty. No 40k generate, no training launch.
- Additional changes beyond the original fix list, found by the new gate:
  (1) the four core tools (GetDateTime, GetLiveContext, HassTurnOn, HassTurnOff)
  are never withheld, neither as decoys (`rendering._decoy_removals`) nor by
  the unavailable family (`planning._WITHHOLDABLE_OPERATIONS`), since
  production HA always offers them; the unsupported family keeps those ops.
  (2) Dedup keys homes on `home_id` (matching preflight), because per-row
  entity injection let real-home exact repeats escape. (3) Real-home retries
  redraw with a retry seed salt instead of shifting `attempt`, keeping the
  `attempt % 125` exclusion cadence. (4) The hygiene gate requires GetDateTime
  only when `get_datetime_positive_min > 0`.
- Known follow-ups: the capped-real-home exclusion patch in `pipeline.py`
  duplicates row_generation's exclusion-trigger condition (tidy before the 40k
  run); the aliases draw uses a side rng stream (and discards one main-stream
  draw) to keep seed-pinned tests stable; synthetic `home_id` can collide
  across retry attempts (harmless: costs an attempt, same as preflight's key);
  Fix 4(c) phrasing deferred to the ablation; Step 0 still gates the 40k run.

## Out of scope

- Recipe allocation shares, eval cases, runtime/integration, training scripts
- Constrained decoding, prompt compaction, model size (v6 plan Step 5, only if
  ablations point there)
- Real-transcript junk realism and STT error-model refit (separate v6 Step 4
  candidates, ablated independently)

## Step 0: prerequisite — Step 1 in-distribution check (gate)

Per `docs/PLAN_V6_DATASET_ARCHITECTURE.md` Step 1, the 300-row in-distribution
check on the current v5b model (300 held-out v5 rows, same generator and
recipe, new seed, no prompt overlap with train, scored on the first supervised
turn) decides whether data work is the right lever at all. High
in-distribution with low eval → the gap is data distribution; proceed to the
fixes. Low in-distribution too → the model isn't learning the task; prompt
length, model size, or training settings come before data, and the 40k run
stops.

Review adjustment (2026-09-26): Step 0 needs the sayso-llm host and the v5b
model, so it can't run from the dev worktree. It therefore gates the 40k run
and the training launch, not the generator changes in this plan (which do
nothing until a run).

## Sequencing

Review-adjusted (2026-09-26): one bounded change per fix, in the order Fix 5
(hygiene gate) first, then Fixes 1, 2, 3, 4; each with its tests first in
`training/tests/test_v6_eval_fixes.py` (new file; never modify other test
files). After all five, regenerate the smoke set (re-verify against the audit
output above) before any 40k run. Step 0 gates the 40k run and training
launch, not these generator changes. Ablations per
`docs/PLAN_V6_DATASET_ARCHITECTURE.md` Step 3.
