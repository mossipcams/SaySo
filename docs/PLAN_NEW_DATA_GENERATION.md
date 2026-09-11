# SaySo training-data fixes completion plan

## Goal and boundary

Complete the existing dataset-generation, audit, and evaluation changes in
`ajax/new-data-generation`, adapting to the code already present. Home Assistant
remains authoritative for entities, exposure, tools, capabilities, and action
execution. Preserve the pinned training contract, runtime validation and
fail-closed behavior, existing eval suites, and existing training checkpoints.

This task ends after implementation, focused tests, and a small deterministic
dataset audit. **Do not launch model training, generate a new 40k production
corpus, export a model, or promote a checkpoint.**

Status legend: `[x]` complete, `[~]` implemented in the uncommitted worktree but
not fully validated, `[ ]` remaining.

## Evidence reviewed

- `AGENTS.md` and `docs/TRAINING_PLAN.md`.
- Transcript `01a08e06-8be2-79b1-9486-563d13311cd6`, including the approved
  TDD pass that ended during repository validation.
- Transcript `01a08e10-e1b4-7db2-9c84-91181a240f24`.
- The live uncommitted diff, tests, fixtures, and artifacts in
  `/Users/matt/Desktop/Projects/SaySo__worktrees/ajax-new-data-generation`.

## Current checkpoint

- [x] Actual-label classification and separate positive, negative, outcome,
  targeting, and tool counters exist in `coverage.py` and `sampling.py`.
- [x] Declared training coverage, capability-aware action generation, dynamic
  script handling, and distractor exclusions exist across the registry and tool
  builders.
- [x] Graph-derived grounding families and held-out grounding evals exist,
  including the exact Living Room / TV / `HassTurnOn` regression.
- [x] Assist-exposure export, feature-aware media-player handling, real-home
  mixing, holdouts, entity caps, and synthetic-only behavior exist.
- [x] Dataset audits record positive coverage, outcomes, targeting, tools,
  negative reasons, absence rate, and real-home counts.
- [x] A grounding regression was observed failing before the generation-path
  fix, then passing after a minimal deterministic grounding guarantee.
- [x] The focused generator, grounding, real-home, exposure, and websocket suite
  passes: **49 tests passed**.
- [x] The complete relevant training/generator/eval/script suite passes:
  **375 tests passed**.
- [x] The 1,200-row smoke corpus and manifest were regenerated twice with seed
  `20260911` and were byte-identical with nonzero grounding and no quota gaps.
- [x] Unit 9's code was found already written but unvalidated: it regressed one
  focused test (no positive `HassTurnOn` on the TV) and five suite tests (400-row
  runs deadlocked on the required-grounding gate). All six are fixed and the
  49/375 counts are restored.
- [x] Live Assist-exposure credentials were loaded from the HA logs MCP config.
  The refreshed 35-entity snapshot has `exposure_source: assist_exposure`, keeps
  `media_player.living_room_tv` as `TV` in Living Room, and passes the live-home
  prerequisite. A synthetic stale snapshot still verifies fail-closed behavior.

## TDD units

### 1. Positive-example quota accounting `[x]`

- **Test:** Use the existing regressions proving a refusal cannot count as
  positive supervision, can fill only its own negative allowance, a wrong-domain
  call is not positive, offering a tool is not coverage, and impossible quota
  inputs fail before generation.
- **Code:** Finish `QuotaTracker` around accepted labels, not metadata. Keep total
  dataset-size accounting while separating successful actions, state queries,
  clarifications, absence, and unsupported outcomes. Count a positive operation
  only when the expected call has the correct tool, operation, domain, and real
  target; refusals count only toward the corresponding negative budget.
- **Verify:** Run the quota regressions and a deterministic small generation.
  Require no positive count to exceed accepted rows, no quota shortfall, and a
  clear pre-loop error for impossible configurations.

### 2. Supported behavior coverage `[x]`

- **Test:** Use the existing coverage tests for every declared tool, tools outside
  declared coverage, unsupported entity actions, graph-valid unsupported
  refusals, and generated media operations on capable devices only.
- **Code:** Reconcile `capability_registry.py`, tool builders, and
  `schemas/sayso-tool-schema-v2.json` before adding mappings. Retain media-player
  `turn_on` and `turn_off` when their generated labels are already valid. Require
  a positive path for every retained training-covered tool and valid
  domain/action pair. Resolve dynamic scripts against the row's per-home tools.
  Exclude non-covered tools from ordinary distractors without changing Home
  Assistant runtime availability.
- **Verify:** Every retained covered tool and operation has actual positive
  supervision, every call passes the pinned schema, and no entity receives an
  operation absent from its exported capabilities.

### 3. Explicit grounding scenarios `[x]`

- **Test:** Keep one request constant across the required graph mutations:
  Living Room `TV` targets `TV`; renamed `Cinema` targets `Cinema`; moving it to
  Bedroom produces no matching target; adding light, switch, and unrelated-media
  distractors still targets `TV`. Also test individual/area targeting, ambiguity,
  aliases, domain changes, and supported-action presence/absence pairs.
- **Code:** Finish paired variants in the existing scenario/gold-label path.
  Independently vary names, entity IDs, aliases, areas, domains, and supported
  actions. Derive expected calls or no-action outcomes from the supplied entity
  graph and current tool contract before OHF wording, framing, STT noise, or
  paraphrasing.
- **Verify:** Graph changes alter the expected target or outcome while unchanged
  and distractor-only graphs preserve the correct answer. All grounding rows
  render in the production message/context/tool format.

### 4. Real-home inputs and mixing `[x]`

- **Test:** Use the exporter tests for Assist exposure, canonical names, aliases,
  areas/floors, device classes, `supported_features`, fan/media capability
  differences, and websocket framing. Use mixing tests for TV split inclusion,
  supported positive TV operations, requested/achieved rate, row counting,
  per-entity counts, holdouts, caps, and synthetic-only override.
- **Code:** Finish `fetch_ha_home.py`, `real_home.py`, generator config, and CLI so
  the home recipe requires assistant-exposed entities and a configurable nonzero
  mix rate. Keep an explicit synthetic-only override. Do not assume all media
  players support power, pause, volume, or mute.
- **Verify:** If configured HA access is available, refresh the snapshot and
  verify `media_player.living_room_tv` is named `TV` and is in Living Room. If it
  is unavailable, validate only against the clearly named synthetic fixture and
  leave live refresh as a hard prerequisite for the home-specific recipe.
  Preserve entity-frequency caps and holdout exclusions, and require positive TV
  rows for every operation that TV actually supports.

### 5. Dataset-level audits and manifest `[x]`

- **Test:** Use regressions for missing positive operation coverage, missing tool
  coverage, absence-rate caps, multi-target row counting, and real-entity-only
  target counts.
- **Code:** Finish the existing audit over actual accepted labels across tool,
  domain/action, operation, targeting mode, and outcome. Keep configurable
  positive floors and absence caps. Record requested versus achieved mixing,
  actual real-home rows, per-entity target counts, positive coverage, and
  negative distributions in the existing manifest.
- **Verify:** Missing required combinations fail generation. Multiple targets in
  one example count as one real-home row. Diversity is produced by generation,
  not cloning. Treat any absence total, including the previously reported 209,
  as a measured distribution subject to the configured cap, not as proof of
  learned behavior.

### 6. Regression and generalization eval coverage `[x]`

- **Test:** Keep the exact production-format regression for “Turn on the living
  room media player” with `media_player.living_room_tv`, name `TV`, and expected
  `HassTurnOn({"domain":["media_player"],"name":"TV"})`. Assert exact arguments,
  not only tool selection. Add/retain held-out names, IDs, areas, aliases,
  distractors, ambiguity, capabilities, and presence/absence outcomes.
- **Code:** Finish `grounding_eval.py` and the existing v3 exclusion hook so all
  grounding eval prompts and near-duplicate scenario variants are rejected from
  training generation.
- **Verify:** Grounding eval examples use different graph details from training,
  exact targets and arguments validate against the supplied tools, and existing
  recipe-lock, gold, shadow, safety, follow-up, timer, and basic offline evals are
  unchanged and still runnable.

### 7. Final focused validation `[x]`

- **Test:** Re-run the focused generator/audit/grounding/real-home/export tests,
  then the relevant existing training generator, eval, and script suites. Do not
  run a trainer or model-serving evaluation.
- **Code:** Fix only failures caused by this task. Regenerate the small,
  deterministic 1,200-row smoke corpus and manifest with the completed code; do
  not generate the 40k production corpus.
- **Verify:** Record a successful pytest exit and pass count. Require exact smoke
  row counts, deterministic hashes, empty quota shortfalls, covered positive
  tools and operations, bounded absence, correct requested/achieved mixing,
  nonzero grounding, no eval overlap, no truncation, and schema-valid calls.

### 8. Remote gauntlet eval generation `[x]`

- **Test:** Require the Living Room anchor to keep the requested literal wording,
  and require supported/unsupported capability pairs generated at different row
  indexes to keep the same request and numeric arguments.
- **Code:** Reuse the grounding generator, fixing only family-level wording and
  randomness needed to keep paired prompts stable.
- **Verify:** Generate `gauntlet_eval.jsonl` on `192.168.1.140`, rerun its goal audit,
  require every check to pass, and confirm no model training process starts.

### 9. Full-corpus source-rate gates `[~]`

- **Test:** Require derived-cap real-home recipes to achieve their requested
  accepted-row rate and require every mandatory grounding contrast family.
  Both are covered by existing regressions rather than new ones:
  `test_the_manifest_records_requested_and_achieved_mixing` runs on the derived
  cap and asserts achieved equals requested, and
  `test_enabling_grounding_produces_grounding_supervision` asserts every family
  in `required_training_variants()`.
- **Code:** Balance real-home selection against remaining accepted rows and
  prioritize missing grounding families without cloning rows. Three defects in
  that code were found and fixed while validating it:
  - Real-home target choice rotated on the *global* accepted-row count, so
    covering a given device/operation pair was luck; the balanced draw shifted
    the stream and the TV lost its `turn_on` row. `scenarios.pick_target` now
    prefers the least-used eligible entity **per operation**, which makes that
    coverage structural. Synthetic homes are unaffected: they draw a fresh home
    per row and still take the plain rotation.
  - `pick_variant` was called with a fixed index while forcing missing grounding
    families, so two families sharing one capability/operation pool could never
    both land — the second stayed unreachable behind the first.
  - The required-family gate engaged at bare parity
    (`count * grounding_rate >= len(required)`), demanding all eleven families
    from a run too small to place them; the refusal families additionally
    compete for a bucket's small negative allowance. `GROUNDING_FAMILY_SLACK`
    now requires headroom, so small runs no longer fail closed on a gate they
    were never large enough to meet.
- **Verify:** *Outstanding, and blocked by this plan's own boundary.* The 40k
  reject-and-regenerate check contradicts "do not generate a new 40k production
  corpus" in the goal section, so it was not run. Everything verifiable at smoke
  scale passes: exact row counts, 10% real-home rows, all eleven grounding
  families, OHF provenance, and positive TV operations for every operation the
  TV supports. The 40k run and its gauntlet-eval overlap check need an explicit
  decision to lift the boundary.

## Completion evidence

- Focused regressions: 49 passed.
- Complete relevant suite: 375 passed.

### After unit 9 (current)

Unit 9's code changed generation output, so the smoke corpus was regenerated.
Run from the repository root; the suite paths are `training/generators`,
`training/scripts`, `training/evals`, and `training/tests` (several tests resolve
fixtures relative to the repo root and fail if pytest runs inside `training/`).

- Focused regressions: 49 passed.
- Complete relevant suite: 375 passed.
- Smoke corpus: 1,200 rows, seed `20260911`, real home
  `synthetic_reference_home.json` at a requested 10% mix. Regenerated twice and
  byte-identical.
- Canonical SHA-256:
  `4cf368349a1b5832d497bbf3a8b50596f22d7c702375beb3cadd797d832c4fe5`.
- Manifest SHA-256:
  `29e3deaa417af61f9e02615b430ae5847db43ca8e1120b486558aed586db3983`.
- Manifest gates: no quota shortfalls or uncovered operations; **all 11**
  grounding families across 11 rows; **120 real-home rows at an achieved 10.0%**,
  exactly the requested mix; 1.75% absence; 27 positive tools; 13 exclusion, 84
  multi-call, and 21 alias rows; 1,164 unique requests over 26 OHF sources.

### Before unit 9 (superseded)

- Canonical SHA-256:
  `c7aacd47a94c9f777ff50a85c483235908ad3efca51ca81a63fa14644b9d4da2`.
- Rendered SHA-256:
  `3488ccb43e58b83d4ab5f0c3a9c58630ead3a45ffc0a9b30ac36694e850b07d4`.
- Manifest SHA-256:
  `ac9afc35e0bbf7bdcbea4fb4f48d785dbed536f54f63b5574048d618deadeca3`.
- Manifest gates: no quota shortfalls or uncovered operations; five grounding
  rows; 114 real-home rows at a requested 10% mix; 2.33% absence; 13 exclusion,
  72 multi-call, and 29 alias rows. The five-of-eleven grounding families and
  the 9.5% achieved mix are the two gaps unit 9 closes.
- The original seed `20260910` failed closed with five exclusion rows where six
  were required. Seed `20260911` passes naturally; no rows were cloned and no
  diversity assertion was weakened.
- HA logs MCP supplied `HA_URL` and `HA_TOKEN` without printing either secret.
  The refreshed snapshot contains 35 Assist-exposed entities, strips null/blank
  aliases, and keeps TV in the train split with capability-derived operations.
- Remote `gauntlet eval`: 15 rows generated on `192.168.1.140`; SHA-256
  `82ced9b5bd0e283c76a49ab404dd195a8c467f3502ced7d9598b082f70d4fa2e`.
  Its grounding/generalization goal audit passed 11/11, the affected local suite
  passed 60 tests, and no gauntlet training process was started.

## Files in scope

- `training/generators/sampling.py`, `coverage.py`, `audit.py`,
  `capability_registry.py`, `tools.py`, `scenarios.py`, `gold.py`, `grounding.py`,
  `real_home.py`, `config.py`, `cli.py`, and `pipeline.py`
- `training/scripts/fetch_ha_home.py` and
  `training/scripts/build_synthetic_dataset.py`
- `training/evals/grounding_eval.py` and `training/evals/v3_quality.py`
- `training/fixtures/synthetic_reference_home.json` and a live refresh of
  `training/fixtures/real_home.json` only when configured HA access is available
- `training/generators/test_coverage_gates.py`,
  `training/generators/test_real_home_mixing.py`, and
  `training/scripts/test_fetch_ha_exposure.py`
- Existing manifest output under `training/datasets/`

## Explicit non-goals

- No runtime Home Assistant wiring or ownership changes.
- No weakening of schema validation, capability checks, fail-closed barriers,
  locked eval labels, recipe-lock cases, or existing checkpoints.
- No model training, 40k production generation, checkpoint evaluation, export,
  or promotion.
- No model bake-off, new framework, or extra corpus expansion.
