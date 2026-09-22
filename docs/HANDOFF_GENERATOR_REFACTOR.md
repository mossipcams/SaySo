# Handoff: generator refactor — review, then fix remaining defects

Status: **layout is done. Do not train. Do not regenerate 40k. Do not edit
`evals/` or SaySo runtime.** Next work is a defect pass on the uncommitted
generator tree, then a second code review of that diff.

Written: 2026-09-17.
Worktree: `SaySo__worktrees/ajax-evals`, branch `ajax/evals`.
HEAD: `d606b71 fix(evals): stop runner from importing training jsonschema chain`.
The generator refactor is **uncommitted** (large `training/` + `docs/TRAINING_PLAN.md`
delta vs HEAD).

## TL;DR

One CLI, one YAML recipe, one accept loop, evals-shaped package. Independent
verification on this tree: **272 passed**
(`training/tests` + `training/scripts`, ignoring the two parser suites) and
smoke dry-run **120/120**. `git diff -- evals/` empty.

The remaining work is **label/recipe honesty**, not more file moves. The most
important bug is the same class as the eval finding that started this refactor:
family quotas can be marked met while gold labels a different outcome.

Work in this order: (1) read this review, (2) fix P0 then P1, (3) re-review the
new diff, (4) stop. No training run.

---

## 1. What already landed (do not redo)

| Piece | Where |
|---|---|
| YAML recipes | `training/configs/generation/{smoke,production}.yaml` |
| CLI | `python -m generators.cli --config …` (`--dry-run` supported) |
| Orchestration | `training/generators/pipeline.py` (~441 lines, analogue of `evals/runner.py`) |
| Per-row build | `training/generators/row_generation.py` |
| Facts | `training/generators/scenarios/` (`core`, `area`, `discrimination`, `unavailable`) |
| Planning | `training/generators/planning.py` (`FamilyTracker` when `allocations` set) |
| Validation / dups / manifest | `validation.py`, `deduplication.py`, `manifest.py` |
| Duals deleted | `area_scenarios.py`, `duplicates.py`, `validate.py`, `validator.py`, colocated `generators/test_*.py`, `build_synthetic_dataset.py`, `v2_scenarios.py`, `llm_curation.py`, `scripts/rendering.py`, `generate_training_supplement.py` |
| Held-out builder | `training/scripts/generate_balanced_test_data.py` wraps `GeneratorConfig.from_yaml` + `run_build` |
| Commands | `training/README.md`, `docs/TRAINING_PLAN.md` implementation map |

Area rows go through `generate_row` → validate → duplicate (no post-loop append).
Rendered rows use `production_catalog(home)` in `rendering.py`.

---

## 2. Code review

Findings are from the current tree, not from historical notes. Severity:

- **P0** — recipe/label lie; same defect class as ha-contract eval (clarify tags vs action labels)
- **P1** — recipe field or gate that does not actually enforce what it claims
- **P2** — leftover duals, docs drift, dead code

### P0 — family slots do not force gold labels

`planning.FAMILY_ROBUSTNESS` maps **both** `clarify` and `absence` to
`robustness="ambiguity"`. `gold_from_scenario` never reads `slot["family"]`.
For `robustness == "ambiguity"` it calls `_ambiguous_gold`, which labels:

| Graph | Label |
|---|---|
| 2+ supporting matches | `clarify` |
| 1 match | **action or status** |
| 0 matches, nothing present | `area_unavailable` (absence) |
| 0 matches, present but incapable | `device_unsupported` |

So a `clarify` quota slot with a unique match is accepted as family `clarify`
and labeled as an action. `FamilyTracker` still counts it. This is the training
analogue of “2598 ambiguity tags vs 382 clarify labels.”

`unavailable` maps to `robustness="unsupported"` and shares gold with the
`unsupported` family. Withheld-capability vs present-device-unsupported is not
a distinct label path. Lawn-mower `SupportLevel.UNAVAILABLE` returns
`expected_no_action("unsupported")` without removing the tool from the catalog.

**Fix:** facts first. The slot family must constrain the graph *before* gold:

- `clarify` — guarantee unresolved multi-match; never accept a unique match
- `absence` — capability missing in the requested area; `inject_missing=False`; gold `area_unavailable` / `device_absent`
- `unavailable` — withhold the capability’s tool from `production_catalog(..., removed_tools=…)`
- `unsupported` — entity present, feature missing; tool still offered
- `ordinary` / `status` / `settings` — unique resolved target of the right kind

Reject the row (retry the slot) if gold’s outcome does not match the family.
Add tests that assert **outcome**, not just `stats["quota"]["achieved_family"]`.
`test_sampling.test_generation_hits_family_allocations` currently only checks
the family tracker.

Files: `planning.py`, `row_generation.py`, `scenarios/core.py`, `gold.py`,
`training/tests/test_canonical_generator.py`, `test_sampling.py`.

### P0 — GetDateTime recipe vs silent drop

Smoke and production YAML both set `coverage.get_datetime_positive_min: 1` and
`recipe_exclusions.withheld_from_catalog: []`. Comments say GetDateTime must be
advertised honestly.

Nothing reads `get_datetime_positive_min` after `GeneratorConfig.from_yaml`.
`TRAINING_COVERAGE_EXCLUDED = {"GetDateTime"}` still strips it from
`covered_tool_names()`, so the audit never requires a positive row.
`test_tools_outside_declared_coverage_are_not_offered` intersects **namespaced**
offered names with the **bare** set `{"GetDateTime"}` — the assertion cannot
fail even if `homeassistant__GetDateTime` is on every row.

`test_production_catalog_is_answer_independent` already expects GetDateTime in
the catalog. The coverage-excluded path and that test contradict each other.

**Fix:** pick one contract and enforce it:

1. Honest coverage (matches the YAML and the original plan): delete
   `TRAINING_COVERAGE_EXCLUDED` (or empty it), add a GetDateTime scenario family,
   fail the audit when `get_datetime_positive_min` is unmet, fix the test to
   compare namespaced names.
2. Explicit recipe exclusion: set `get_datetime_positive_min: 0`, put
   `homeassistant__GetDateTime` in `withheld_from_catalog`, and actually withhold
   it from `production_catalog`. Do not leave a min of 1 that is never checked.

Prefer (1) unless a GetDateTime scenario cannot be labeled without copying eval
wording. Files: `capability_registry.py`, `audit.py`, `pipeline.py`,
`scenarios/`, `test_coverage_gates.py`, `training/README.md` (still documents
the silent drop).

### P1 — smoke area rounding drops 11/13 families

`required_counts` is `max(min, round(count * per_1000 / 1000))` with every
`min: 0`. At smoke `n=120`:

| `per_1000` | rows |
|---:|---:|
| 8 | 1 (`implicit_satellite_area`) |
| 6 | 1 (`explicit_area`) |
| 4 | **0** |
| 3 | **0** |

Smoke accepted 2 area rows and never exercises duplicate-name clarify, missing
satellite, exclusion-within-area, etc. Production 40k is fine (all families
≥120). The smoke recipe cannot catch area-label bugs.

**Fix:** give smoke a feasible integer plan — either a smoke-specific area
distribution with `min: 1` on a subset that still sums under 120, or raise
smoke `count` until `round(n * 4 / 1000) >= 1` (n≥125, safer n=200+). Fail
closed if a scenario in the **recipe** resolves to 0 while `enabled: true`.
Do not lower production `per_1000`.

Files: `training/configs/area_distribution_v1.json` and/or a
`area_distribution_smoke.json` referenced from `smoke.yaml`;
`scenarios/area.py` `required_counts`; `test_canonical_generator.py`.

### P1 — area is a sibling family, not a cross-cut

Plan: area conditions hold *inside* primary families. Implementation: area
minimums are subtracted from `count`, remaining slots are primary families,
`family="area"`. That is better than the old append, but smoke/production
“ordinary 40%” is 40% of `(count - area_total)`, and area rows are not also
status/exclusion/etc.

Acceptable if documented. If you keep this model, say so in `TRAINING_PLAN.md`
and stop calling area “cross-cutting inside primary.” If you honor the plan,
tag area as metadata on ordinary/clarify/exclusion slots instead of a separate
family.

### P1 — post-accept case shuffle skips validation

`generate_row` already randomizes request case. `_run_generation` then mutates
accepted `user["content"]` again (`pipeline.py` ~329–335) **after** overlap
and token checks. Holdout uses `casefold()`, so exclusion probably still holds,
but token length and STT metadata can drift.

**Fix:** one casing site (utterance transform), then validate. Delete the
post-loop shuffle.

### P1 — two planners

`config.allocations` → `FamilyTracker`. Empty allocations (bare
`GeneratorConfig()` in many tests) → `QuotaTracker` in `sampling.py`. One loop,
two planners. Production YAML always has allocations.

**Fix:** default `GeneratorConfig.allocations` to `default_allocations()`, or
require a recipe in `run_generation`. Migrate tests that still need the old
quota grid onto YAML. Do not keep a silent QuotaTracker production path.

### P1 — production `grounding.rate: 0.0`

ha-contract shipped ~2.88% grounding; Gauntlet v1 silently shipped 0.22%.
`config.py` still defines unused `DEFAULT_GROUNDING_RATE = 0.028`. Production
recipe requests 0. If that is deliberate (eval failures were clarify/status/
exclusion, not grounding delivery), document it. If not, restore a feasible
nonzero rate and keep the fail-closed gate.

`discrimination.rate: 0.04` is set; delivery ceiling in
`scenarios/discrimination.py` is still `0.02`. `enforce_rate_gate` compares
against `discrimination_available_share` so 4% request vs 2% ceiling can pass
as “reached ceiling.” Confirm that is intended; if the recipe asks 4%, raise
capacity or lower the recipe.

### P1 — `offered_tools(..., called_names first)` still exists

`rendering.py` uses `production_catalog` only. `offered_tools` in `tools.py` is
the answer-first subset the plan said to delete. Dead code with a dangerous
default (`full_catalog=False`). Delete it and any test that only exists for
the subset path.

### P2 — docs and second entry

- `training/README.md` still describes `--home-recipe`, `--real-home-rate`,
  `--synthetic-only` CLI flags. Current CLI is `--config` and `--dry-run` only.
  Mixing knobs live in YAML.
- `training/configs/lfm25-230m-synthetic-v3-40k-trl.yml` and
  `lfm25-230m-40k-grounded-gauntlet-trl.yml` comments still name
  `build_synthetic_dataset.py`.
- `generate_balanced_test_data.py` is a second argparse entry that calls
  `run_build`. Fine if it stays a 50-line wrapper; do not grow it. Optional:
  subprocess/import `generators.cli`.
- `GeneratorConfig.from_yaml` uses `recipe_path.parents[2]` when `repo_root`
  is omitted (`training/`, not repo root). CLI/tests pass `repo_root`. Fail
  if `repo_root` is missing rather than guessing.
- `rates.py` `slot_ceiling` swallows all exceptions and returns `1.0`.
- `row_generation.py` `if variant is not None: pass` is a no-op branch.

---

## 3. Fix plan (implement after this review)

Save no new architecture plan unless scope changes. This file is the plan:
scope, files, verification.

**Out of every increment:** `evals/`, `custom_components/sayso/` runtime,
training a model, writing a 40k corpus, `ARCHITECTURE.md`.

### Increment A — family → gold (P0)

Files: `planning.py`, `row_generation.py`, `scenarios/core.py`, `gold.py`,
`training/tests/test_canonical_generator.py`, `test_sampling.py` (and any
smoke-scale test that only asserted family counts).

Acceptance:

- A `clarify` slot never accepts an action/status label.
- An `absence` slot never accepts an action or a script-clarify template.
- An `unavailable` slot omits the withheld tool from `row["tools"]`.
- An `unsupported` slot keeps the tool in the catalog and labels no_action.
- Manifest `outcome_coverage` for clarify/absence/unsupported is nonzero on
  smoke and consistent with family intent (not merely `achieved_family`).

Verify:

```bash
PYTHONPATH=training python3 -m pytest training/tests -q \
  --ignore=training/tests/test_lfm_python_parse.py \
  --ignore=training/tests/test_llamacpp_parse.py
cd training && PYTHONPATH=. python3 -m generators.cli \
  --config configs/generation/smoke.yaml --dry-run
```

Inspect smoke rows for families `clarify`, `absence`, `unavailable`,
`unsupported`, `status`, `exclusion` (outcome vs metadata.family).

### Increment B — GetDateTime contract (P0)

Files: `capability_registry.py`, `audit.py` / `pipeline.py`, scenario supply,
`test_coverage_gates.py`, `training/README.md`.

Acceptance: YAML `get_datetime_positive_min` is enforced or explicitly 0 with
withholding. No test that intersects bare `GetDateTime` with namespaced tool
names. Catalog and positive coverage tell the same story.

Same pytest + smoke as A.

### Increment C — smoke area feasibility + drop post-loop casing (P1)

Files: area distribution YAML, `scenarios/area.py`, `pipeline.py`, smoke
config if count changes, tests.

Acceptance: every area scenario listed in the **smoke** distribution has
`required_counts >= 1` or the recipe omits it. No user-text mutation after
`validate_row`.

Same pytest + smoke. Confirm `git diff -- evals/` empty.

### Increment D — one planner, dead catalog helper, docs (P1/P2)

Files: `config.py`, tests still constructing bare `GeneratorConfig()`,
`tools.py` (`offered_tools`), `training/README.md`, trainer YAML comments,
optional `from_yaml` repo_root check.

Acceptance: recipe-backed runs never enter `QuotaTracker`. `offered_tools`
gone. README matches `generators.cli`. Grounding/discrimination recipe vs
ceiling documented or aligned.

Same pytest + `python3 -m pytest training/scripts -q`.

### Stop

After D: re-read the diff against this review. Do not start a 40k generate.
Do not claim model accuracy from generator tests. Record in `TRAINING_PLAN.md`
only if allocation semantics changed (area as sibling family, grounding rate).

---

## 4. Current verification baseline

Already run on this worktree (2026-09-17):

```bash
PYTHONPATH=training python3 -m pytest training/tests training/scripts -q \
  --ignore=training/tests/test_lfm_python_parse.py \
  --ignore=training/tests/test_llamacpp_parse.py
# 272 passed

cd training && PYTHONPATH=. python3 -m generators.cli \
  --config configs/generation/smoke.yaml --dry-run
# accepted 120; area achieved only implicit_satellite_area=1, explicit_area=1
```

Treat 272 as the floor. New family/outcome tests must fail before the gold
fix if you are doing TDD.

---

## 5. Constraints

- Python, existing pytest trees only (`training/tests/`, `evals/tests/`,
  colocated `custom_components/sayso/test_*.py`). No new top-level `tests/`.
- Do not train on ChatML `<tool_call>` labels or on `evals/cases/` utterances.
- Do not commit `context.json`.
- Commit only if the user asks.

Eval evidence that motivated the refactor (do not treat as current scores
after this fix; no new host run is required for the generator pass):

Host `192.168.1.140`, promotion suite, T=0, max 160, same hashes. Gauntlet
31/120, ha-contract-v2-step2500 52/120. Status/ambiguity/exclusion/unavailable
were label-or-contrast failures, not “add more of the same volume.”
