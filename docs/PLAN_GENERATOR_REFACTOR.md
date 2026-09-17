# Plan: canonical training-data generator refactor

**Status:** implement this plan. Do not train, do not regenerate a full
production corpus, do not change runtime behavior, do not edit the locked
eval suite.

## Scope

Refactor SaySo’s training-data generator onto one canonical pipeline whose
labels, catalogs, and prompts match production, and whose allocation and
acceptance rules are derived from the latest comparable eval evidence.

Out of scope: training a model, publishing a 40k corpus, changing
`custom_components/sayso` request-path behavior, editing `evals/cases/` or
suite expectations.

## Files to touch

Primary (replace / reorganize):

- `training/generators/` (entire package)
- `training/configs/generation/production.yaml` (new)
- `training/configs/generation/smoke.yaml` (new)
- `training/tests/` and colocated `training/generators/test_*.py` (migrate into
  `training/tests/`)
- `training/scripts/build_synthetic_dataset.py` and callers
  (`generate_training_supplement.py`, `generate_balanced_test_data.py`,
  `v2_scenarios.py`, `rendering.py`, `llm_curation.py`)
- `docs/TRAINING_PLAN.md` (design map and allocation rationale only)
- `training/README.md` (commands)

Do not touch: `evals/cases/`, `evals/suites/`, `custom_components/sayso/`
runtime modules except *importing* the existing prompt renderer / tool
compiler / parser / validator. `ARCHITECTURE.md` is unchanged.

---

## 1. Eval evidence (latest completed artifacts)

Host: `ubuntu@192.168.1.140` (`ai-inference`). Suite is the locked 120-case
`promotion` set (`evals/suites/promotion.yaml`).

All three scored runs share:

| Field | Value |
|---|---|
| suite | `promotion` |
| adapter | `endpoint` |
| server | `http://127.0.0.1:8091` |
| decoding | `temperature=0`, `max_output_tokens=160` |
| case_hash | `58b5892320175371677f393f4b562906f77aa49d7e39851b6744ac6ed992b5c9` |
| suite_hash | `1c13634d2192a74caa6e2553b6afd8672a66a14ffdbd941ed9f279f3f2341017` |
| production_contract_fingerprint | `9014652e2c35bf209222f5b5c953d6919664f3f656a85ff1d0d50c509a07476e` |
| outcomes | 120 rows + 120 raw files |

These three are comparable to each other. They are **not** comparable to
TRAINING_LOG `structured` / host `rawparse` rows.

| Run | Checkpoint | Completeness | Passed |
|---|---|---|---|
| `20260917T170643Z-57f5cf7d` | `lfm-base` | **incomplete** (3 transport errors) | 2/120 |
| `20260917T171446Z-e6178ea3` | `v3-40k-gauntlet` (Gauntlet v1 / Run 013 step-2500) | complete | 31/120 |
| `20260917T172123Z-8061c3ad` | `ha-contract-v2-step2500` | complete | 52/120 |

Base is recorded but not used as a score peer because the run is incomplete.

Later ha-contract checkpoints exist only as HF trainer states; training was
still at ~step 3713/5000 when these evals ran. No later GGUF was scored.
The evaluated ha-contract artifact is
`/srv/training-runs/ha-contract-v2-eval/artifacts/step-2500-Q8_0.gguf`.

### Dataset that produced each checkpoint

| Checkpoint | Dataset | Seed / generator notes |
|---|---|---|
| Gauntlet v1 | `/srv/datasets/sayso_40k_grounded_gauntlet_20260911/` | seed `20260911`, generator `6fdd46b`. Bare tool names. Grounding requested 3.0% **achieved 0.22%**. Manifest accepted the shortfall. |
| ha-contract-v2 step-2500 | `/srv/datasets/sayso_40k_ha_contract_v2_20260914/` | seed `20260913`, built from `sayso-gauntlet-v2-disc-20260914-source`. `namespaced_tools=1.0`, `full_tool_catalog=1.0`, grounding 2.88%, discrimination 2.0%, real-home 10%. |

Gauntlet-v2 corpus (`sayso_40k_grounded_gauntlet_v2_20260914`) mixed contracts
(`namespaced_tool_rate=0.35`, `full_catalog_rate=0.35`) and was **not** the
evaluated checkpoint.

### Latest comparable scores (promotion, same scorer)

| Metric | Gauntlet v1 | ha-contract-v2 |
|---|---:|---:|
| overall | 0.258 | **0.433** |
| action_execution_accuracy | 0.20 | **0.49** |
| false_action_rate | 0.35 | **0.80** |
| clarification_accuracy | 0.30 | 0.00 |
| refusal_accuracy | **0.80** | 0.30 |
| malformed_output_rate | 0.008 | 0.00 |
| invalid_argument_rate | 0.10 | 0.042 |
| missing_tool_call | **29** | 1 |
| unexpected_tool_call | 46 | 61 |
| climate | 0.2 | **0.9** |
| routines_vacuum | 0.5 | **0.9** |
| light_fan_settings | 0.5 | **0.7** |
| ordinary | 0.1 | 0.4 |
| status | 0.0 | 0.2 |
| aliases | 0.1 | 0.5 |
| multi_action | 0.4 | 0.5 |
| exclusion | 0.0 | 0.0 |
| ambiguity | 0.3 | 0.0 |
| unavailable | **0.8** | 0.3 |

### Historical findings vs this checkpoint

Treat the prompt’s earlier list as hypotheses. Against ha-contract-v2:

| Finding | Latest status |
|---|---|
| Better action execution with ha-contract-v2 | **Confirmed.** 0.49 vs 0.20. Climate / vacuum / light-set are the preserved successes. |
| Excessive refusals and malformed calls in Gauntlet v1 | **Refusals confirmed** (29 missing_tool_call; ordinary_01 refuses “turn on kitchen counter lights”). Malformed is low (1); Base was the malformed-heavy run and is incomplete. |
| Calls instead of clarification | **Confirmed, worse on ha-contract.** 10/10 ambiguity → `action_on_ambiguity`. Gauntlet 5/10. |
| Calls to unavailable capabilities | **Confirmed on ha-contract** (6 `tool_call_on_unavailable`). Gauntlet refuses these (0.8) *and* many valid actions. |
| State-changing calls for status questions | **Confirmed.** 8 `status_state_change`. Example: “is the lock locked?” → `intent__HassTurnOn` instead of `homeassistant__GetLiveContext`. |
| Failure to handle exclusions | **Confirmed, 0/10 both models.** Typical: “turn off kitchen lights except the ceiling light” → calls **only** the excluded entity. |

### Corpus vs behavior (ha-contract-v2, 40k accepted)

| Outcome | Rows | Share |
|---|---:|---:|
| action | 29,229 | 73.1% |
| status | 9,065 | 22.7% |
| absence | 965 | 2.4% |
| clarify | **382** | **0.95%** |
| unsupported | 359 | 0.9% |
| exclusion_rows | 424 | 1.06% |
| multi_call | 2,621 | 6.6% |
| by_difficulty ambiguity | 2,598 | 6.5% |
| by_difficulty unsupported | **11** | ~0% |

Status volume is already high, so status failures are **not** a data-volume
problem. Ambiguity *tags* are common, but clarify *labels* are rare: most
“ambiguity” difficulty rows are still action-labeled. That is a label defect.

### Representative traces (do not copy into training)

**Status (incorrect tool family, not missing rows).**
`status_01`: expected `homeassistant__GetLiveContext(domain=lock, name=Front Door Lock)`;
model `intent__HassTurnOn(device_class=[door], name=Front Door Lock)`.

**Ambiguity (action instead of clarify).**
`ambiguity_01`: expected empty calls / clarification; model
`intent__HassTurnOn(name=Living Room Floor Lamp)`.

**Exclusion (wrong remaining targets).**
`exclusion_01`: expected turn off Counter Lights + Sink Light, forbid Ceiling Light;
model turns off **Kitchen Ceiling Light** only.

**Unavailable (call or wrong abstention family).**
`unavailable_01`: expected refusal; model turns on Kitchen Counter Lights.
`unavailable_04`: both checkpoints reply “Which routine did you mean?” — the
script-clarify template leaking into a withheld-capability case.

**Successful controls to preserve.**
`ordinary_01`, `climate_01`, `routines_vacuum_01`, `light_fan_settings_01`,
`multi_action_01` on ha-contract: correct namespaced tools and arguments.

---

## 2. Defect classification

| Symptom | Class | Generator implication |
|---|---|---|
| Status → HassTurnOn despite 22% status rows | Incorrect / weak contrast labels | Matched status/action twins. Same home and wording family; only the request semantics change. Label status as `GetLiveContext`, never a state-changing intent. |
| 2598 “ambiguity” tags vs 382 clarify labels | Incorrect training labels | Difficulty tags are not labels. Genuine multi-match with unresolved production context → clarify. Unique match resolved by satellite/area/name → action. |
| Exclusion calls the forbidden entity | Incorrect labels + missing completeness | Remaining-target calls must omit excluded entities. Partial one-call labels are wrong. |
| Unavailable → action on ha-contract; ordinary → refusal on Gauntlet | Missing withheld-tool coverage *and* over-refusal risk | Distinct families: withheld capability, absent device, unsupported feature, genuine clarify. Never fill action quotas with refusals. Do not flood negatives. |
| “Which routine did you mean?” on unavailable_04 | Contradictory / leaked label family | Script-clarify text is only legal for script ambiguity. |
| Gauntlet bare names vs production `intent__HassTurnOn` | Prompt / schema mismatch | Every row uses production names and `production_catalog(home)`. Remove mixed-contract controls (`namespaced_tool_rate`, answer-first `offered_tools` subsets). |
| Grounding 0.22% shipped as success (Gauntlet v1) | Silent shortfall | Fail closed. No `--allow-rate-shortfall`, no `min_rate_achieved_fraction`, no empirical ceiling that redefines success. |
| Area rows reserved then appended | Bypass of acceptance | Area scenarios go through the same generate → label → wording → validate → duplicate → coverage path. |
| `offered_tools(..., called_names first)` | Answer-dependent catalog | Catalog from household exposure + production availability. Withhold only for explicit unavailable scenarios. |
| `TRAINING_COVERAGE_EXCLUDED = {GetDateTime}` hidden from every row | Hiding advertised tools | If the pinned contract offers it, offer it. Represent coverage honestly; add positive rows or declare and enforce scope in the recipe, do not silently drop. |
| Dual CLI (`generators/cli.py` + `build_synthetic_dataset.py` defaulting to v1/v2 LLM) | Multiple generators | One CLI, one config loader. Delete the LLM verbalizer/judge path after migrating any still-useful deterministic coverage. |
| Token budget 4096 vs eval prompts ~5011 | Invalid length measurement | Measure with the pinned tokenizer and actual chat template, including tools and output. Reject/regenerate; never truncate supervision. |
| Climate/vacuum/light-set already good | Preserve coverage | Do not “fix abstention” by drowning the corpus in negatives. |

Not classified as volume problems: status, climate, vacuum. Not classified as
eval bugs: the locked promotion expectations match production (status uses
GetLiveContext; ambiguity wants no call; exclusion lists remaining targets).
Do not change eval expectations.

Potential capacity limit (record, do not “fix” with more of the same): 230M
models overfit verbatim names (Run 008 diagnosis). Keep name variation and
discrimination rows; do not copy eval wording.

Evidence gap: code_revision in eval metadata is `unknown`; dataset identity
is recovered from host manifests and the ha-contract training YAML, not from
the eval `metadata.json`.

---

## 3. Target layout

Mirror the consolidated `evals/` package shipped in PR #74 and released as
1.5.0 by [PR #75](https://github.com/mossipcams/SaySo/pull/75). That package
is one CLI, one runner, YAML suites, package tests, and no second entry
path. The generator is the same shape under `training/`:

| evals (PR #74 / 1.5.0) | training generator |
|---|---|
| `evals/cli.py` — `python -m evals.cli run --suite smoke` | `generators/cli.py` — `python -m generators.cli --config configs/generation/smoke.yaml` |
| `evals/runner.py` — orchestration only | `generators/pipeline.py` — orchestration only |
| `evals/cases.py` + `evals/homes/` | `generators/scenarios/` + `homes.py` / `real_home.py` |
| `evals/scorer.py` | `generators/validation.py` |
| `evals/outcomes.py` + `write_run` | `generators/manifest.py` |
| `evals/suites/{smoke,promotion}.yaml` | `training/configs/generation/{smoke,production}.yaml` |
| `evals/config/gates.yaml` | recipe `allocations:` / `coverage:` blocks |
| `evals/tests/` | `training/tests/` (not `training/generators/test_*.py`) |
| `evals/results/` | `training/datasets/` |

Do not add modules that have no evals analogue (`acceptance.py`,
`contamination.py`). Row generation belongs in `pipeline.py` the way
`evaluate()` lives in `runner.py`. Holdout/overlap checks belong in
`validation.py`. Keep existing focused helpers that already map to work
(`tools.py`, `context.py`, `ohf.py`, `coverage.py`, `grounding.py`, …).

One package, one CLI, one config loader:

```text
training/
├── generators/
│   ├── cli.py
│   ├── config.py          # YAML recipe loader; no argparse-only second config
│   ├── pipeline.py        # orchestration only (like evals/runner.py)
│   ├── planning.py        # recipe feasibility + quota retries
│   ├── homes.py
│   ├── real_home.py
│   ├── scenarios/         # structured facts (homes, semantics, area, exclusions)
│   ├── labels.py          # facts + production policy → expected behavior
│   ├── utterances.py      # wording after labels; OHF/Hassil + STT
│   ├── rendering.py       # production prompt/tools/messages
│   ├── validation.py      # semantic + serialized-format + token length
│   ├── coverage.py
│   ├── deduplication.py
│   └── manifest.py
├── configs/generation/
│   ├── production.yaml    # versioned allocation recipe
│   └── smoke.yaml         # generator smoke (not evals/suites/smoke.yaml)
├── datasets/
└── tests/                 # all generator tests live here
```

`pipeline.py` must not accumulate scenario-specific rules. Scenario families
live under `scenarios/`. Area scenarios currently in
`training/generators/area_scenarios.py` move into that package and run
through the main acceptance loop. Delete leftover dual names after the
move: `validate.py` (use `validation.py`), `duplicates.py` (use
`deduplication.py`), `validator.py`, `area_scenarios.py`.

Reuse: `homes.py`, `real_home.py`, `utterances.py` (OHF/Hassil/STT),
`coverage.py` accounting ideas, `grounding.py` contrast families (as scenario
supply, not a post-hoc append), `capability_registry.py` as data,
`sayso_contract` / integration `area_context`, `tool_schema.compile_source_tools`,
`lfm_parse`, `adapters/schema.py`.

Delete after migrating useful coverage (no wrappers, aliases, or fallbacks):

- `training/scripts/build_synthetic_dataset.py` v1/v2 LLM path
- `training/scripts/v2_scenarios.py`, `llm_curation.py`, `scripts/rendering.py`
- argparse-only second CLI surface; `--allow-rate-shortfall`;
  `full_catalog_rate`; `namespaced_tool_rate`; answer-first `offered_tools`
  subset mode
- colocated `training/generators/test_*.py` (move into `training/tests/`)
- `generate_training_supplement.py` if its only job was a second generator;
  any still-needed held-out builder (`generate_balanced_test_data.py`) must
  call the canonical CLI/config

Every caller updates to `python -m generators.cli --config ...`.

---

## 4. Pipeline contract

1. **Facts first.** Structured scenario: household, exposed entities,
   request semantics, satellite context, available capabilities, ambiguity
   conditions, exclusions, expected-behavior *kind* (not wording).
2. **Labels from facts + production policy.** Tool names/schemas are the
   pinned production contract. Catalog =
   `production_catalog(home, removed_tools=...)` from exposure. Never select
   tools around the expected answer. Explicit unavailable scenarios may
   withhold a capability.
3. **Wording last.** Deterministic OHF/Hassil, then optional
   meaning-preserving variation and STT noise. Optional paraphraser may
   change wording only. Revalidate targets, settings, negation, exclusions,
   and action completeness after every transform.
4. **Independent families from eval failures.** New homes, names, wording.
   Never copy or closely paraphrase `evals/cases/` utterances. Existing
   `excluded_train_utterances()` / `quality_eval_overlap` stay as a hard gate
   on the *final* serialized user text, before and after wording transforms.
5. **Area context** from runtime-available inputs
   (`build_area_context(utterance, home_areas, satellite_area)`). Never set
   `target_area` from the expected label.
6. **Validate the serialized row** (messages, tools JSON, assistant masks),
   not just the spec dict.
7. **Token length** via pinned tokenizer + actual chat template, including
   tools and output. Oversized → reject/regenerate, never truncate.
8. **Split** by household / scenario family / contrast group, not by row.
   Preserve both sides of a contrast through acceptance and splitting.
9. **Failed builds** write nothing over existing training inputs. Partial
   artifacts stay in a temp dir and are discarded.

---

## 5. Allocation recipe (initial, not failure% → sample%)

Rationale lives in `docs/TRAINING_PLAN.md`. The YAML recipe is the
enforceable artifact. Counts below are *shares of accepted rows* for the
production recipe; the smoke recipe is a scaled-down feasible subset.

Primary behavioral allocation (accepted rows):

| Family | Initial share | Why |
|---|---:|---|
| Ordinary / unique-target actions | 0.40 | Preserve climate, vacuum, light/fan, turn-on/off. Core usage. |
| Status reads (`GetLiveContext`) | 0.12 | Already 22% by count and still failing; keep a solid floor as **contrasts**, not extra volume. |
| Settings (brightness/speed/temp/volume) | 0.10 | ha-contract is 0.7 here; keep it. |
| Multi-action complete | 0.08 | 0.5 pass; failures are partial/wrong-target, not missing the skill entirely. |
| Genuine clarification | 0.08 | Eval 0/10; corpus clarify 0.95%. Raise because labels were wrong, not because 100% of failures become 8% of data. |
| Exclusions (complete remaining targets) | 0.06 | 0/10; 1% of corpus and often incomplete. |
| Unavailable / withheld capability | 0.05 | Distinct from refusal-of-present-device. Keep small so Gauntlet-style ordinary refusals do not return. |
| Absence (device/area missing) | 0.04 | Distinct from unavailable-tool. |
| Unsupported device feature | 0.03 | Distinct from withheld tool and absence. |
| Aliases / discrimination | 0.04 | Aliases 0.5; keep without turning the corpus into name-echo. |

Cross-cutting (must hold *inside* the primary families, not as extra rows
appended later):

- Area families: implicit satellite area; explicit area override; exact
  entity in/out of satellite area; unique vs genuine ambiguous matches;
  area aliases; area-wide actions; missing/unresolved area; multi-action
  across areas; exclusions within an area.
- Repeated names do **not** require clarification if production context
  resolves the target.
- Contrast groups: status↔action, clarify↔action, available↔withheld,
  present↔absent, supported↔unsupported, complete↔partial multi-action,
  included↔excluded. Both sides survive split.
- Positive coverage counted on actual calls, not offered schemas.
- Clarification, refusal, absence, unsupported-device remain distinct.
- GetDateTime (or any advertised in-scope tool) has honest positive
  coverage or an explicit recipe exclusion that is also withheld from
  distractors — never silently dropped while still advertised.

Before generation: requested count × recipe must be feasible given
scenario supply × near-duplicate limit. Fail if not.

During generation: count accepted rows only; retry inside the unmet
allocation; fail on exhausted retries or impossible quotas.

---

## 6. Manifest

Must include and reconcile with the written dataset:

- recipe identity (path + hash) and generator code identity (git sha)
- seed and split
- schema, renderer, and tokenizer identities
- dataset hash of the final JSONL
- requested vs achieved allocations (primary + cross-cutting)
- actual positive coverage by tool and operation
- outcome, area, and contrast coverage
- rejections and duplicate checks
- token-length distribution (min/p50/p95/max, oversized rejects)
- final row count

Every total must sum to the written row count. A failed build must not
publish partial artifacts or overwrite existing training inputs.

---

## 7. Tests (independent scenarios, never eval utterances)

Focused generator tests for confirmed defects:

- Determinism (same seed → byte-identical JSONL + manifest hashes)
- Quota enforcement (impossible quota fails before the loop; exhausted
  retry fails; no silent shortfall)
- Production rendering (namespaced tools, shared prompt/area renderer)
- Answer-independent catalogs
- Withheld-capability rows actually omit the tool
- Area resolution (satellite vs explicit vs unresolved)
- Status reads → GetLiveContext, not HassTurnOn
- Genuine ambiguity → clarify; context-resolved repeated names → action
- Exclusions omit forbidden targets and include the rest
- Multi-action completeness
- Contamination prevention (holdout utterances, including post-STT)
- Token limits (reject oversized; no truncated assistant masks)
- Manifest reconciliation
- Contrast groups preserved through split
- Area rows pass the same validators as quota rows

Record in TRAINING_PLAN which locked eval *categories* each correction is
intended to affect. Do not claim model accuracy from generator tests.

## Verification

1. `python -m pytest training/tests -q`
2. Generator smoke: `python -m generators.cli --config training/configs/generation/smoke.yaml`
3. Inspect smoke rows from status, clarify, unavailable, exclusion,
   multi-action, and successful action controls.
4. Confirm eval suite files are unchanged (`git diff -- evals/` empty).
5. Confirm no runtime integration edits except unused-import-free reuse of
   existing production helpers.

Smoke YAML must be small, deterministic, and able to satisfy its own
recipe. It is not `evals/suites/smoke.yaml`.
