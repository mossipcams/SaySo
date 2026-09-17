# Plan: one production contract, realistic eval v3

Superseded as an execution path by `docs/PLAN_EVAL_CUTOVER.md`. The locked 120
utterances and production contract remain; runners listed below were replaced
by `python -m evals.cli`.

## Problem

Training, evaluation and the integration disagree about what the model sees and
what counts as a correct answer:

- **Area.** Production appends `area=<name>` to the system prompt, HA itself adds
  `You are in area X (floor Y) ...`, and the generator copies HA's sentence.
- **Tool names.** HA 2026.9 names tools `intent__HassTurnOn`. The generator
  emits bare names by default, and `schema.build_tool_map` silently maps bare
  names onto namespaced tools.
- **Tool lists.** Realistic eval v2 offers each case eight tools chosen around
  the expected answer.
- **Scoring.** `eval_v3_rawparse.py` renders the prompt with the Hugging Face
  template, treats parse failures as "no call", and scores abstention without
  telling a clarification from a refusal.

Work in this repository. Fix the shared training and evaluation contract
rather than creating another unrelated eval suite. Do not invent 120 new
utterances; `training/fixtures/realistic_eval_20260908_v2.json` is the
behavioral source of truth.

Inspect before changing: production request path, prompt renderer, tool
compiler, parser, validator, training generator, and

- `training/fixtures/realistic_eval_20260908_v2.json`
- `training/fixtures/realistic_eval_20260908_v2.md`
- `training/scripts/eval_v3_rawparse.py`
- `training/evals/lfm_python_parse.py`
- `training/evals/recipe_lock.py`
- `training/evals/v3_quality.py`
- `training/evals/grounding_eval.py`
- `evals/cases/v1.json`
- `evals/runner.py`
- `evals/scorer.py`
- `custom_components/sayso/area_context.py`
- `custom_components/sayso/lfm_parse.py`

## Scope

### 1. One production contract

The generator, evaluator, and Home Assistant integration must call the same
pure functions for system prompt rendering, tool schema compilation, tool
names, argument schemas, area context, response parsing, and tool-call
validation. Do not copy production prompt or schemas into the eval runner.

- `custom_components/sayso/area_context.py` (pure, no HA imports):
  `AreaContext(satellite_area, target_area, target_area_source)`,
  `build_area_context(utterance, areas, satellite_area)`,
  `render_system_prompt(system_prompt, context)` (strips HA's three legacy area
  sentences and appends the structured block), and `apply_area_context(messages,
  context)`.
  Sources: `explicit` (exactly one area or alias named), `multiple` (more than
  one named), `satellite` (none named, satellite has an area), `none`.
- `conversation.py`: build the context from the satellite/device area and the
  area registry (names + aliases), and apply it to every message list sent to
  the model. `_system_prompt_with_area` is removed.
- `schema.py`: exact tool names only. Remove suffix aliasing from
  `build_tool_map` and `expand_compiled_tool_name_aliases`. Add
  `compile_source_tools(entries)`, the same cached path `compile_tools` uses.
- `tool_contract.py` (pure): `check_tool_calls(calls, compiled_tools,
  exposed_domains)` covers exact name, argument object, unknown/required keys,
  `anyOf` required groups, JSON types, enums, integer bounds, and `domain`
  values that must be exposed. Production runs it before HA's voluptuous
  validation; failures go through the existing correction path.
- Production parser is public and used by eval. A model emitting a tool name
  production would reject must fail the case. No silent
  `HassTurnOn` → `intent__HassTurnOn` scoring. No legacy
  `You are in area Kitchen` in new training or eval rendering.

### 2. Preserve area training

Do not remove areas from the corpus. Newly generated rows use the production
area representation (`satellite_area`, `target_area`, `target_area_source`) via
the production builder/renderer. No generator-only format. No mixing legacy
and production area formats in one dataset.

Area-grounded examples should be common in actionable rows. Configurable,
deterministic coverage for:

- Implicit satellite area: “Turn off the lights”
- Explicit requested area: “Turn off the bedroom lights”
- Exact entity name in the satellite area
- Exact entity name outside the satellite area
- Unique entity resolution across the home
- Duplicate names across areas requiring clarification
- Area-wide actions
- Area aliases
- Missing satellite area
- Invalid or unresolved target area
- Explicit area conflicting with satellite area
- Multi-action requests spanning areas
- Exclusions within an area

`generators/area_scenarios.py` + `training/configs/area_distribution_v1.json`:
deterministic rows for those scenarios. `run_generation` reserves their
quota, then fails when a scenario falls below its minimum, when the
satellite-area share of actionable rows is under its floor, or when any row
carries a legacy area prompt.

Generator prompt path: namespaced tool names, no HA area sentence, final
prompt via `render_system_prompt`. Labels use production names through
`compile_source_tools`. Metadata records `area_context` and
`contract: production_v3`. `namespaced_tool_rate` is removed.
`production_catalog(home, removed_tools)` is the household catalog shared
with the eval.

### 3. Migrate realistic eval v2 to structured v3

`evals/cases/realistic_v3.json` is derived from v2 by
`evals/migrate_realistic_v2.py`. Preserve case IDs, original utterances, 12
categories × 10 cases, five households, expected behavioral outcomes,
intentional unavailable-tool conditions, expected exclusions, and ambiguity
conditions. Replace rendered system prompts with structured inputs:

- Household/entity state
- Satellite area
- Target area and source when applicable
- Available production capabilities
- Intentionally unavailable capability, if any
- User utterance
- Expected response type
- Expected tool calls
- Expected clarification behavior
- Forbidden tool calls or affected entities

Render the prompt at evaluation time through the production renderer. Keep
the v2 fixture immutable.

### 4. Realistic tool availability

Normal cases advertise the household production catalog. Unavailable cases
use that catalog and remove only the capability under test, recorded in the
fixture. Do not offer eight answer-conditioned tools.

### 5. Production-path v3 runner

`evals/realistic_v3.py` stages: build area context → production prompt →
compile production catalog → invoke model → production parse → production
validation → score. Keep raw model output on each case result. Do not
recreate a Hugging Face prompt format if production uses another renderer.

### 6. Malformed output is an explicit failure

Never convert parser failures into an empty call list / successful
abstention. Classify:

- `valid_tool_call`
- `valid_multi_tool_call`
- `valid_clarification`
- `valid_refusal`
- `malformed_output`
- `unknown_tool`
- `invalid_arguments`
- `unexpected_tool_call`
- `missing_tool_call`
- `wrong_response_type`

A parser exception, malformed JSON/Python call, missing tool name, invalid
`inton_domain`, or any production-rejected output fails.

### 7. Score clarification and refusal separately

Ambiguity: no action tool call, and a genuine clarification that asks the
user to disambiguate device, area, or target. Generic “I can’t do that”
fails.

Unavailable: no unsupported tool call, and a response explaining the
capability cannot be performed. Clarification must not pass.

Keep text-matching rules narrow, inspectable, and tested.

### 8. Category-specific behavior

- `status`: namespaced GetLiveContext equivalent; reject state-changing
  calls.
- `ambiguity`: clarification and zero action calls.
- `unavailable`: refusal/capability explanation and zero unsupported calls.
- `exclusion`: excluded entities are not affected; score the complete
  requested action.
- `multi_action`: every requested action; partial execution fails.
- Action categories: reject extra calls that affect unrelated entities.
- Aliases: resolve only aliases in the household fixture or production alias
  data.
- Area-relative: implicit requests use satellite area; explicit areas
  override.

Keep exact-call expectations unless a case declares a safe semantic
equivalent. No broad fuzzy matching.

### 9. Promotion scoring

Realistic eval v3 is the primary promotion suite. Smaller *model* suites
(`eval_v3_rawparse.py`, recipe-lock, gold/shadow, grounding, `evaluate_checkpoint`)
are smoke/diagnostics only and must use the same parser, validator, outcome
model, and scoring primitives. They advertise the household production catalog,
not an 8-tool answer-conditioned subset. Do not rewrite `evals/runner.py`: that
is the Home Assistant conversation-agent path for `evals/cases/v1.json`.

Scores: overall exact pass rate, per-category pass rate, action execution
accuracy, false-action rate, clarification accuracy, refusal accuracy,
malformed-output rate, unknown-tool rate, invalid-argument rate.

`evals/config/promotion_gates_v1.json` fails when a required category
regresses below threshold, ambiguity produces action calls, unavailable
capabilities produce tool calls, malformed output is treated as abstention,
status requests produce state-changing calls, or excluded entities are
affected.

### 10. Deterministic tests

Prove:

- Eval renderer and production renderer produce identical bytes for
  identical structured inputs.
- Production and eval expose identical tool names and schemas.
- Malformed response fails instead of becoming zero calls.
- Refusal does not pass an ambiguity case.
- Clarification does not pass an unavailable case.
- Status action call fails.
- Exclusion that affects the excluded entity fails.
- Partial multi-action output fails.
- Bare legacy tool names fail when production requires namespaced names.
- Explicit area overrides satellite area.
- Implicit area request resolves to the satellite area.
- Duplicate entity names across areas require clarification.
- Generation is deterministic for a fixed seed.
- Area scenario quotas are enforced.
- The v3 fixture contains exactly 120 cases with 10 cases in each existing
  category.

## Files

Integration: `area_context.py`, `tool_contract.py`, `conversation.py`,
`schema.py`, `inference.py`, `lfm_parse.py` / `completion.py` as needed for
the shared parser.
Eval: `evals/contract.py`, `evals/realistic_v3.py`,
`evals/migrate_realistic_v2.py`, `evals/cases/realistic_v3.json`,
`evals/config/promotion_gates_v1.json`, `evals/runner.py`, `evals/scorer.py`
if they must share primitives.
Training: `generators/{context,labels,tools,pipeline,config,cli,validate}.py`,
`generators/area_scenarios.py`, `configs/area_distribution_v1.json`,
`scripts/{eval_v3_rawparse,rendering}.py`,
`training/evals/{v3_quality,recipe_lock,grounding_eval,lfm_python_parse}.py`.
Tests: `tests/test_realistic_v3.py`, `tests/test_schema.py`,
`tests/test_conversation.py`, `training/tests/test_area_scenarios.py`,
plus existing tests that assert the legacy prompt or bare names.

Do not overwrite `training/fixtures/realistic_eval_20260908_v2.json` or
`evals/cases/v1.json`.

## Out of scope

Training a model, running the eval against a checkpoint, and rewriting any
utterance. The promotion thresholds are initial values and need calibrating
against the first checkpoint run.

## Verification

Tests run on `ubuntu@192.168.1.140` from `~/sayso-evals-test` with
`~/sayso-evals-venv`. Sync this worktree to that checkout, then:

- `pytest tests` (integration, eval contract, v3 fixture)
- `cd training && pytest tests evals scripts generators adapters`

Baseline before changes: integration 272 passed / 1 env failure (`mutagen`);
training 388 passed / 9 env failures (cwd-relative `real_home.json`, socket).
