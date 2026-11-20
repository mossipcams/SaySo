# Plan: batch 5 hygiene (Fix 9 steps 2-3, Fix 16, Fix 15, Fix 17)

Branch: `ajax/review-batch5` from `f2db683` (batch 4 tip). This document is the
plan only. No code, test, or config changes are made by it, and nothing is committed.
Parent plan: `docs/PLAN_review_fixes.md` (Batch 5, line 5 of "Order").

## 0. Baseline measured today

| Check | Result |
| --- | --- |
| `ruff check .` (CI config) | All checks passed |
| `ruff format --check .` (CI config) | 11 files already formatted |
| `pytest tests evals/tests --ignore=tests/test_realistic_v3.py` | 2668 passed, 1 skipped |
| `~/.cache/sayso-venvs/satellite/bin/pytest -q satellite/sayso` | 308 passed, 5 skipped |
| `lint-imports` | 2 kept, 0 broken (`pyproject.toml:65-80`) |
| `python -m evals.generate_realistic --check` | exit 0 |
| Stub eval, smoke (in memory, same stub as `evals/results/fix13_stub_eval.py:20-37`) | 16/24 (0.6667) |
| Stub eval, promotion (same) | 441/530 (0.8321) |

The stub-eval numbers were produced in memory. No file under `evals/results/` was
written. They match the batch 3 record (`evals/results/batch3_pr_body.md:26`).

`tests/test_realistic_v3.py::test_a_perfect_run_passes_every_case_and_every_gate` fails today
with `KeyError` at `tests/test_realistic_v3.py:219`. The follow-up completion after tool
execution is not in `by_prompt`. Current `ask()` is at lines 218-228, and the assertion
that fails is at line 230.

## 1. Decisions and approvals needed

Nothing below is done without the user's explicit OK. Items are ordered by
which slice needs them.

| # | Item | Why it needs approval |
| --- | --- | --- |
| A1 | `tests/test_realistic_v3.py` stub change (approved in the batch brief) | Approved. The 530 assertion still fails after it (section 2.1). Asking: fix grounding (Fix 11 residual) or change the expected data? The plan does not weaken the assertion. |
| A2 | `tests/test_compat.py:42-46` (`len(MATRIX) == 1`, `CURRENT_HA_VERSION == "2026.8.3"`) | Existing test assertions change for the latest row (Fix 16). CLAUDE.md: never modify `tests/` unless asked. |
| A3 | `.github/workflows/ci.yml` compat matrix (`include` at lines 127-129 and 159-161) | D4: the gh token lacks `workflow` scope. Blocked until the user approves the edit or grants the scope. Without it the `latest` row is defined and tested locally, but CI never runs it. |
| A4 | `tests/test_grammar_wiring.py:44,67,85,132-133` and `tests/test_grammar_fallback.py:278-279` | Existing assertions read span metadata keys `model`, `grammar`, `grammar_skip_reason`. These rename under Fix 17. |
| A5 | `evals/tests/test_grounding_targets.py:6,89` (imports `DOMAIN_WORDS`) | Only if `DOMAIN_WORDS` moves into the vocabulary table (Fix 9 step 3). Import path edit. |
| A6 | `tests/` format and lint autofix (41 files to format; 20 UP/SIM/RUF findings, section 3) | CLAUDE.md "never modify files in tests/". Format-only edits are still edits. |
| A7 | Deleting executed plans (section 4.2) and `docs/CODE_IMPROVEMENT_PLAN.md` | Deleting docs. Look at each target first, as AGENTS requires. The user confirms the list. |
| A8 | New dev dependency for the typing test (`ty` or `mypy`) in `pyproject.toml` `test` extra (lines 18-31) | Dependency change. `.venv` has neither tool today (section 3.3). |
| A9 | `AGENTS.md:63` pointer change; `ARCHITECTURE.md` update for export | `AGENTS.md` is a repo doc. Export sends traces off-box, which is a new runtime communication path (`ARCHITECTURE.md:239-241` trust statement). CLAUDE.md requires an ARCHITECTURE update in that case. |
| A10 | Satellite and scripts: lint and format scope (section 3.1) | `pyproject.toml:43-47` extend-excludes `satellite`, `training`, `scripts`. Removing that for `satellite` adds 128 lint findings and 59 format changes to a separate Python 3.12 CI job. |
| A11 | `target-version` for ruff | `pyproject.toml` `requires-python = ">=3.12"` (line 9); HA needs Python 3.14.2 (`homeassistant/const.py:29` in the 2026.11 snapshot, `REQUIRED_PYTHON_VER`); satellite runs 3.12. Pick one. |
| A12 | Whether `docs/PLAN_batch4_config_v2.md` and `docs/PLAN_fix11_grounding.md` stay active | Batch 4 is in flight on this branch (commits `76e3261`..`d64a3b8`). Fix 11 has two open residual clarifies (section 2.1). |
| A13 | Fix 17 export: OTLP endpoint default, JSONL size cap or rotation, whether a failed export is retried | Product decisions. Section 5.4. |

### Parent decisions

- A1: apply the stub change. Fix `multi_action_03` in grounding, because it's a
  real Fix 11 defect: when checking one call, the whole-utterance rule must also
  accept tokens explained by the other calls in the same validated batch (their
  entity labels and areas, using the same rules as the call's own). It must not
  accept tokens that no call explains. `ordinary_03` waits for the user.
- A4, A5: `tests/test_grammar_wiring.py`, `tests/test_grammar_fallback.py` and
  `evals/tests/test_grounding_targets.py` were created in Batches 2-3, so
  their key/import updates are in scope.
- A7: delete only plans whose work has shipped in a merged PR, or is on these
  stacked branches and finished. Read each one first. `PLAN_review_fixes.md`,
  `PLAN_fix10_grammar.md`, `PLAN_fix11_grounding.md`, `PLAN_batch4_config_v2.md`
  and this plan stay active.
- A8: mypy, pinned exactly in the `test` extra. Strict for `core/`, non-strict
  for the integration. It runs from `tests/test_typing_contracts.py`.
- A9: the AGENTS.md pointer and the ARCHITECTURE.md export note are in scope.
- A10: `satellite`, `training` and `scripts` stay excluded from lint and format
  (separate Python 3.12 deploy target).
- A11: `target-version = "py312"` (the repo's `requires-python`).
- A13: JSONL goes to `<config>/sayso_traces.jsonl`, rotated once at 10 MB. OTLP
  has no default endpoint (the user enters one). A failed export is dropped and
  logged at debug, never retried, and never delays a turn.
- Waiting for the user: A2 (`tests/test_compat.py`), A3 (`ci.yml` matrix), A6
  (format and lint autofix under `tests/`), and `ordinary_03`.

## 2. Fix 9 steps 2-3: speech strings and per-language vocabulary

### 2.1 Step 1 is already in the tree, and the test realistic stub result

- `supported_languages` already returns `["en"]` at `custom_components/sayso/conversation.py:434-435`,
  with a test at `tests/test_supported_languages.py:7-9`. The review's cited
  `conversation.py:393-394` is stale. Step 1 is done.
- The current stub-fix prediction for `tests/test_realistic_v3.py` (A1), measured in a
  temporary copy and then removed:

  Proposed change, inside `ask()` (line 218), following the pattern already used in
  `evals/results/fix13_stub_eval.py:24-25`:

  ```python
      def ask(messages, tools):
          if messages[-1]["role"] == "tool":
              return _body("Done.")
          case = by_prompt[json.dumps(messages)]
  ```

  Result with that change: `summary["passed"] == 528`, not 530. Gates
  (`check_gates(..., expected_count=530) == []`, line 231) still pass. The count
  assertion at line 230 is the only failure.

- The two residual cases are both `grounding_clarify` on the first pass, before any
  follow-up completion. A stub change cannot reach them:
  - `realistic_eval_20260908_multi_action_03` (`evals/cases/realistic_v3.jsonl:24`):
    the model's second call targets `Bedroom Tower Fan`, and grounding returns
    "Did you mean the Bedroom Tower Fan?" (`core/grounding.py:497-504`), so no call runs.
  - `realistic_eval_20261118_ordinary_03` (`evals/cases/realistic_v3.jsonl:77`):
    `Lounge Television` is clarified to `"Sorry, I didn't understand that."`
    (`core/grounding.py:509`).

  So the prediction is "every case passes" does not hold, and there are two residual
  cases, not one. Both are inside grounding (`core/grounding.py`), which is Fix 11's
  residual. The honest options are: fix those two grounding decisions (Fix 11), or
  have the user decide the expected data. The assertion itself must stay as written.
  The file is excluded from CI (`--ignore=tests/test_realistic_v3.py` in
  `.github/workflows/ci.yml`), so CI is unaffected either way.

### 2.2 Spoken constants and where each is produced

Core (HA-free) producers:

| Constant or literal | Defined | Produced |
| --- | --- | --- |
| `ERROR_CANCELLED` ("Okay.") | `core/vocabulary.py:28` | `core/post_model.py:196` |
| "Done." (literal, not in vocabulary) | none | `core/post_model.py:314` |
| `ERROR_NOT_UNDERSTOOD` | `core/vocabulary.py:13` | `core/grounding.py:509` |
| `ERROR_DID_YOU_MEAN` | `core/vocabulary.py:14` | `core/grounding.py:497,504` |
| `ERROR_CHOOSE` | `core/vocabulary.py:15` | `core/grounding.py:507` |
| `ERROR_AREA_HAS_NO_DOMAIN` | `core/vocabulary.py:16` | `core/grounding.py:488` |
| `DOMAIN_PLURALS` (used inside the area message) | `core/vocabulary.py:1-11` | `conversation.py:800` (format input) |

HA-layer producers (`custom_components/sayso/conversation.py`, imports at lines 47-58):

| Constant | Produced at |
| --- | --- |
| `ERROR_NOT_UNDERSTOOD` | default param `:188`; `:222`; `:496` (held answer); `:650`; `:656` |
| `ERROR_REQUEST_TIMEOUT` | `:216` |
| `ERROR_MODEL_UNAVAILABLE` | `:224` |
| `ERROR_EMPTY_RESPONSE` | `:640` |
| `ERROR_TOOL_ITERATION_LIMIT` | `:678` |
| `ERROR_ACTION_FAILED` | `:738`, `:759`, `:778`, `:788`, `:791`, `:797` |
| `ERROR_NO_TIMERS` | `:757` |
| `ERROR_NAME_NOT_FOUND` | `:765` |
| `ERROR_MEDIA_NOT_PLAYING` / `ERROR_MEDIA_NOT_PAUSED` | `:775`, `:777` |
| `ERROR_AREA_HAS_NO_DOMAIN` | `:798` |

Not produced anywhere in the integration (grep, excluding tests): `ERROR_MODEL_NOT_LOADED`
(`core/vocabulary.py:19`, re-exported in `const.py:16`). Check and delete or keep.

`const.py:5-21` re-exports the vocabulary. The review's `const.py:81-95` is stale.

English-only lexical tables that are not speech:

- `core/grounding.py:15` `_TOKEN_RE` (ASCII `[a-z0-9']+`).
- `core/grounding.py:17-28` `GENERIC`; `:30-38` `NOUN`; `:42-47` `DOMAIN_WORDS` (derived from `NOUN`).
- `core/grounding.py:51` `STOP_WORDS`.
- `core/held.py:8-19` `YES_ANSWER`, `NO_ANSWER`, `FILLER` (used at `held.py:34-49`).
- `core/area_context.py:44` strips the English possessive `'s`.
- `custom_components/sayso/routing.py:26` has its own ASCII `_TOKEN_RE`. It is outside
  `core/`. Check whether the live path uses it before Fix 9 lands.

### 2.3 Design

1. `core/vocabulary.py` becomes the one place for core language data:
   - A `Speech` mapping of keys to templates. Keys: `not_understood`, `did_you_mean`,
     `choose`, `area_has_no_domain`, `cancelled`, `done`, `model_unavailable`,
     `model_not_loaded`, `request_timeout`, `empty_response`, `action_failed`,
     `no_timers`, `name_not_found`, `media_not_playing`, `media_not_paused`,
     `tool_iteration_limit`. (`done` is new; it covers `post_model.py:314`.)
   - A `Vocabulary` dataclass with `generic`, `noun`, `domain_words`, `stop_words`,
     `yes`, `no`, `filler`, `domain_plurals`, `possessive_suffixes`.
   - `SPEECH_EN` and `VOCAB_EN` hold the current English values. `VOCABULARY = {"en": VOCAB_EN}`.
     The current `ERROR_*` names stay as aliases of `SPEECH_EN` entries, so existing
     imports keep working (`const.py`, `tests/test_grounding.py:5`,
     `tests/test_match_failure_speech.py:19`, `tests/test_conversation.py`).
2. Core takes the table as data. `TurnInput` (in `core/post_model.py`) gets
   `speech: Speech = SPEECH_EN` and `vocabulary: Vocabulary = VOCAB_EN`. Grounding entry
   points (`core/grounding.py:488-509` and the held helpers in `core/held.py`) take the
   same parameters with English defaults. Existing tests and evals call without
   arguments and keep English output (`evals/runner.py` stays on the defaults).
3. HA layer resolves speech once per turn:
   - `translation.async_get_translations(hass, language, "speech", [DOMAIN])`
     (signature confirmed at `homeassistant/helpers/translation.py:333-339` in the
     2026.8.3 venv). Keys come back as `component.sayso.speech.<key>`
     (`translation.py:327` builds that prefix). Strip the prefix.
   - Language base is taken from `user_input.language`. If the base code is not in
     `VOCABULARY`, use English for speech and vocabulary both. Speech and vocabulary
     must switch together.
   - Missing keys fall back to `SPEECH_EN`.
4. `translations/en.json` and `strings.json` get a top-level `speech` section with
   the 17 keys above. Today both files have only `config`, `options`,
   `config_subentries`, `services`, `issues`, `exceptions` at the top level.
5. `en.json` and `SPEECH_EN` must match. A new test asserts equality, so there is one
   source of truth for the values. An alternative is evals reading `en.json` directly.
   It is not chosen because core unit tests import the `ERROR_*` names.
6. `supported_languages` stays `["en"]` (`conversation.py:434-435`). Widen it only when a
   language has both a speech section and a `VOCABULARY` entry.

### 2.4 Verify

- Stub eval parity: smoke 16/24, promotion 441/530 (section 0 numbers). Run with the
  in-memory probe, or with `evals/results/fix13_stub_eval.py` under a new label so the
  committed `fix13_after_*` files are not overwritten.
- `tests/test_supported_languages.py` still passes.
- New tests: `tests/test_speech_translations.py` (a spoken error resolves through
  `async_get_translations`; English fallback for an unknown language; `en.json` speech
  equals `SPEECH_EN`; a non-English language with no vocabulary uses English for both).
- CI gate (section 6).

## 3. Fix 15: ruff rules, format excludes, typing

### 3.1 Lint measurements

Measured with `ruff 0.17.6` (`.venv/bin/ruff`), `--no-cache`, `extend-exclude` removed
with `--config 'extend-exclude=[]'`, explicit paths `custom_components evals scripts satellite tests`:

| Scope | B,ASYNC | UP,SIM,RUF | Fixable |
| --- | --- | --- | --- |
| `custom_components` | 4 | 9 | — |
| `evals` | 1 | 4 | — |
| `scripts` | 4 | 16 | — |
| `satellite` | 2 | 126 | — |
| `tests` | 0 | 20 | — |
| Total | 11 | 175 | B,ASYNC: 3 fixable, 4 unsafe. Whole set: 132 fixable, 19 unsafe. |

Biggest rules, B,ASYNC: B905 ×3, B023 ×2, B009 ×2 (fixable), ASYNC109, B010 (fixable),
B007, B018. Biggest rules, UP,SIM,RUF: UP045 ×84 (fixable), UP035 ×19, RUF012 ×12,
RUF001 ×8, SIM905 ×7 (fixable), RUF100 ×6 (fixable), UP037 ×5 (fixable), RUF043 ×4.

Caveat to resolve in slice S2: `ruff check --select B,ASYNC .` (root, extend-exclude
removed) reports 5, not 11. Explicit paths report 11. The root run is not trusted until
the cause is found (likely discovery or ignore rules). Use explicit paths for counts.

Config today: `pyproject.toml:43-47` (`extend-exclude`), `:50-51` (`select = ["E9", "F63", "F7", "F82"]`),
`:53-64` (format `exclude`: `custom_components/**`, `docs/**`, `evals/**`, `satellite/**`,
`scripts/**`, `tests/**`, `training/**`, `sayso_contract.py`).

### 3.2 Format measurement (format excludes removed)

`ruff format --check` with `format.exclude=[]` and `extend-exclude=[]`:

| Dir | Would reformat |
| --- | --- |
| `custom_components` | 20 |
| `evals` | 16 |
| `scripts` | 30 |
| `satellite` | 59 |
| `tests` | 41 |
| Total | 166 of 232 files |

Under the current config `ruff format --check .` passes (11 files, section 0).

### 3.3 Typing measurement

Checkers were installed only into `/tmp/tc-venv` (created from the HA 2026.11 snapshot
interpreter, `~/.cache/sayso-venvs/ha-2026.11/bin/python`). `.venv` has neither tool
(`ls .venv/bin`), and `llama_cpp` is not installed in `.venv` (checked), so
`inference.py:131` fails to import.

| Tool | Scope | Result |
| --- | --- | --- |
| `ty 0.0.107` | `custom_components/sayso/core`, `--error-on-warning` | 0 diagnostics (passes) |
| `ty 0.0.107` | `custom_components/sayso` | 26 diagnostics, including 3 `import-not-found` (`llama_cpp`) |
| `mypy 2.4.1` | `core`, `--strict --explicit-package-bases --ignore-missing-imports` | 95 errors in 15 files (`attr-defined` 49, `no-any-return` 12, `type-arg` 11, `arg-type` 9, `no-untyped-def` 5) |
| `mypy 2.4.1` | `custom_components/sayso`, default, same flags | 22 errors in 10 files (`arg-type` 9, `attr-defined` 7, others 1 each) |

Without `--explicit-package-bases`, mypy stops with "Source file found twice under
different module names" (`schema.py`), so the flag is required.

Recommendation: ty. `core/` is already clean under ty, and strictness can be set through
rule configuration. Adopting mypy strict on `core/` means about 95 hand fixes. Either
way, `llama_cpp` needs a module override.

### 3.4 Plan

1. Lint, B and ASYNC first, one commit for the autofixable rules
   (`ruff check --select B,ASYNC --fix`, 3 fixes) and one commit for hand-fixes (8).
2. UP, SIM, RUF next, one commit per rule family. Autofix first, with `--unsafe-fixes`
   off. UP045 (84) is the largest family and is mechanical.
3. Set `target-version` per A11.
4. Remove the format excludes for `custom_components/**`, `evals/**`, `tests/**` and
   `scripts/**` in one formatting-only commit (166 files minus `satellite` if A10 keeps
   it excluded). Do not mix this commit with lint fixes.
5. Typing: `tests/test_typing_contracts.py`, modeled on `tests/test_import_contracts.py`.
   It runs the chosen checker through `sys.executable` and asserts return code 0. CI
   runs it through the existing pytest step, so `.github/workflows/` is not edited (D4).
   Requires A8.
6. Run after Fix 9 (so the new `core/` signatures are the ones being typed).

Note: the CLAUDE.md "never modify tests/" rule applies to steps 1, 2 and 4 wherever they
touch `tests/` (A6). The lint and format scope in step 1 and 2 for `tests/` is 20 findings
and 41 files.

## 4. Fix 16: ADRs replacing executed plans and the compat row

### 4.1 Plan inventory

35 `docs/PLAN_*.md` files exist (the review said 31). `docs/CODE_IMPROVEMENT_PLAN.md`
is outside that glob. Status is from `git log` (first add and last touch per file) and the
commit subjects.

Executed (shipped code). Deletion is in A7 and each becomes one ADR:

| Plan | Shipped by |
| --- | --- |
| PLAN_AREA_ARG_DROP | 71efc71 |
| PLAN_DUCKING | a7e4ce2 |
| PLAN_DUCKING_FIXES | b59f649 |
| PLAN_DUCK_STALE_VOLUME | a139728 |
| PLAN_EVAL_CONTEXT_GATE | df49822 |
| PLAN_EVAL_NATURAL_LANGUAGE | 24dcd08 |
| PLAN_EVAL_SCALE | 24dcd08 |
| PLAN_EVAL_SCORING_FIXES | c7703c5 |
| PLAN_PR132_REVIEW_NOTES | 58e89fa |
| PLAN_STT_VAD_WARMUP | cd833f8, c0cd6ec |
| PLAN_WAKE_HOP_DROPS | 8ddd013 |
| PLAN_WAKE_TV_ENSEMBLE | 008b859 (plan commit only; deployment is recorded in project memory, not git. Confirm) |
| PLAN_WAKE_TV_SYNTHETIC | b1ebc9a |
| PLAN_WAKE_V2_PIPER | d7e4d06, 3701c7c |
| PLAN_agents-md-and-comment-removal | 6c30d14 |
| PLAN_ci_ruff_pytest | 3d5932a |
| PLAN_embedded_prefix_checkpoint | 382623c |
| PLAN_eval_case_fixes | 2f9276c |
| PLAN_eval_clarify_candidates | 41efaec |
| PLAN_eval_live_prompt | 99fa376 |
| PLAN_eval_runtime_tool_layout | d531eaa |
| PLAN_fix10_grammar | 7376fab, 58ca251, c8ae9b1 (grammar builder), 5f0fa4b, 589b5ab |
| PLAN_fix13_post_model_evals | b48303e, 7ebc407, 82f026a, f257c2c, 1f58f42, b2b2c96 |
| PLAN_ha_tool_catalog | 7446648 |
| PLAN_history_question_gate | 5d6a1de |
| PLAN_lva_1_1_18 | 9843212 |
| PLAN_miner_spool_evict_oldest | f0786be |
| PLAN_satellite_deploy_docs | e842d00 |
| PLAN_stable_prefix_production | 382623c |
| PLAN_stable_prefix_tools_first | 382623c |
| PLAN_volume_step | e7aaf93 |
| PLAN_volume_step_guard | 189c971 |
| SCHEMA_LOCK_PLAN.md (outside the glob) | 22c0f93 |

Ask (A12) or active:

| Plan | Status |
| --- | --- |
| PLAN_review_fixes | Active (this series) |
| PLAN_batch4_config_v2 | Active; batch 4 commits are on this branch |
| PLAN_fix11_grounding | Shipped (`97ae2e8`, `4528c98`), but two residual cases open (section 2.1). Keep active until resolved |
| PLAN_batch5_hygiene (this file) | Active |
| docs/CODE_IMPROVEMENT_PLAN.md | Status line says "proposed; no implementation is authorized" (line 3). Its contents are stale: voluptuous_openapi is named as the source conversion (lines 27, 83), and it says pyproject declares `>=2024.12.0` (line 37), but `pyproject.toml:11` declares `>=2026.8.3`. Shipped parts: 7bf0fc0 and 22c0f93. Ask: supersede with ADRs and delete, or keep |
| SAYSO_WAKE_WORD_TRAINING_PLAN.md (outside the glob) | Not in the batch 5 brief. Leave it alone |

### 4.2 ADR format and cluster

- Location: `docs/adr/NNNN-slug.md`. The directory does not exist today.
- Format: Status (Accepted), Context, Decision, Consequences, Shipped in (commit list).
- One ADR per decision cluster, not one per plan. Proposed clusters: compiled schema and
  validation (7bf0fc0, 22c0f93); post-model path and evals (fix13, fix10, fix11, eval
  commits); grammar decoding (fix10); ducking (three plans); wake pipeline and TV
  ensemble (WAKE_* plans, PLAN_WAKE_HOP_DROPS); satellite deploy and LVA pin
  (PLAN_lva_1_1_18, PLAN_satellite_deploy_docs); prompt prefix cache (embedded prefix
  plus stable prefix plans); volume and area handling (volume plans, AREA_ARG_DROP);
  CI and agent rules (ci_ruff_pytest, agents-md-and-comment-removal). Confirm the
  cluster list before writing.

### 4.3 AGENTS.md pointer (A9)

`AGENTS.md:63` says to save a plan under `docs/PLAN_<feature>.md`. New text: decisions go
to `docs/adr/`; `docs/PLAN_*.md` is only for active work. The rest of the Workflow
section (review the plan before implementing) stays.

### 4.4 Compat latest row

- Current: `scripts/compat_matrix.py:15` `CURRENT_HA_VERSION = "2026.8.3"`; `:17`
  `DECLARED_MINIMUM_HA_VERSION`; `MATRIX` at `:49-56` has one entry;
  `COMPONENT_REQUIREMENT_ROOTS` at `:58-60`.
- Latest version: the HA snapshot in `~/.cache/sayso-venvs/ha-2026.11` reports
  `2026.11.1` (`homeassistant/const.py:24-26`). Its required Python is 3.14.2
  (`const.py:29`), which matches the `python="3.14"` entries.
- Change: add `MatrixEntry(id="latest", label="Home Assistant latest release",
  homeassistant="2026.11.1", python="3.14")` to `MATRIX`, and a `"latest"` key in
  `COMPONENT_REQUIREMENT_ROOTS` with the same roots as `"current"`.
- Tests (A2): `tests/test_compat.py:43` asserts `len(MATRIX) == 1`, and `:44-46` asserts
  the current version. Change them to expect `{"current", "latest"}`. Add one assertion
  that `latest` is not older than `current`. `test_matrix_python_version` (`:88-91`)
  reads `get_entry("current")` and is unaffected.
- CI (A3, blocked): `.github/workflows/ci.yml` `include` lists at lines 127-129 (compat job)
  and 159-161 (live-llama job) need a `latest` row. Until that edit lands, the local
  `pytest tests evals/tests` step still runs `tests/test_compat.py` (it is in `tests/`),
  so the matrix metadata is checked in CI, but the venv for `latest` is never built.
- Caveat to note in the PR: the `latest` row is not an installed venv in CI until A3 is
  approved.

## 5. Fix 17: OTel GenAI trace names and optional export

### 5.1 Current span metadata and where it is written

| Key today | Written at | Proposed key |
| --- | --- | --- |
| `model` | `conversation.py:241-245` (INFERENCE open) | `gen_ai.request.model` |
| `correction` | `conversation.py:241-245` | `sayso.correction` |
| `tool_calls` (count) | `conversation.py:260` | `sayso.tool_calls` (no standard attribute for a count) |
| `prompt_tokens` | `conversation.py:261-262` (from `core/completion.py:215`) | `gen_ai.usage.input_tokens` |
| `reused_tokens` | `conversation.py:265-266` (from `inference.py:395`, `core/completion.py:217`) | `sayso.reused_tokens` |
| `grammar`, `grammar_skip_reason` | `conversation.py:698-702` (`_turn_grammar`) | `sayso.grammar`, `sayso.grammar_skip_reason` |
| `tool` | `boundary.py:186` (`action_metadata`) | `gen_ai.tool.name` |
| `batch` | `boundary.py:188` | `sayso.batch` |
| `domain` | `boundary.py:193` | `sayso.domain` |
| `target` | `boundary.py:196` (entity name) | `sayso.target` |
| `calls` | `post_model.py:221` (TOOL_PARSE open) | `sayso.calls` |
| `area_dropped` | `post_model.py:270` | `sayso.area_dropped` |
| `clarify` | `post_model.py:282` | `sayso.clarify` |
| `vad`, `text_length` | `tracing.py:389,396` (`replay_pipeline_events`) | `sayso.vad`, `sayso.text_length` |
| `stream` | `tracing.py:442` (`observe_pipeline_completion`) | `sayso.stream` |

Operation names, as span metadata: `gen_ai.operation.name` is `chat` on the INFERENCE
open (`conversation.py:241`), and `execute_tool` on the HA_ACTION open (`conversation.py:569`).
The review's `tool.call.id` (`gen_ai.tool.call.id`) has no recording site today. `ToolCall`
carries an `id` (`core/completion.py`), so adding it is new data. It is optional and
needs a yes.

Not renamed: stage names (`Stage` values in `core/trace_stages.py`, and the `*_started`
and `*_completed` event names built in `tracing.py:169,181`), and the summary fields
(`tracing.py:247-271`: `wake_ms`, `model_ms`, `tool`, `target`, and so on). They are not
span metadata. The review's Fix 17 is about span metadata.

### 5.2 Consumers

- `sayso.get_trace` (`__init__.py:159-188`, service schema at `:99`, `services.yaml:1-8`)
  returns stored records raw. Renamed keys show up for new traces only.
- `sayso.list_traces` and `diagnostics.py:152-157` read summaries (`trace_store.query`).
  They do not read span metadata, so they are not affected (grep).
- `trace_store.py:86-96` (`async_record`) stores `trace.stage_records()` as-is. `STORAGE_VERSION = 1`
  (`trace_store.py:31`).
- Tests reading span metadata (A4): `tests/test_grammar_wiring.py:44` (helper),
  `:67` (`model`), `:85` (`grammar`), `:132-133`; `tests/test_grammar_fallback.py:278-279`.
  `tests/test_tracing.py:908-925` reads `summary` and stage names only, so it is unaffected.
- `satellite/sayso/tracing.py` has its own trace module. It is not in this change. Flag only.

### 5.3 Backward compatibility

- Recommended: no storage migration. Old records keep old names. New records carry
  `"sayso.trace_schema": 2` in their metadata, or a top-level `"schema": 2` in the summary
  (an additive field). `get_trace` returns both as stored. Readers that need both
  name sets handle both.
- Alternative: bump `STORAGE_VERSION` to 2 and rename on load. More code and a risk to
  retention. Not recommended.

### 5.4 Export (off by default)

- Config: an options-flow select `trace_export` with values `off` (default), `jsonl`,
  `otlp_http`, plus `trace_export_endpoint` for OTLP. Trace options already live in
  `config_flow.py:129-137`. New labels go in `strings.json` and `translations/en.json`.
- Redaction stays where it is: the utterance is cleared before storage
  (`trace_store.py:89-90`) and export reads the same record, so export never sees more
  than storage does.
- JSONL: append one line per record from `TraceStore.async_record` via
  `hass.async_add_executor_job`, to `hass.config.path("sayso", "traces.jsonl")`.
  Unbounded growth is a concern (A13). Options are a size cap or rotation.
- OTLP/HTTP JSON: `POST {endpoint}/v1/traces` with a `resourceSpans` body, using
  `async_get_clientsession(hass)` (aiohttp, already a Home Assistant dependency; no new
  package). Two conversions are needed:
  - OTLP trace ids are 16 bytes, hex-encoded (32 characters). SaySo trace ids are HA
    `PipelineRun` ULIDs (26 Crockford base32 characters). Convert the 128 bits to hex.
  - Span ids are 8 bytes (16 hex). Derive them from `(trace_id, sequence)`.
- Failures: log a warning, never block the voice path, no retry queue (A13).
- Trust boundary: OTLP sends data off the box. `ARCHITECTURE.md:239-241` says traces
  never contain prompts, audio, schemas, or state dumps, and that is still true for
  exported fields. The new outbound communication still needs an ARCHITECTURE update
  (A9), per CLAUDE.md.
- No OTel SDK. `pyproject.toml` dependencies do not change for export.

### 5.5 Tests

- Existing (renamed, A4): `tests/test_grammar_wiring.py`, `tests/test_grammar_fallback.py`.
- New, `tests/test_trace_export.py`: export is off by default (no file, no HTTP call);
  one JSONL line per record; utterance absent when `trace_store_utterances` is off; OTLP
  payload shape (trace id is 32 hex, span ids 16 hex); a failing endpoint logs and does not raise.
- New, `tests/test_trace_names.py`: every key in section 5.1's table, for each span.

## 6. Slices (ordered)

Each slice is one or more commits. Each runs the CI gate in section 7 before the
next slice. "Approval" marks the existing-test or out-of-scope edits.

| Slice | Scope | Files | Verify | Approval |
| --- | --- | --- | --- | --- |
| S0 | Realistic stub fix (already approved) | `tests/test_realistic_v3.py:218-228` (2 lines) | `pytest tests/test_realistic_v3.py -k perfect_run` expects 528 and the line-230 failure only | A1 is approved. Decide the two grounding cases separately |
| S1 | Fix 9 steps 2-3 | `core/vocabulary.py`, `core/post_model.py` (`:196`, `:314`, `TurnInput`), `core/grounding.py` (`:15-51`, `:488-509`), `core/held.py` (`:8-19`), `core/area_context.py:44`, `conversation.py` (speech resolution, `:188-800`), `const.py` (re-exports), `translations/en.json`, `strings.json`, new `tests/test_speech_translations.py` | smoke 16/24, promotion 441/530; `test_supported_languages.py`; CI gate | A5 if `DOMAIN_WORDS` moves |
| S2 | Fix 16 docs and compat row | `docs/adr/*.md` (new), deletion of executed plans, `AGENTS.md:63`, `scripts/compat_matrix.py:49-60`, `tests/test_compat.py:42-46`, `.github/workflows/ci.yml` (blocked) | `pytest tests/test_compat.py`; `python scripts/compat_matrix.py matrix-json` | A2, A3, A7, A9, A12 |
| S3 | Fix 15 lint: B,ASYNC then UP,SIM,RUF | `pyproject.toml:43-51`, `custom_components`, `evals`, `scripts`, `satellite` (if A10), `tests` (A6) | `ruff check` with the new rules; CI gate | A6, A10, A11 |
| S4 | Fix 15 format, one commit | `pyproject.toml:53-64` and the reformatted files | `ruff format --check .`; CI gate | A6, A10 |
| S5 | Fix 15 typing | `pyproject.toml` (`[tool.ty]` or mypy config, test extra), new `tests/test_typing_contracts.py`, core and integration fixes | `pytest tests/test_typing_contracts.py`; CI gate | A8 |
| S6 | Fix 17 names | `tracing.py`, `conversation.py:241-266,569,698-702`, `boundary.py:186-196`, `post_model.py:221,270,282`, `tests/test_grammar_wiring.py`, `tests/test_grammar_fallback.py` | `pytest tests/test_grammar_wiring.py tests/test_grammar_fallback.py tests/test_tracing.py`; CI gate | A4 |
| S7 | Fix 17 export (off by default) | `config_flow.py`, `trace_store.py`, `const.py`, `strings.json`, `translations/en.json`, `ARCHITECTURE.md`, new `tests/test_trace_export.py` | export tests; CI gate | A9, A13 |

Order rationale: S1 first, so the typing test in S5 types the new signatures. S3 and S4
are separate so the formatting commit has no lint changes in it. S6 before S7 so export
uses the new names.

## 7. CI gate

Run from the worktree with `.venv/bin` first on `PATH`:

```bash
export PATH=$PWD/.venv/bin:$PATH
ruff check .
ruff format --check .
pytest tests evals/tests --ignore=tests/test_realistic_v3.py
~/.cache/sayso-venvs/satellite/bin/pytest -q satellite/sayso
lint-imports
python -m evals.generate_realistic --check
```

Plus, per slice: `pytest tests/test_realistic_v3.py -k perfect_run` (expected to fail
until the two grounding cases are resolved, section 2.1) and the stub-eval parity check
(smoke 16/24, promotion 441/530) for S1.

## 8. Risks and open points

- The realistic perfect-run test cannot reach 530/530 until the two grounding cases in
  section 2.1 are fixed or their data is decided. The test file is excluded from CI, so
  CI is not blocked by it.
- `ruff check --select B,ASYNC .` (root) reports 5, explicit paths report 11 (section
  3.1). The counts in this plan use explicit paths. The cause is open and should be
  found in S3 before the rule counts are trusted.
- `ruff format` excludes for `tests/` and removing the satellite extend-exclude both
  change files under CLAUDE.md's "tests/" rule or a separate CI job (A6, A10).
- `ty` is pre-1.0 (0.0.107). Pin exactly in the test extra (A8).
- The OTLP export is an outbound path. A failed export must never delay or fail a voice
  turn (section 5.4).
- Stub-eval numbers in section 0 come from an in-memory run. No `evals/results/` file
  was written. The batch 3 record matches.
- No commits, no push, no branch operations beyond the creation of `ajax/review-batch5`.
