# Plan: architecture review fixes (Nov 2026)

Source: the Nov 19, 2026 review of `main` at #207. Items 1-4 were verified
against the tree before this plan was written. Items 5-18 were verified afterwards and their
solutions are folded in below.

## Decisions (user, Nov 19, 2026)

- D1: move the colocated tests into `tests/`. Approved to edit `tests/` for
  Fix 1, Fix 16 and Fix 18.
- D2: batch the fixes into PRs (batches listed under Order).
- D3: tooling installed: `.venv` (Python 3.14, HA 2026.8.3, ruff 0.17.6,
  same as CI) and `~/.cache/sayso-venvs/ha-2026.11` (HA 2026.11.1).
  Baseline on `.venv`: CI pytest line 2099 passed, 1 skipped; the 18
  colocated files 244 passed; ruff check and format clean.
- D4: no hassfest or HACS validation workflows. The gh token has no
  `workflow` scope, so any later change under `.github/workflows/`
  (Fix 14's import-linter step, Fix 15's type-check step, Fix 16's compat
  row) needs the user's OK first.

## Fix 1: colocated tests never run in CI

Verified: 18 `custom_components/sayso/test_*.py` files; `testpaths` is
`["tests", "evals/tests", "satellite/sayso"]`; CI runs
`pytest tests evals/tests --ignore=tests/test_realistic_v3.py`.

Solution (D1 = move):
1. Run the 18 files in place first (`pytest custom_components/sayso`) and
   record which fail. Failures are implementation or stale-test findings to
   report, not tests to delete or weaken.
2. `git mv custom_components/sayso/test_*.py tests/`, resolving any name
   clashes with existing `tests/` files by prefixing `test_integration_`.
3. Fix any relative or sibling imports the move breaks.
4. Update AGENTS.md so it no longer lists colocated tests as a location.

Solution (D1 = testpaths only): add `custom_components/sayso` to `testpaths`
and to the CI pytest line. The tests still ship with HACS installs.

Files: `custom_components/sayso/test_*.py`, `tests/`, `AGENTS.md`
(or `pyproject.toml`, `.github/workflows/ci.yml`).

Verify: `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`
collects the moved tests (count rises by the moved test count); `ruff check .`;
`ruff format --check .`.

## Fix 2: `ToolResult.error` ignored

Verified: `boundary.py:170` `tool_result_data` returns `content.result.data`
and drops `.error`; `conversation.py:274` decides failure only from
`is_tool_execution_failure(data)`. `transcript.py:55` only needs the data.

Solution: add `tool_result_failed(content) -> bool` in `boundary.py`
returning `content.result.error` when `content` has `result`, else
`is_tool_execution_failure(content.tool_result)`. Use it at
`conversation.py:274`. When the flag is set and `data["error"]` is not a
string, fall back to the existing generic failure speech. No comments.

Files: `custom_components/sayso/boundary.py`,
`custom_components/sayso/conversation.py`, one test next to the existing
boundary/conversation tests.

Verify: new test where a tool returns `ToolResult(data={"ok": 1},
error=True)` and the turn reports failure, not "Done."; existing
`tool_result_data` tests pass; full CI pytest line.

## Fix 3: llama-cpp-python unpinned

Verified: `const.py` `LLAMA_CPP_MIN_VERSION = "0.3.33"`; `model_store.py:42`
and `:50` use `>=`; `CheckpointingLlama` reads private internals.

Solution: rename to `LLAMA_CPP_VERSION`, set it to the version the
checkpoint code was last verified against (confirm from the compat matrix
or the deployed install), and use `==` in both `model_store.py` sites. Add a
regex custom manager in `renovate.json` that tracks the constant in
`const.py`, so bumps open PRs that run the compat matrix. Release note:
existing installs with a newer wheel reinstall the pinned one at next setup.

Ships together with Fix 7a (the `UV_EXTRA_INDEX_URL` race), same function.

Files: `custom_components/sayso/const.py`,
`custom_components/sayso/model_store.py`, `renovate.json`, the model_store
test.

Verify: model_store tests cover `==` install and the is-installed check;
CI pytest line.

## Fix 4: manifest metadata

Verified: `documentation`/`issue_tracker` point at `github.com/sayso/sayso`;
`codeowners` is `[]`.

Solution: set `documentation` to `https://github.com/mossipcams/SaySo`,
`issue_tracker` to `https://github.com/mossipcams/SaySo/issues`,
`codeowners` to `["@mossipcams"]`. No hassfest or HACS workflow (user
decision D4).

Files: `custom_components/sayso/manifest.json`.

Verify: `python -m json.tool` on the manifest; CI pytest line.

## Items 5-18

Evidence: `evals/results/review_verify_5_11.md` and
`evals/results/review_verify_12_18.md` (gitignored; local delegate,
spot-checked). Verdicts: 5-10, 13, 14, 16-18 confirmed; 11 partly; 12 and 15
partly because the delegate environment has no `homeassistant` or `ruff`.

### Group A: small, independent

**Fix 9: `supported_languages` claims every language.** Verified:
`conversation.py:393-394` returns `MATCH_ALL`, while spoken errors
(`const.py:81-95`), grounding (`grounding.py:15,17-38`) and held-call answers
(`conversation.py:495-500`) are English-only. So HA sends non-English
pipelines into an English-only path. Solution:
1. Return `["en"]` now, so the declared support matches the real support.
2. Move every spoken constant (`ERROR_*`, the clarify templates, "Done.")
   into `translations/en.json` under a `speech` section. Resolve them
   with `translation.async_get_translations(hass, language, "speech",
   [DOMAIN])` using the conversation's language, with English as the
   fallback.
3. Put the grounding vocabulary (`GENERIC`, `NOUN`, `YES_ANSWER`,
   `NO_ANSWER`, `FILLER`) in a table keyed by language. Widen
   `supported_languages` only for languages that have both speech strings
   and a vocabulary table.

Files: `conversation.py`, `const.py`, `grounding.py`, `translations/en.json`,
`strings.json`, and tests. Verify: a test asserts `["en"]`; a test
resolves a spoken error through translations; full CI pytest line.

**Fix 6: option rewrites belong in a migration.** Verified:
`__init__.py:262-267` calls `_synced_embedded_options` and
`_migrated_embedded_options` on every setup. The package has no `VERSION`,
`MINOR_VERSION` or `async_migrate_entry`. Only `_migrated_embedded_options`
(legacy `CONF_MODEL_PATH` -> `CONF_MODEL`) is a one-time migration.
`_synced_embedded_options` re-points an entry whose model is no longer in
the shipped `EMBEDDED_MODELS` catalog, and that can happen after any
release (#47307e1), so it stays a setup-time check. Solution:
1. Set `MINOR_VERSION = 2` on the config flow.
2. Add `async_migrate_entry`, which runs `_migrated_embedded_options` once
   for `version == 1 and minor_version < 2` and then sets
   `minor_version=2`. Return `False` for `version > 1`.
3. Remove the legacy rewrite from `async_setup_entry`. Keep the catalog
   sync.

Files: `__init__.py`, `config_flow.py`, and a migration test. Verify: the
test migrates a 1.1 embedded entry with `CONF_MODEL_PATH` and checks the
options and `minor_version`; a 1.2 entry is untouched; a future version 2
entry fails migration.

**Fix 7a: process-wide env mutation.** Verified:
`model_store.py:46-57` sets `os.environ["UV_EXTRA_INDEX_URL"]` from an
executor thread, which leaks into any concurrent HA package install.
Solution: pass the index to the installer directly, not through
`os.environ`. First check whether `pkg_util.install_package` in HA
2026.8.3+ takes an index or env argument. If it doesn't, run `uv pip
install` / `pip install` through `subprocess.run(..., env={**os.environ,
...})`. Combine with Fix 3, since both touch `_install_llama_cpp`. Files:
`model_store.py` and its test.

**Fix 18: utterances stored by default.** Verified:
`const.py:76` `DEFAULT_TRACE_STORE_UTTERANCES = True`, with 30-day
retention and 500 records, and a user-facing toggle in the options flow.
Solution: default it to `False` for new entries. Existing entries keep
their stored value. One constant plus the test that asserts the default
(that test lives in `tests/` or is colocated, so it needs approval if
it's under `tests/`).

**Fix 16: stale docs and compat range.** Verified: 31 `docs/PLAN_*.md`
files; `CODE_IMPROVEMENT_PLAN.md:27,37-38,83` describe `voluptuous_openapi`
and an HA 2024.12 baseline. `compat_matrix.py:15` sets `CURRENT_HA_VERSION =
"2026.8.3"`, and `tests/test_compat.py:45-46` asserts it equals the declared
minimum. Meanwhile evals run against the 2026.11.1 snapshot
(`evals/runner.py:37`). So the matrix tests only the minimum, never the
latest release. Solution:
1. For each executed PLAN (including `CODE_IMPROVEMENT_PLAN.md`), write
   the decision that landed as a short ADR in `docs/adr/NNNN-<slug>.md`
   (context, decision, consequences, and the PR it shipped in). Then
   delete the plan. Plans still in progress stay. Point AGENTS.md at
   `docs/adr/` for decisions and at `docs/PLAN_*` only for active work.
2. Add a `latest` row (2026.11.1) to the compat matrix next to the minimum.
   This changes `tests/test_compat.py`, so it needs approval.

### Group B: platform patterns (one "config entry 2" change)

These all change entry structure, so they ship together, after Group A.

**Fix 5: VAD select platform.** Verified: `vad_sensitivity.py:80-93`
calls `async_setup_component("select")`, builds an `EntityPlatform` by
hand, and uses a constant unique id (`:46`) with no device info. Solution:
add a `select.py` platform that sets `unique_id = f"{entry_id}-vad-sensitivity"`
and the entry's `device_info` (the same as `conversation.py:380-385`).
Keep `patch_satellite_entities`, and remove the manual platform and the
teardown. The old entity is orphaned. Remove it from the registry in the
Fix 6 migration (bump to 1.3).

**Fix 8: one engine per process.** Verified: one global ducker in
`hass.data` (`ducking.py:300-310`, so the last entry's duck volume wins),
and a new `Llama` for each entry (`__init__.py:246-251`,
`inference.py:306-312`). Solution: one entry with config subentries.
1. The main entry owns the connection (embedded model or llama.cpp URL),
   the single engine, ducking (duck volume becomes an entry option, so
   there's no "last one wins"), and the VAD select.
2. Each conversation agent is a `conversation` subentry with its own
   prompt, LLM API and temperature, handled by a
   `ConfigSubentryFlow` registered through `async_get_supported_subentry_types`.
   `conversation.py` then creates one entity per subentry, using
   `config_subentry_id`.
3. Add an `ai_task` subentry type (an `ai_task.py` platform) on the same
   engine.
4. Migration (the next minor version after Fix 6): fold the existing
   entries into one main entry plus one subentry per old agent. Keep entity
   unique ids so automations and pipelines that reference
   `conversation.sayso_*` don't break. Remove the merged entries.
5. Remove the global `DUCKER_KEY` and `ENTRY_COUNT_KEY` ref counting from
   `ducking.py`, and store runtime state on `entry.runtime_data`.

Files: `__init__.py`, `config_flow.py`, `conversation.py`, `ducking.py`,
`select.py`, new `ai_task.py`, `strings.json`, plus migration and
subentry-flow tests.

**Fix 7b: downloads and permanent failures.** Verified: wheel and GGUF
downloads run in `async_setup_entry` (`__init__.py:229,233`), and
`ConfigEntryNotReady` (`:243-244,255-257`) retries forever, even for a
missing wheel or a hash mismatch. Solution:
1. Move the wheel install and the GGUF download into the config flow as
   an `async_show_progress` step (`progress_action` "download"). It shows
   progress, verifies the hash, and finishes with `async_show_progress_done`.
   Setup then only loads files that are already present.
2. In `async_setup_entry`, a missing or corrupt file (deleted model, hash
   mismatch, `model_store.py:107-112`) or a missing wheel for the platform
   (`model_store.py:71-76`) creates an `issue_registry` Repairs issue with a
   fix flow that re-runs the download, and setup returns `False`.
   `ConfigEntryNotReady` stays only for transient failures (llama.cpp
   server unreachable).
3. Options-flow model changes re-enter the same progress step.

Files: `config_flow.py`, `__init__.py`, `model_store.py`, new `repairs.py`,
`strings.json`, and flow and repairs tests.

### Group C: measurement first, then model boundary

**Fix 13: evals bypass the post-model path.** Verified:
`evals/runner.py:336-342` goes straight from complete to score. The scorer
matches cues by regex (`scorer.py:18-20`). The post-model path is
`_async_answer` (`conversation.py:431`, held replay `488-519`) and
`_async_run_tool_calls` (`:536`: correction `583-621`, area drop `631`,
grounding/clarify `642-654`, "Done." `683-684`). It is already split
behind two seams, `_Turn.complete` (`:214`) and `_Turn.execute` (`:248`).
Solution:
1. Lift the body of `_async_run_tool_calls` and the held replay into a
   function that takes `complete` and `execute` callables and has no HA
   imports.
2. `conversation.py` calls it with the real seams.
3. `evals/runner.py` calls it with the eval adapter as `complete` and a
   fake executor that returns a success `ToolResult`, then scores the
   final spoken text and the executed calls instead of the raw completion.

This depends on Fix 14, so that the runner can import the function
without HA. Fix 1 and Fix 2 should land first so the moved tests guard
the refactor. This is the largest item and should be planned on its own.

**Fix 14: real HA-free package.** Verified: `sayso_contract.py:9-21`
fakes a `_sayso_pure` package in `sys.modules`. Seven modules are
transitively HA-free (`area_context`, `completion`, `exceptions`,
`history`, `lfm_parse`, `tool_contract`, `tool_schema`). `grounding.py`
is held back only by `const.py:3` importing `homeassistant.const`.
Import-linter isn't configured. Solution:
1. Move the seven modules (and `grounding.py`, once its three `CONF_*`
   imports come from a plain module) into `custom_components/sayso/core/`.
2. Point the integration and `sayso_contract.py` at it, then delete the
   shim.
3. Add an import-linter forbidden contract (`core` must not import
   `homeassistant`), plus a CI step.

Use ast-grep for the import rewrite. This lands before Fix 13.

**Fix 11: reuse HA's matcher in grounding.** Partly verified: grounding
(308 lines) and held replay (about 70 lines) are an English-token
resolver, but `volume_step.py:41-55` already calls
`intent.async_match_targets` with constraints and preferences.
`async_get_match_preferences` isn't used. Solution:
1. Reuse the `volume_step.py` pattern to resolve the model's name or area
   against HA, with preferences from the satellite device.
2. A unique match executes. Several matches clarify using HA's candidate
   names, and held replay stores entity ids, not text.
3. Keep `is_grounded` as the narrow "did the user say it" check.

Measure the change through the Fix 13 evals before and after. It needs
HA at runtime, so it stays outside `core/`.

**Fix 10: grammar-constrained decoding.** Verified: no grammar on either
path (`inference.py:351-357`, `client.py:50-66`), and the fingerprint
already exists (`schema.py:373-376`). Solution:
1. Add `core/grammar.py`, which builds GBNF from the compiled tools as
   `root ::= call | text`, typed per tool, cached by fingerprint.
2. Thread an optional `grammar` through `async_chat_completion`:
   `LlamaGrammar.from_string` for the embedded engine, and the `grammar`
   field for llama-server.
3. Default it off behind an option.

Adopt it only if the Fix 13 evals show accuracy at least equal and
acceptable p50 latency on v11.

**Fix 12: tool annotations.** Verified on HA 2026.11.1: `llm.Tool` has
`annotations: ToolAnnotations` (defaults: `read_only=False`,
`destructive=True`, `idempotent=False`, `open_world=True`), and
`components/intent/llm.py` sets a mapping for each intent
(`INTENT_ANNOTATIONS`: `_CONTROL`, `_REPEATS`, `_ADDS`, `_READ_ONLY`).
`APIInstance` copies them through namespacing. So the review's claim that
there's no per-intent mapping is wrong for 2026.11. HA 2026.8.3 (the
declared minimum) has no `ToolAnnotations`, and SaySo reads none.
Solution:
1. Add `tool_is_read_only(tool)` in `boundary.py` returning
   `getattr(getattr(tool, "annotations", None), "read_only", None)`. On
   `None` (HA older than 2026.11), fall back to a read-only set in
   `tool_contract.py` that mirrors HA's `_READ_ONLY` intents
   (`HassGetState`, `GetLiveContext`, `HassGetCurrentDate`,
   `HassGetCurrentTime`, `HassGetWeather`, and the list/timer status
   intents).
2. In the Fix 11 grounding path, skip grounding and held-call clarify for
   read-only tools.
3. Don't use `destructive` to require confirmation. HA marks every control
   intent (`_CONTROL`, `_REPEATS`) as destructive by default, so that
   would confirm every "turn off". Keep confirmation driven by the
   existing grounding outcome.
4. Record the annotation table in `docs/HA_2026_11_TOOLS.md`. When the
   minimum HA reaches 2026.11, delete the fallback set.

Files: `boundary.py`, `tool_contract.py`, `conversation.py`,
`docs/HA_2026_11_TOOLS.md`, tests.

### Group D: lint and traces

**Fix 15: ruff and typing.** Verified config: `select = ["E9", "F63",
"F7", "F82"]`, format excludes `custom_components/**` and more, and there's
no type checker. Measured with ruff 0.17.6 on `custom_components evals`:
`--select B,UP,SIM,ASYNC,RUF` finds 21 findings (RUF012 ×5, RUF001 ×4,
B009 ×2, and one each of ASYNC109, SIM102, SIM103, SIM105, SIM110, SIM300,
B007, B905, RUF046, RUF100), 4 of them autofixable. Solution:
1. Also measure `satellite`, `scripts` and `tests` once they're
   un-excluded.
2. Enable `ASYNC` and `B` first (the bug-finding rules). Autofix in one
   commit, and hand-fix the rest.
3. Add `UP`, `SIM` and `RUF`, with `target-version` set to HA's minimum
   Python.
4. Remove the `format` excludes for `custom_components/**`, `evals` and
   `tests` (one formatting-only commit).
5. Add strict mypy (or ty) for `custom_components/sayso/core/` after Fix
   14, and non-strict checking for the rest of the integration. Add both
   to CI.

**Fix 17: trace field names.** Verified: custom names
(`tracing.py:23-40,79-91,276-303`, metadata `prompt_tokens`,
`reused_tokens`, `tool_calls`), no `gen_ai`, no export. Solution:
1. Rename the span metadata to the OTel GenAI names:
   - `gen_ai.operation.name` (`chat` / `execute_tool`)
   - `gen_ai.request.model`
   - `gen_ai.usage.input_tokens`, from `prompt_tokens`
   - `gen_ai.tool.name`
   - `gen_ai.tool.call.id`
   Keep SaySo-only fields (`reused_tokens`, `area_dropped`, `clarify`)
   under `sayso.*`. Update `sayso.get_trace`, diagnostics, and their tests
   in the same change.
2. Add an optional export (off by default): an options-flow setting that
   writes one JSON line per finished trace to a file under the HA config
   dir, or sends it to an OTLP/HTTP endpoint using a plain `aiohttp` JSON
   POST. Don't add an OTel SDK dependency.

Files: `tracing.py`, `trace_store.py`, `diagnostics.py`, `config_flow.py`,
`const.py`, `strings.json`, tests.

## Order (PR batches)

Each batch is one PR. Inside a batch, every fix is its own delegated
`EXECUTION` (at most 4 files per QWEN slice) and its own commit. The batch
runs the CI gate before the PR. Batch 1 goes out from `ajax/codebase`; each
later batch gets its own branch off `main` (or stacks on the previous batch
if that one hasn't merged yet).

1. **Batch 1, quick fixes:** Fix 4, Fix 2, Fix 3 with Fix 7a, Fix 9 step 1
   (`["en"]`), Fix 6, Fix 18, Fix 1, plus this plan.
2. **Batch 2, measure the real path:** Fix 14, then Fix 13.
3. **Batch 3, grounding and decoding:** Fix 11 with Fix 12, then Fix 10,
   each judged on the Fix 13 evals.
4. **Batch 4, config entry v2:** Fix 5, Fix 8 (subentries, migration,
   AI Task), Fix 7b (progress flow and Repairs).
5. **Batch 5, hygiene:** Fix 9 steps 2-3 (translations), Fix 16 (ADRs and
   compat row), Fix 15 (lint and typing), Fix 17 (trace names and export).

CI gate (run with `.venv/bin` first on `PATH`): `ruff check .`,
`ruff format --check .`,
`pytest tests evals/tests --ignore=tests/test_realistic_v3.py`,
`pytest -q satellite/sayso`.
