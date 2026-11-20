# Plan: batch 6, code review findings

User request (2026-11-20): "Fix those and batch them". This covers all nine verified findings from the code review of `main...ajax/review-batch5`. The work goes on a new branch `ajax/review-batch6`, created from `ajax/review-batch5`, with a PR stacked on #212.

## Findings and fixes

| # | Finding (verified) | Fix | Files |
| --- | --- | --- | --- |
| F1 | Confirming a held clarification replays the whole held batch with grounding off (`core/post_model.py:227`), so later calls that were never grounded run anyway | Grounding is skipped only for the confirmed call (`held.index`). Every other call in the replayed batch goes through grounding as usual. If one of them is ungrounded, the turn clarifies again and nothing runs (fail closed) | `core/post_model.py`, `core/held.py` if needed, new `tests/test_held_replay_grounding.py` |
| F2 | AI Task leaks raw `SaySoError`s, and a structured JSON-array answer is parsed as an LFM tool call (`ai_task.py:75`, `inference.py:291`) | Both engines extract content tool calls only when tools were offered on that request. `ai_task.py` maps `SaySoError` to `HomeAssistantError` with a translation key (`strings.json` and `translations/en.json`) | `inference.py`, `ai_task.py`, `strings.json`, `translations/en.json`, new `tests/test_ai_task_errors.py` |
| F3 | The LFM parser rejects decimals (`temperature=21.5`), and the grammar maps `number` to `v-int` | `lfm_parse` accepts decimal literals (optional sign, digits, `.`, digits) and returns `float`; integers stay `int`. The grammar adds a `v-num` rule for `number` and keeps `v-int` for `integer`. The bare-minus fix from `271b93a` stays | `core/lfm_parse.py`, `core/grammar.py`, new `tests/test_lfm_parse_decimals.py` |
| F4 | External grammar plus tools probably gets rejected on every turn. That is unconfirmed, because the llama-server source wasn't reachable. `_rejects_grammar` also treats any non-auth 4xx as a grammar rejection | Retry without the grammar only on HTTP 400. Once the backend rejects a grammar, remember it on the client and skip the grammar up front on later turns, with `grammar_skip_reason` = `backend_rejected`, so no turn pays a second request. This works whether or not llama-server refuses grammar with tools | `client.py`, `tests/test_grammar_fallback.py` (Batch 3 file, ours) |
| F5 | OTLP export emits a zero-length `*_started` span and a separate `*_completed` span for each stage | Pair the started and completed events of a stage into one span named after the stage, from the started timestamp to the completed timestamp. Interaction start and end stay on the root span | `trace_export.py`, `tests/test_trace_export.py` (Batch 5 file, ours) |
| F6 | The embedded engine silently drops tools when the system prompt has no `\nArea context:\n` block, or there is no system message (`core/area_context.py:149`) | `EmbeddedEngine` passes `tools=list(tools)` and `tool_choice='auto'` to `create_chat_completion` when `tools_before_area_messages` returns the messages unchanged (no area block, or no system message). `core/area_context.py` is unchanged. The production layout, which has the area block, is byte-identical | `custom_components/sayso/inference.py`, `tests/test_grammar_threading.py` (helper gives the prompt an area block), new `tests/test_tools_without_area_block.py` |
| F7 | `SaySoConversationEntity._held` never evicts, because entries are only popped by the same conversation | Purge entries whose `expires` has passed whenever a held state is stored or looked up | `conversation.py`, new `tests/test_held_eviction.py` |
| F8 | The full GGUF SHA-256 is computed on every setup and reload (`model_store.py:124`) | After a hash passes, write a sidecar `<model>.verified` containing sha256, size and mtime_ns. Setup skips hashing when the sidecar matches the file's current size and mtime and the expected sha256. The download flow writes the sidecar after it verifies. A mismatch rehashes, so a tampered or replaced file is still caught | `model_store.py`, new `tests/test_model_verify_cache.py` |
| F9 | Each turn makes several full `hass.states.async_all()` and `async_should_expose` scans | Compute the exposed states once per turn and pass them to exposed domains, the grounding home, the routing catalog and the per-domain lookups | `conversation.py`, `routing.py`, new `tests/test_single_exposure_scan.py` |

## Rules

- **Fail closed:** keep every fail-closed barrier. F1 tightens one.
- **Code style:** no comments or docstrings (`tests/test_no_comments.py`). mypy stays clean, with core strict.
- **Tests:** existing tests are edited only where a row above names the file (our own Batch 3/5 files). Assertions are never weakened. A delegate stops if another existing test conflicts.
- **Commits:** one commit per finding, by path only. No AI attribution.

## Verification (every slice)

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/pytest tests evals/tests --ignore=tests/test_realistic_v3.py -q
.venv/bin/pytest tests/test_realistic_v3.py -k perfect_run -q
~/.cache/sayso-venvs/satellite/bin/pytest -q satellite/sayso
.venv/bin/python -m mypy custom_components/sayso
.venv/bin/lint-imports
.venv/bin/python -m evals.generate_realistic --check
PYTHONPATH=. .venv/bin/python evals/results/fix13_stub_eval.py b6 (smoke 17/24, promotion 442/530 unless a fix explains the change)
```

Then push `ajax/review-batch6` and open the PR with base `ajax/review-batch5`.

## Risks

- **F3:** a decimal for an `integer` field (`brightness=50.0`) now parses as a float and fails schema validation, which triggers a correction retry. Before this change the parse failed outright.
- **F8:** the sidecar file sits next to the model in the models directory. Removing the model removes the sidecar's meaning, and a stale sidecar can't match a new file's size and mtime.
- **F4:** with a working backend the grammar is still sent. Only a rejection makes the client skip it, for the life of that client.
