# Plan: history question gate

Scope: stop earlier voice commands from reaching the model prompt, except when
SaySo's previous reply was a question (answer to a clarifying question). The
rule lives in a Home Assistant-free contract module, `history.py`, next to
`area_context.py` and `tool_schema.py`, and is exposed through `sayso_contract.py`.

## Why

Home Assistant reuses one conversation_id for satellite commands within 5-10
minutes, so `chat_log.content` accumulates every earlier command, tool call and
tool result. `_Turn.messages` sends all of it, and the model repeats the last
action on acknowledgements ("Thank you.") and loops on unrelated commands.

## Behaviour

`model_history(messages)` runs on the OpenAI-style dicts from
`chat_log_to_messages`, after conversion and before area context:

- Keep every `system` message.
- Keep the last `user` message (current command) and everything after it.
- If the last `assistant` message before it has non-empty string content whose
  stripped value ends with `?`, `;` or `？`, also keep the turn that
  preceded that question: from the last `user` before it (or the assistant
  itself if none) up to the current `user`.
- Otherwise drop earlier turns.
- No `user` message: return messages unchanged.

Same three endings as Home Assistant's `ChatLog.continue_conversation`.

## Files

- `custom_components/sayso/history.py` — new, standard library only, one public
  function `model_history`.
- `custom_components/sayso/transcript.py` — restored to `origin/main`.
- `custom_components/sayso/conversation.py` — `_Turn.messages` becomes
  `model_history(_chat_log_to_messages(...))`; area context unchanged.
- `sayso_contract.py` — exposes `history`.
- `tests/test_history_question_gate.py` — rewritten; imports through
  `sayso_contract`; covers scenarios a-i, one `_Turn.messages` HA-content case,
  and a source check that `history.py` has no `homeassistant` import.
- `ARCHITECTURE.md` — one sentence in the per-request list about earlier turns.

Existing test files and `evals/`, `training/` are not modified.

## Out of scope

- Changing `chat_log_to_messages`, the prompt, routing, or eval cases.
- Commit, push, PR.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_history_question_gate.py
.venv/bin/python -m pytest -q tests evals/tests --ignore=tests/test_realistic_v3.py -p no:cacheprovider
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
git diff origin/main -- custom_components/sayso/transcript.py   # must be empty
git diff origin/main --stat && git status --short
```

Full-suite failures must match `origin/main` (33 locally, HA 2026.9.4 venv).
