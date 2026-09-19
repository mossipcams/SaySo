# Plan: parse tool-call arguments for embedded chat templates

## Problem

Embedded inference on HAOS (`conversation.sayso_local` at 192.168.1.35) speaks
"The local model is unavailable" after a successful first generate. The GGUF's
LFM2.5-Base chat template calls `function.arguments.items()` and requires a
dict. SaySo sends OpenAI JSON-string arguments. Correction, follow-up, and
later turns in the same conversation crash Jinja with
`'str object' has no attribute 'items'`.

Live trace `01M2XASAZAAA06ZAVA25QER041`: first inference OK (1 tool call,
11727 ms), tool parse fails, correction inference dies in 1 ms.

## Scope

- Convert OpenAI-string `function.arguments` to dicts **only** on the embedded
  path, immediately before `create_chat_completion`.
- Leave the canonical transcript / HTTP envelope as JSON strings.
- Do not change external llama-server behavior.

## Files

- `custom_components/sayso/inference.py` — normalize messages in
  `EmbeddedEngine._complete` (or a tiny helper next to it).
- `tests/test_inference.py` — prove string args become dicts for embedded
  `create_chat_completion`, dicts pass through, and the HTTP engine is
  unchanged.

## Approach

Parse `tool_calls[].function.arguments` when it is a JSON object string.
Leave already-dict arguments alone. Do not invent a new message format.

Invalid JSON stays a string so the existing invalid-response path still
fail-closes rather than guessing.

## Verification

- `python -m pytest tests/test_inference.py -q`
- Existing conversation tests still pass if the helper is only used by
  `EmbeddedEngine`.
