# Plan: evals prompt the model with the embedded runtime's tool layout

## Why

Since 1.17.0 (382623c) the embedded engine prompts the model with
`tools_before_area_messages(messages, tools)`: the tool list is written into the
system message before the `Area context:` block and no separate `tools` go to
the chat template. `evals/runner.py:evaluate` still passes `messages` and `tools`
separately, so every adapter renders the tool list after the area block. Evals
therefore score a prompt production no longer sends: 1.17.0 kept 91/120 on
promotion while live v11 started calling `HassTurnOn` for "go tv" and split
0.53/0.46 between unpause and a refusal on "play the tv" (ajax-optimization #33).

## Scope

- `evals/runner.py`, `evaluate`: send the adapter
  `_area_context.tools_before_area_messages(rendered["messages"], rendered["tools"])`
  and an empty tool list when the tools were moved (the function returns the
  messages unchanged when there is no area block; then keep sending the tools).
  Scoring and validation still use `rendered["tools"]`.
- `evals/adapters/endpoint.py`: omit the `tools` key when the list is empty, so
  llama-server renders exactly the inlined prompt.
- No change to `render_case`, case files, suites, gates or fingerprints of case
  inputs.

## Files

- `evals/runner.py`
- `evals/adapters/endpoint.py`
- `evals/tests/test_adapters.py` (new assertions)

## Verification

- New test: an `InMemoryAdapter` receives a system message whose
  `List of tools:` precedes `Area context:`, and an empty tools list.
- New test: `EndpointAdapter` payload has no `tools` key for an empty list.
- `python -m pytest evals/tests tests/test_eval.py -q` passes.
