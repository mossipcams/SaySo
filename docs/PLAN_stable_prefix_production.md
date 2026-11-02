# Plan: ship the stable prompt prefix (embedded backend)

## Evidence

v11 Q4_K_M, CPU, 120-case promotion suite, cases grouped by household
(`scripts/eval_stable_prefix.py`, results in /tmp/sayso-prefix-bench):

| | current layout | tools before area |
|---|---:|---:|
| passed | 91 | 91 (2 flips each way, read; noise) |
| cache hits | 34 | 100 |
| median prompt tokens re-read | ~4,300 | 44 |
| wall time | 1,109 s | 393 s |

Already in this worktree: the two-boundary embedded checkpoint
(`Area context:\n` and `<|im_start|>user`) in `inference.py`.

## Scope

Slice 1: layout and full catalog
- One shared function, e.g. in `area_context.py` or `tool_schema.py`, that
  takes messages (area block at the end of the system message) plus the tools,
  and returns messages with `"\nList of tools: [" + tools joined + "]"`
  inserted before the area block. The text must be byte-identical to the LFM2.5
  chat template (Jinja `tojson`, `", "` separator).
- `EmbeddedEngine._complete`: use it and pass no `tools`/`tool_choice` to
  llama. `ExternalEngine` is unchanged.
- `conversation.py`: the model and validation use the complete schema. Domain
  routing no longer narrows `active_schema`. Keep the computed `domain_hint` in
  the trace metadata.
- `evals/runner.py`: evals keep matching what production sends.
- Tests: the layout function is byte-identical to the template, with the area
  block last; the conversation offers the complete schema.

Slice 2: cache tracing
- Port only the trace fields from the `ajax-question` worktree:
  `ChatCompletionResult.prefix_hash`, `reused_tokens`, and span metadata.
  `prefix_hash` covers the stable part (system text before `Area context:` plus
  tools). `reused_tokens = prompt_tokens - prefill tokens actually evaluated`,
  taken from a counter in `CheckpointingLlama`, not from a token-list diff.
- Tests for both fields.

## Out of scope

`ExternalEngine` layout, the compact tool format, prefix pre-build at load, and
the routing module itself (left in place, just not used to narrow).

## Verification

- Unit tests above; CI gate: `ruff check .`, `ruff format --check .`,
  `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`,
  `pytest -q satellite/sayso`.
- Production path check: run promotion categories through `EmbeddedEngine`
  (not raw `Llama`) and confirm the prompt equals the eval script's
  `tools_first` prompt and the pass counts match (91/120 overall).
- PR against main from `ajax/cache`.
