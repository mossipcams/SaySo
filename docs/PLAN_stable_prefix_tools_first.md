# Plan: stable prefix with JSON tools before the area block (v11, no retrain)

## Why

Production misses the embedded checkpoint because the area block and the
routed tool list change per command, and both sit before the cached point.
Moving the area block into the user turn broke v11 (aliases 6/10 -> 2/10,
wrong entity picks). This tests a smaller change: keep the area block at the
end of the system message, but render the full JSON tool list (byte-identical
to what the chat template emits) before it:

```
system: instructions + entities + rules + "\nList of tools: [...]" + "\n" + area block
user:   command
```

Everything before the area block is then identical for every command in a home.

## Scope

- `scripts/eval_stable_prefix.py` (uncommitted measurement script):
  - new arm `tools_first`: no `tools` passed to `create_chat_completion`; the
    tool list text is appended to the system head before the area block, and
    must be byte-identical to the template's own rendering (assert it against
    the template-rendered baseline prompt).
  - `--order household`: stable-sort cases by household so consecutive cases
    can share a prefix; used for both arms.
  - per case: prefill tokens actually evaluated (exclude decode tokens).

## Measurement

v11 Q4_K_M, CPU (`GGML_METAL_DEVICES=0`), 4 threads, temperature 0, promotion
suite. Arms `baseline` and `tools_first`, both `--order household`. Aliases
first as a smoke check, then all 120 cases. Report per-category pass counts,
prefill tokens evaluated per case, wall time, and every flipped case with both
outputs and whether it is model or grader.

## Gate

Ship path (separate PR) only if `tools_first` has no category regression
beyond 1 case and no drop in unsupported-request refusals. Then: send the
model the full catalog in that layout (routing kept for validation),
checkpoint at the last user message, cache tracing counting evaluated tokens,
and a prefix pre-build at load and on exposure change.
