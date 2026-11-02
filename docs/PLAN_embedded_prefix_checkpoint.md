# Plan: embedded prefix checkpoint for LFM2 (latency)

## Why

Embedded inference re-evaluates the full 4-6k token prompt on every new turn
(8-13 s on HA, ~500 tok/s). The system prompt and tools are byte-identical
across turns, but LFM2 is hybrid (attention + recurrent shortconv), so
`Llama.generate()`'s `kv_cache_seq_rm(0, prefix, -1)` returns false when the
cache diverges before its end, and llama-cpp-python falls back to `reset()`.

Findings from llama-cpp-python 0.3.33 source (vendored llama.cpp):

- `cache_prompt`, `--ctx-checkpoints` (default 32) and `--checkpoint-min-step`
  (default 8192) are llama-server features only (`tools/server`). The
  `Llama` class has no checkpoint support, so the embedded backend has none
  of this today. The external backend (llama-server) already gets it by default.
- llama-server always checkpoints right before the last user message
  regardless of the 8192 min step (`is_last_user_message ||` in
  `server-context.cpp`); the min step only thins intermediate checkpoints. For
  SaySo prompts (< 8192) that means one checkpoint per prompt, at the
  boundary that matters.
- `llama_state_seq_get_data_ext` / `llama_state_seq_set_data_ext` with
  `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` save/restore only the recurrent part of
  hybrid memory (kilobytes for LFM2). After a restore at position P the
  recurrent tail is at P-1, so `seq_rm(0, P, -1)` succeeds and trims the
  attention KV. This is what llama-server does.

## Scope

Mirror llama-server's last-user-message checkpoint inside SaySo, without
changing llama-cpp-python or llama.cpp:

- `custom_components/sayso/inference.py`: `_load` returns a small `Llama`
  subclass that overrides `generate(tokens, ...)`:
  1. P = start of the last `<|im_start|>user` token run in `tokens`
     (tokenized once with `special=True`). If not found, behave exactly as today.
  2. If a checkpoint exists for exactly `tokens[:P]`, the live cache
     (`input_ids[:n_tokens]`) shares at least P tokens with `tokens`, and the
     cache does not already prefix-match the whole cached prompt, restore the
     checkpoint (PARTIAL_ONLY, seq 0), `seq_rm(0, P, -1)`, set `n_tokens = P`.
  3. Otherwise, if no checkpoint for `tokens[:P]`: run the normal reset path,
     `eval(tokens[:P])`, save the PARTIAL_ONLY state as the checkpoint.
  4. Delegate to `super().generate(...)`, which now prefix-matches P tokens
     and evaluates only the suffix.
  One checkpoint only (it is a few hundred KB at most); replaced when the
  prefix changes. No new config, no new dependency.
- `tests/test_inference.py`: one test for the boundary/decision logic with a
  fake context (no real model).

## Out of scope

- llama-server flags (external backend already caches).
- `n_rs_seq`, `LlamaRAMCache`/`LlamaDiskCache`, disk snapshots, idle reload.

## Verification

- Benchmark (scratch script, not committed) on the Mac or `ssh llm` CPU with
  `SaySo-LFM-v11-Q4_K_M.gguf`, n_ctx 8192, 4 threads, production-shaped
  prompt from `evals/runner.py` (system prompt + tools) and 5 different
  utterances. Report per-turn prompt time for: stock `Llama` vs checkpointed
  subclass, first turn and turns 2-5, plus the checkpoint byte size.
- Correctness: greedy (temperature 0) outputs identical between stock and
  checkpointed runs for all utterances, including a 2-turn follow-up and a
  tool-call/tool-result round trip.
- `ruff check .`, `ruff format --check .`,
  `pytest tests evals/tests --ignore=tests/test_realistic_v3.py`,
  `pytest -q satellite/sayso`.

## Constraints

- No code comments or docstrings (AGENTS.md).
- No changes to llama-cpp-python/llama.cpp; no added standing RAM beyond the
  checkpoint bytes.
- Do not modify existing test assertions.
