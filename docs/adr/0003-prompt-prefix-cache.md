# 0003. Prompt prefix cache on the embedded backend

## Status

Accepted.

## Context

The embedded backend runs LFM2, which is hybrid: attention plus recurrent short convolution. llama-cpp-python cannot trim the attention cache when the prompt diverges before its end, so it reset and re-evaluated the whole prompt. Each turn took a full 4,000–6,000-token prefill, which was 8–13 seconds on a Home Assistant host at about 500 tokens per second.

The system prompt and tools are identical across commands in a home. Two things defeated reuse. The area block and the routed tool list both sat before any cache point, and both changed with every command. The external llama-server backend already caches, so this decision concerns only the embedded backend.

## Decision

- The full tool list is written into the system message before the Area context block. The text is byte-identical to the LFM2.5 chat template (Jinja `tojson`, `", "` separator). The embedded backend sends no `tools` or `tool_choice` to llama (`382623c`). Tool key order follows the training order (`b5f615f`).
- The external backend is unchanged and still sends tools through the API.
- The recurrent state is checkpointed with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` at two boundaries: before the Area context block and before the last user message. The last-user checkpoint mirrors llama-server. Only one checkpoint per boundary is kept, and the least recently used one is evicted (`39a7533`, PR #167).
- Traces record `prefix_hash` over the stable part and `reused_tokens` from a counter in `CheckpointingLlama`.
- The model and validation always receive the complete compiled schema. Domain routing does not narrow it.

Measured on v11 Q4_K_M, CPU, 120-case promotion suite:

| | before | after |
| --- | --- | --- |
| Passed | 91 | 91 (4 flips) |
| Cache hits | 34 | 100 |
| Median prompt tokens re-read | about 4,300 | 44 |
| Tokens re-read in total | 430,362 | 102,835 |
| Wall time | 1,109 s | 393 s |

Tool key order: 91/120 with the 1.17 layout, 94/120 with the training key order (`b5f615f`).

Rejected:

- Moving the area block into the user turn. Aliases fell from 6/10 to 2/10 and entity picks went wrong.
- Filtering tools by domain to shrink the prompt. The tool block would change with every command, so the cache could not hit.
- llama-server flags. They do not apply to the embedded backend.

## Consequences

- A catalog change invalidates the checkpoint for the next command. Within a home, commands after the first reuse the prefix.
- The embedded prompt carries the full catalog, so it is larger than the per-command routed prompt it replaced. Prefill is paid once per home and boundary instead of per command.
- Each checkpoint costs memory, a few hundred KB by the plan's estimate. This is the only standing RAM the cache adds.
- Not built: pre-building the prefix at model load and on exposure change.

## Shipped in

- `382623c` feat(integration): reuse the cached prompt across embedded voice commands
- `b5f615f` fix(integration): inline the tool list in the key order the model trained on
- `39a7533` fix(integration): evict the least recently used prompt checkpoint
- `2da5fe9` Merge pull request #166 from mossipcams/ajax/cache
