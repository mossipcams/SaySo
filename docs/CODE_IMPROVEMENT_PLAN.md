# SaySo Code Improvement Plan

Status: proposed; no implementation is authorized by this document.

## Outcome

Make SaySo's Home Assistant tool path protocol-correct, smaller, safer, diagnosable,
and measurably compatible without changing ownership of wake word, audio, STT, TTS,
satellites, or llama.cpp hosting.

The implementation order is intentionally:

1. Correct tool-call transcripts.
2. Compile deterministic tool schemas.
3. Harden validation and correction at the model boundary.
4. Enable conservative context-based tool filtering behind that boundary.
5. Measure the completed path on the minimum and current Home Assistant versions.

Boundary hardening precedes filtering because a reduced schema is safe only when an
invalid route can fail closed or retry once with the complete schema.

## Current baseline

- `custom_components/sayso/conversation.py` currently adds one assistant message per
  returned tool call. A single model turn with multiple calls therefore becomes
  multiple assistant turns.
- `_format_tools()` converts every HA tool directly with `voluptuous_openapi.convert`
  on the initial request and every follow-up. There is no canonical ordering,
  fingerprint, or cache.
- Tool names and the top-level argument type are checked, while HA performs detailed
  argument validation during execution. There is no pre-execution batch validation or
  correction attempt.
- All HA tools are sent on every request. Home Assistant remains the source of entity
  exposure and target resolution.
- User-facing errors are intentionally short, but diagnostics do not distinguish model
  boundary failures.
- `pyproject.toml` declares Home Assistant `>=2024.12.0`; `README.md` still says 2024.8.
  The compatibility matrix will treat 2024.12 as the minimum unless that dependency is
  deliberately changed first.
- The checkout has a root `ARCHITECTURE.md`, but the `docs/ARCHITECTURE.md` and
  `docs/EVALUATION_PLAN.md` named by repository instructions are absent. This plan does
  not assume undocumented runtime components.

## Guardrails

- Keep Home Assistant's `APIInstance`, `Tool`, tool parameter schemas, execution, entity
  exposure, ambiguity handling, and context as the authorities.
- Never execute any call in a returned batch until every call name and argument object
  in that batch has passed pre-execution validation.
- Never retry after a tool has executed; that could duplicate side effects. Correction
  is allowed only before execution and only once per model turn.
- Never hide tools solely because a target entity, area, or floor guess is uncertain.
  Uncertain routing sends the complete compiled schema.
- Filtering changes only the tool definitions sent to llama.cpp. It does not remove
  entities from HA context or implement a second entity exposure policy.
- Unknown tool metadata is retained, not filtered out.
- Keep spoken failures generic. Put explicit, redacted failure categories in traces,
  logs, and config-entry diagnostics.
- Extend the existing test suite; do not create another `tests/` directory or add a new
  test framework/dependency.
- Use only the standard library and already-installed HA dependencies.

## Design decisions

### Transcript shape

For each llama.cpp model turn, build one `conversation.AssistantContent` containing all
returned `llm.ToolInput` values in model order. Pass it once to
`chat_log.async_add_assistant_content()`, consume all emitted tool results in order, and
only then decide whether to request a follow-up or return a failure. This produces:

```text
assistant(tool_call A, tool_call B)
tool(result A)
tool(result B)
```

Successful results remain recorded if a later call in the same batch fails. SaySo then
returns a failure without retrying or claiming total success.

### Schema compiler and cache

Keep `voluptuous_openapi.convert()` as the source conversion so executable HA
constraints are not reimplemented. The compiler will:

- sort tools by name;
- recursively sort object keys and order-insensitive `required` arrays;
- preserve validation keywords, defaults, enums, formats, descriptions, and custom
  selector output;
- remove only provably redundant text: empty descriptions, top-level `$schema`, and a
  `title` identical to its containing property/tool name;
- encode canonical JSON with stable separators;
- expose a SHA-256 compatibility fingerprint of the exact emitted schema; and
- use a small `functools.lru_cache` keyed by canonical source JSON.

Home Assistant 2024.12 and current `APIInstance` APIs expose tools but no schema revision
token. Safe cross-turn change detection therefore still requires converting the current
HA schema before comparing canonical input. The cache avoids rebuilding normalized
output when that input is unchanged, and the compiled object is reused directly across
all follow-up iterations in the same conversation turn. It must not use object identity
or `repr()` as an invalidation signal.

### Routing and filtering

Derive a conservative routing hint from the user's current command and HA registry
metadata:

- an explicit entity/domain term can identify a command domain;
- an exact area or floor name can narrow candidate domains only when all matching,
  exposed candidates agree;
- a satellite's preferred area/floor is supporting evidence, never sufficient by
  itself; and
- zero, conflicting, or fuzzy matches mean "unknown" and use the complete schema.

Filter only tools whose HA metadata explicitly declares incompatible domains. Retain
generic tools, scripts, queries, and tools without usable domain metadata. Keep the full
compiled schema beside the selected subset so a filtered-tool miss can receive one
pre-execution correction request with the full schema.

### Boundary diagnostics

Use these stable diagnostic codes:

- `schema_mismatch`: missing/unexpected argument names or a request/schema fingerprint
  inconsistency;
- `invalid_arguments`: arguments have the right shape but fail HA type, enum, range, or
  pattern validation;
- `unavailable_tool`: the name is absent from the complete current HA tool map;
- `request_timeout`: llama.cpp timed out;
- `iteration_limit`: the configured tool-iteration limit was reached; and
- `tool_execution_failed`: HA returned an error after execution began.

Store only counts plus the last code, compatibility fingerprint, phase, and timestamp.
Do not store prompts, argument values, tool results, entity names, API keys, or URLs in
the new boundary diagnostics.

## TDD implementation units

Every numbered unit is sized for roughly 5–15 minutes. Execute one unit at a time. For
each unit: add the named failing test first, run it and show the failure, add the minimum
implementation, run the focused test and relevant regression set, then stop and ask
`Task N done. Continue?`.

### Phase 1 — Protocol-correct transcripts

#### Task 1 — Batch a model turn's tool calls

- **Test:** Extend `tests/test_conversation.py` with one scenario covering single,
  multiple, failed, and partial-success call batches. Assert one assistant message holds
  every call in order, each result follows it with the matching ID, a full success gets a
  follow-up, and any failure prevents a follow-up and a success claim.
- **Implement:** Replace the per-call `AssistantContent` loop with one batch content
  object and consume its results before branching on failure.
- **Verify:** Run the focused four cases, then the complete conversation test module.

#### Task 2 — Lock the llama.cpp transcript serialization contract

- **Test:** Add a focused serialization case for two calls and mixed success/error tool
  results, asserting exact OpenAI message roles, order, IDs, function names, and JSON
  argument/result strings.
- **Implement:** Make the smallest `_chat_log_to_messages()` adjustment needed for exact
  one-assistant/many-tool ordering and deterministic argument JSON.
- **Verify:** Run the serialization case plus `tests/test_client.py` payload tests.

### Phase 2 — Deterministic compiled schemas

#### Task 3 — Preserve HA constraints during compilation

- **Test:** Add a fake HA tool with nested objects, required keys, arrays, enums,
  min/max, pattern, format, defaults, descriptions, and a custom serializer. Assert each
  executable constraint survives compilation.
- **Implement:** Introduce the compiler around the existing `convert()` call; do not
  hand-translate Voluptuous validators.
- **Verify:** Run the compiler test and compare the generated function shape with the
  existing llama.cpp request contract.

#### Task 4 — Remove only redundant schema text

- **Test:** Show that empty descriptions, `$schema`, and titles duplicating their
  containing names disappear, while distinct titles/descriptions and all executable
  keywords remain.
- **Implement:** Add one recursive normalizer with the narrow removal rules above.
- **Verify:** Run the focused test and inspect the before/after serialized byte counts.

#### Task 5 — Make ordering and fingerprint deterministic

- **Test:** Compile semantically identical tools supplied in different orders and with
  different dictionary insertion orders. Assert byte-identical output and fingerprints;
  then change one constraint and assert the fingerprint changes.
- **Implement:** Canonicalize tool order, mapping keys, and `required`; hash the exact
  canonical emitted JSON with `hashlib.sha256`.
- **Verify:** Run the test repeatedly in separate Python processes to rule out hash/order
  dependence.

#### Task 6 — Cache unchanged schemas and invalidate changed schemas

- **Test:** Spy on the normalization/build function. Assert two identical canonical
  inputs build once, a changed description or constraint rebuilds, and a follow-up model
  iteration reuses the same compiled object without another HA conversion.
- **Implement:** Add a bounded standard-library `lru_cache` for normalized compilation
  and carry one compiled full schema through `_async_handle_tool_calls()`.
- **Verify:** Run cache tests and the full conversation module; expose cache hit/miss
  counts only if they are useful in existing diagnostics.

#### Task 7 — Replace direct `_format_tools()` calls

- **Test:** Assert the initial and every follow-up llama.cpp call receive the compiled
  schema and the same compatibility fingerprint is associated with that model turn.
- **Implement:** Route both initial and follow-up request construction through the
  compiler; remove the now-unused direct formatting path.
- **Verify:** Run all client and conversation tests and search for remaining production
  `_format_tools()` callers.

### Phase 3 — Harden the model boundary

#### Task 8 — Prevalidate the whole call batch

- **Test:** Return two calls where the first is valid and the second has a missing ID,
  duplicate ID, or unknown name. Assert no HA tool executes.
- **Implement:** Build one complete-name tool map and validate non-empty unique IDs,
  non-empty names, name availability, and argument-object type before execution.
- **Verify:** Run the batch validation cases and existing unknown/malformed-call tests.

#### Task 9 — Validate and normalize arguments with HA schemas

- **Test:** Cover valid normalized values, missing/unexpected fields, wrong types, enum
  and range violations, and nested failures. Assert validation happens before HA
  execution and classify schema mismatch separately from invalid values.
- **Implement:** Call the selected HA tool's Voluptuous schema once, catch its validation
  errors, and pass the validated copy to `llm.ToolInput`.
- **Verify:** Run focused argument tests on both supported HA environments because tool
  schema behavior changed across HA versions.

#### Task 10 — Allow one pre-execution correction

- **Test:** Have the model first return a repairable invalid call and then a valid call.
  Assert the correction request contains a protocol-valid assistant tool-call message,
  matching synthetic tool error result, allowed tools, and current fingerprint; assert
  only the corrected call executes.
- **Implement:** Build correction messages locally without recording an unexecuted call
  as a successful HA action. Reuse the normal llama.cpp request method and compiled
  schema.
- **Verify:** Run the repair case and inspect the exact second POST payload.

#### Task 11 — Cap repair and protect side effects

- **Test:** Assert a second invalid response fails after one correction, timeout never
  retries, and a failure after any tool result never retries or re-executes earlier
  calls.
- **Implement:** Add one per-model-turn correction flag and keep correction entirely
  before the execution branch.
- **Verify:** Run retry-limit, timeout, execution-failure, and partial-success cases;
  assert exact client and HA call counts.

#### Task 12 — Record explicit redacted diagnostics

- **Test:** Trigger each boundary code and assert diagnostics contain counts and safe
  last-failure metadata while excluding prompts, args, results, entity names, keys, and
  URLs.
- **Implement:** Add a small runtime dictionary/counter and include its redacted snapshot
  in `diagnostics.py`; log the stable code and phase at debug level.
- **Verify:** Run conversation and diagnostics tests, including existing secret-redaction
  assertions.

#### Task 13 — Distinguish timeout and iteration-limit paths

- **Test:** Cover timeouts on initial, correction, and follow-up requests plus the exact
  configured iteration boundary. Assert spoken output stays generic and diagnostics use
  `request_timeout` or `iteration_limit`.
- **Implement:** Centralize the existing repeated client-exception mapping and record the
  phase before returning the current user-facing error.
- **Verify:** Run all failure-path conversation tests and confirm no success response is
  emitted.

### Phase 4 — Reduce inference context safely

#### Task 14 — Identify only confident command domains

- **Test:** Add exact domain/entity-name commands, conflicting matches, unknown terms,
  and ordinary non-control chat. Assert only exact, unambiguous cases produce a domain
  hint; all others return unknown.
- **Implement:** Use simple normalized token matching against HA-provided names and
  domains. Do not add fuzzy matching, embeddings, or a second model.
- **Verify:** Run routing tests with punctuation, case, plural wording, and alias cases.

#### Task 15 — Add area and floor evidence

- **Test:** Cover exact area, exact floor, duplicate names, mixed-domain contents,
  preferred satellite area, and no registry match. Assert area/floor evidence narrows a
  route only when candidate domains agree.
- **Implement:** Read HA area, floor, device, and entity registries without changing
  exposure or target lists; combine the evidence with Task 14's confidence result.
- **Verify:** Run routing tests and assert no entity list or HA API prompt is mutated.

#### Task 16 — Select a safe tool subset

- **Test:** For a confident light command, assert incompatible domain-declared tools are
  removed while generic, query, script, and metadata-unknown tools remain. Assert an
  uncertain command receives the complete schema byte-for-byte.
- **Implement:** Add a pure selector over compiled tools plus their source HA tool
  metadata. Never infer applicability from tool-name substrings when HA metadata is
  absent.
- **Verify:** Run selector tests against representative minimum/current HA tool objects
  and compare full versus selected prompt token counts.

#### Task 17 — Recover from a filtered-schema miss

- **Test:** Simulate a call absent from the selected subset but present in the complete
  HA map. Assert one correction request uses the complete schema; an actually unavailable
  name reports `unavailable_tool`; neither path executes before validation.
- **Implement:** Integrate selection with Task 10's single correction budget and keep the
  complete compiled schema for fallback.
- **Verify:** Run confident, ambiguous, false-route, and unavailable-tool end-to-end
  conversation cases with exact execution counts.

### Phase 5 — Performance and compatibility evaluation

#### Task 18 — Define the fixed evaluation cases and scorer

- **Test:** Add scorer cases for correct tool/name/args, wrong tool, invalid call,
  clarification, multi-call order, partial failure, and final spoken result.
- **Implement:** Add a small versioned JSON case set and Python runner using current
  project types. Include core control, safety/ambiguity, query, multi-call, failure, and
  follow-up cases; keep case IDs independent from any future tuning data.
- **Verify:** Run the scorer entirely offline and produce deterministic JSON results.

#### Task 19 — Measure request size and tool quality

- **Test:** Feed recorded llama.cpp responses into the runner and assert calculations for
  serialized request bytes, prompt tokens, tool accuracy, invalid-call rate, and latency
  percentiles.
- **Implement:** Capture the exact `/v1/chat/completions` payload emitted by
  `LlamaCppClient`; prefer llama.cpp `usage.prompt_tokens` and record serialized bytes as
  the always-available fallback.
- **Verify:** Compare baseline and improved reports from the same fixtures and reject
  missing metric fields.

#### Task 20 — Measure live TTFT and end-to-end latency

- **Test:** Use a fake SSE server to assert TTFT starts immediately before POST and ends
  on the first generated-token event, while end-to-end latency ends after the final HA
  result.
- **Implement:** Add an opt-in eval-only live runner. Send the production Chat
  Completions payload directly to llama.cpp; add only `stream: true` for the TTFT probe.
  Do not add production streaming support. Record warmups, repetitions, median, and p95.
- **Verify:** Run once against a real `llama-server --jinja` and save machine-readable
  output containing llama.cpp build, GGUF hash, chat template, hardware, and exact server
  arguments.

#### Task 21 — Exercise minimum and current HA compatibility

- **Test:** First run the suite in clean environments for HA 2024.12 and the latest
  released HA version allowed by the project, recording any incompatibility before
  changing automation.
- **Implement:** Add a two-entry CI/manual matrix using standard virtual environments and
  exact resolved versions. Align the README minimum with `pyproject.toml`; do not broaden
  or lower support silently.
- **Verify:** Both environments run transcript, compiler, boundary, routing, request
  contract, and offline eval tests. The live llama.cpp job may be manual but must use the
  same matrix and publish its report.

#### Task 22 — Record a non-moving baseline and release gates

- **Test:** Run the same evaluation runner against `origin/main` and the feature branch
  in separate worktrees with identical HA, llama.cpp, model, template, hardware, warmup,
  and repetition settings. Fail report comparison when metadata differs.
- **Implement:** Add a compact Markdown/JSON comparison report and gates: no tool-accuracy
  regression, no invalid-call-rate increase, lower prompt tokens for confidently routed
  cases, and no unexplained TTFT or end-to-end latency regression. Set numeric latency
  tolerance from the recorded baseline before release, not after seeing failures.
- **Verify:** Produce reports for both supported HA versions and archive the exact inputs
  and fingerprints needed to reproduce them.

## Completion criteria

- One assistant transcript message represents every model tool-call turn, with all
  corresponding tool results preserved in order for success, failure, and partial
  success.
- The schema sent to llama.cpp is deterministic, constraint-preserving, fingerprinted,
  cached for unchanged canonical input, and rebuilt when canonical HA tool output
  changes.
- Confident routing reduces tools; uncertain routing sends the full schema; a false route
  cannot silently hide a valid complete-schema tool.
- Every call batch is validated before execution, at most one pre-execution correction is
  allowed, and no executed action is retried.
- Diagnostics distinguish schema mismatch, invalid arguments, unavailable tools,
  request timeouts, iteration limits, and execution failure without storing sensitive
  request data.
- The fixed suite reports prompt size, TTFT, tool accuracy, invalid-call rate, and
  end-to-end latency using the actual llama.cpp Chat Completions request on HA 2024.12
  and the exact current supported HA release.
- Existing conversation, client, config-flow, diagnostics, and setup tests remain green.

## Explicit deferrals

- No new router model, embeddings, fuzzy matcher, schema framework, cache service,
  streaming production client, generalized satellite layer, model bake-off, tuning run,
  or large eval corpus.
- Add those only if the fixed evaluation suite shows the minimal deterministic approach
  cannot meet accuracy or latency goals.

## Compatibility references

- [Home Assistant 2024.12 LLM helper source](https://github.com/home-assistant/core/blob/2024.12.0/homeassistant/helpers/llm.py)
- [Home Assistant 2026.8 LLM helper source](https://github.com/home-assistant/core/blob/2026.8.0/homeassistant/helpers/llm.py)
- [llama.cpp server Chat Completions, tool calls, streaming, and timings](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

Plan ready. Approve to proceed.
