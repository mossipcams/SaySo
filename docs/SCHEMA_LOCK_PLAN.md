# SaySo Tool Schema Lock Plan

Status: proposed; no schema implementation is authorized by this document.

## Outcome

Produce one reviewed, deterministic SaySo tool-schema artifact that can be treated as
immutable after its lock gates pass. The lock covers the emitted llama.cpp function-tool
contract and the exact reference artifact. Home Assistant remains the authority for tool
definitions and argument validation.

## Current assessment

The current compiler already has the right foundation:

- Home Assistant `Tool` objects and `voluptuous_openapi.convert()` remain the source of
  tool names, descriptions, parameter shapes, and executable constraints.
- The emitted envelope uses the OpenAI-compatible `type: function` shape expected by
  llama.cpp.
- Normalization removes only redundant text, canonicalization sorts object keys and
  order-insensitive `required` arrays, and tools are sorted by name.
- Tool arguments are validated with the original Home Assistant Voluptuous schema before
  execution.
- The complete schema remains available for the single pre-execution correction path.

Do not lock the current implementation yet. Four issues remain:

1. A confidently routed request can send a filtered tool tuple while diagnostics retain
   the fingerprint of the complete tool tuple. The fingerprint therefore does not always
   identify the exact emitted schema.
2. `schema_fingerprint()` hashes its input directly instead of routing through the one
   canonical byte emitter. Correct callers happen to provide canonical input, but the
   function does not enforce its own contract.
3. The compiler does not explicitly reject an invalid outer tool envelope before sending
   it to llama.cpp, including duplicate or invalid names and non-object parameter roots.
4. There is no versioned, checked-in reference artifact or documented rule that makes a
   successful lock immutable.

## Lock boundary

### Lock these invariants

- Each emitted entry has this shape:

  ```json
  {
    "type": "function",
    "function": {
      "name": "ToolName",
      "description": "Optional description",
      "parameters": {
        "type": "object"
      }
    }
  }
  ```

- `function.name` is unique and satisfies the llama.cpp/OpenAI-compatible function-name
  constraint used by SaySo.
- `function.parameters` is a JSON-serializable object schema.
- Home Assistant conversion output remains authoritative; SaySo does not recreate field
  constraints or force `additionalProperties: false`.
- Canonicalization recursively sorts object keys, sorts `required`, preserves other array
  order, and sorts tools by function name.
- Empty descriptions, a top-level `$schema`, and titles identical to their containing
  property or function name are the only metadata removed.
- A fingerprint is `sha256:` followed by the SHA-256 digest of the exact canonical UTF-8
  bytes emitted for that schema.
- Complete and filtered schemas each carry their own exact fingerprint.
- Every returned tool-call batch is validated before any Home Assistant action executes.

### Do not lock these runtime values

- The tool catalog of every Home Assistant installation.
- Entity, area, floor, or device names.
- Tool ordering supplied by Home Assistant before canonicalization.
- Model, GGUF, chat template, prompt, or llama.cpp build.
- Cache object identity or cache hit counts.

The checked-in artifact is one named reference schema. A different Home Assistant setup
may produce another fingerprint without violating the locked compiler contract.

## Target internal shape

Reuse the existing `CompiledToolSchema` type; do not add another schema hierarchy.

```python
complete_schema = compile_llm_tools(llm_api)
active_schema = select_schema_for_domain(complete_schema, llm_api.tools, domain_hint)
```

Both values contain canonical `tools` and their own fingerprint. Unknown routing returns
the same complete object. Confident filtering returns a new `CompiledToolSchema` only when
the selected tuple differs. Correction requests use `complete_schema`; ordinary initial
and follow-up requests use `active_schema`.

## TDD implementation units

Execute one numbered unit at a time. For each unit, write the focused failing check first,
run it and record the failure, add the minimum implementation, run the focused check and
relevant regressions, then stop unless continued execution has already been authorized.
Do not modify files under `tests/` without explicit permission.

### Task 1 — Make fingerprinting canonical and self-describing

- **Test:** Show that equivalent tools with different dictionary and tool order produce
  the same fingerprint, any canonical emitted-byte change produces a different
  fingerprint, and every value matches `sha256:<64 lowercase hex characters>` even when
  the caller supplies non-canonical input.
- **Implement:** Make `schema_fingerprint()` hash only `emit_canonical_json()` output and
  add the `sha256:` prefix. Keep one hashing implementation.
- **Verify:** Run the focused schema check repeatedly in separate Python processes, then
  run the existing schema regression module unchanged.

### Task 2 — Give the active schema its own identity

- **Test:** For a confident light route, assert that the selected schema contains fewer
  tools, its fingerprint equals its exact emitted bytes, and it differs from the complete
  fingerprint. For an uncertain route, assert the complete object is returned unchanged.
- **Implement:** Change the pure routing selector to accept and return
  `CompiledToolSchema`, constructing a new instance only for a real subset.
- **Verify:** Run the focused routing and schema checks and confirm the selected request
  remains smaller than the complete request.

### Task 3 — Carry exact schema identity through every request phase

- **Test:** Cover initial, ordinary follow-up, argument correction, and filtered-miss
  correction requests. Assert each request and boundary diagnostic uses the fingerprint
  of the tools actually sent in that phase.
- **Implement:** Keep `complete_schema` and `active_schema` in the conversation flow.
  Send and record the active schema for normal requests and the complete schema for
  correction requests. Do not alter execution or retry policy.
- **Verify:** Run focused conversation and diagnostics checks, then search production code
  for any request that passes a raw tool tuple without its corresponding schema identity.

### Task 4 — Validate the outer tool envelope before transport

- **Test:** Reject duplicate names, names outside the supported character/length rule,
  non-object parameter roots, malformed function wrappers, and non-JSON-serializable
  values. Assert representative Home Assistant 2026.8.3 tools still compile unchanged.
- **Implement:** Add one small compiler-boundary validator after Home Assistant conversion
  and before caching or transport. Validate only the outer contract; leave parameter-field
  semantics to Home Assistant and Voluptuous.
- **Verify:** Run the focused invalid-envelope cases, the existing compiler tests, and the
  current Home Assistant compatibility check.

### Task 5 — Generate the immutable reference artifact

- **Test:** Generate the artifact twice from the same controlled Home Assistant API tool
  set and assert byte-identical output. Recompute the embedded fingerprint from `tools`
  and assert it matches.
- **Implement:** Add the smallest standard-library script that accepts the exact compiled
  tool payload from the controlled reference setup and writes one canonical JSON artifact
  containing:
  - `contract_version`: `sayso-tool-schema/v1`
  - exact Home Assistant version
  - SaySo source commit
  - Home Assistant LLM API identifier
  - schema fingerprint
  - canonical complete `tools`
- **Verify:** Regenerate into a temporary path, compare it byte-for-byte with the checked-in
  artifact, and independently recompute its fingerprint with production code.

The initial locked path will be `schemas/sayso-tool-schema-v1.json`. The generator must
refuse to overwrite an existing locked artifact. Any later byte change requires a new
contract version and file rather than editing v1 in place.

### Task 6 — Pass the lock gates and declare v1 locked

- **Test:** Run all focused schema-identity checks, the complete project suite, the pinned
  Home Assistant compatibility check, the basic core/safety/follow-up eval cases, and one
  llama.cpp `--jinja` tool-call smoke test against the reference payload.
- **Implement:** After every gate passes, update this document's status to `locked`, record
  the artifact path, fingerprint, Home Assistant version, source commit, and lock date.
  Make no schema changes in the locking edit.
- **Verify:** Confirm a real safe Home Assistant control travels through schema selection,
  llama.cpp tool calling, pre-execution validation, execution, verification, and spoken
  response. Confirm `git diff --check`, artifact regeneration, and the full suite are
  clean.

## Lock gates

Schema v1 is locked only when all of these are true:

- The active fingerprint always identifies the exact tools sent in its request.
- The complete fingerprint always identifies the exact complete fallback tools.
- Canonical output and fingerprints are stable across processes.
- Invalid outer envelopes fail before network transport.
- Existing safety, ambiguity, capability, correction, execution, and verification barriers
  remain intact.
- The reference artifact regenerates byte-for-byte from its recorded environment.
- The full project suite and pinned Home Assistant compatibility check pass.
- The basic eval set has no tool-accuracy or invalid-call-rate regression.
- llama.cpp accepts the artifact and returns a parseable tool call.
- One safe physical-device voice path succeeds end to end.
- A human review confirms the artifact contains only intended tool definitions and no
  prompts, entity context, credentials, URLs, or tool results.

## Change policy after lock

- Never edit `schemas/sayso-tool-schema-v1.json` in place.
- Any emitted-byte change creates a new fingerprint.
- Any change to the locked envelope, normalization, canonicalization, fingerprint format,
  validation semantics, or reference tool definitions requires a new contract version and
  artifact.
- Home Assistant installation-specific catalog drift is not itself a compiler-contract
  break, but it must not be presented as the locked v1 reference artifact.
- Cache refactors that preserve canonical bytes and behavior do not require a new schema
  version.

## Explicit deferrals

- No new schema framework, router model, cache service, custom Home Assistant API, or
  production streaming path.
- No hand-written replacement for `voluptuous_openapi.convert()`.
- No generalized artifact registry or migration system; add one only after a second
  schema version exists.
