# 0001. Compiled tool schema and the model boundary

## Status

Accepted. The v1 schema lock has not been declared; see Consequences.

## Context

SaySo gives Home Assistant's Assist tools to llama.cpp as function tools. Before this work, every tool went to the model on every request with no canonical order or fingerprint. Model calls ran without validation before execution, and a turn with several calls was recorded as several assistant messages.

Home Assistant 2026.9 replaced voluptuous with probatio. The existing converter then returned an empty parameter object for every tool, so the model could not produce arguments. Restoring parameters took three reported utterances from 0/3 to 3/3 tool calls (issue #52).

Home Assistant stays the authority for tool definitions, exposure, argument validation, and execution. SaySo must not hand-translate HA's validators.

## Decision

- Convert each tool with HA's own converter: `voluptuous_openapi`, falling back to `probatio.to_openapi`. A tool that cannot be converted is withheld. It is never sent with empty parameters.
- Compile to the OpenAI-compatible `function` envelope. Tools are sorted by name, object keys are sorted, and `required` is order-insensitive. Only redundant metadata is removed: empty descriptions, a top-level `$schema`, and titles that repeat their property or tool name.
- The fingerprint is `sha256:` plus the SHA-256 of the canonical emitted bytes. Each request phase records the fingerprint of the tools it actually sent.
- Outer envelopes are validated before transport. Every call batch is validated before any HA execution. One pre-execution correction is allowed, and no executed action is retried.
- The reference artifact `schemas/sayso-tool-schema-v1.json` is produced by `scripts/generate_sayso_tool_schema.py`. The generator refuses to overwrite it, and any byte change needs a new contract version.
- Evals offer the HA 2026.11.1 Assist catalog per home, built from `schemas/ha-2026.11.1-tools.json` with HA's list-time rules.
- Grammar decoding is built from the same compiled catalog.
- The model always receives the complete compiled schema. Domain-based filtering was removed. The domain hint is kept for traces only; ADR 0003 explains why a per-command tool list was dropped.

Rejected:

- A hand-written replacement for the HA converter.
- Filtering tools by domain to shrink the prompt. It was shipped, then removed, because it broke prompt reuse. The filtered-miss correction path was never shipped.
- An empty-parameter fallback.
- A router model, embeddings, fuzzy matching, or a schema framework.

## Consequences

- Schema identity can be checked. Envelope errors fail before llama.cpp is called. Argument failures are classified as schema mismatch, invalid arguments, or unavailable tool in redacted diagnostics.
- The full catalog is sent on every embedded request, so prompt size is set by the HA catalog, not by the command.
- Open: `docs/SCHEMA_LOCK_PLAN.md` Task 6 has not run. The plan still reads "proposed", so v1 is a reference artifact but is not declared locked. Its physical-device gate and human review are outstanding.
- `docs/CODE_IMPROVEMENT_PLAN.md` tasks 14–17 (routing and filtered-miss recovery) were superseded by the decision above. Tasks 20–22 (live TTFT measurement and the baseline comparison) are not in the repo.

## Shipped in

- `7bf0fc0` Add compiled tool schemas, batch validation, and conservative domain routing.
- `2eba964` Add conservative tool filtering, eval scoring, and HA compatibility matrix. (Filtering later removed by `382623c`.)
- `10b8567` fix(schema): compile Home Assistant tool parameters through probatio
- `22c0f93` Add compiled tool schema identity, envelope validation, and the v1 reference artifact.
- `382623c` feat(integration): reuse the cached prompt across embedded voice commands (removed domain filtering)
- `c8ae9b1` feat: build GBNF grammars from the compiled tool catalog
- `7446648` fix(evals): offer the HA 2026.11.1 Assist tool list per home
- `134a0a3` feat: add the latest Home Assistant compat row
