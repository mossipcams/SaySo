# SaySo clean rebuild plan

Status: draft, awaiting approval  
Baseline date: 2026-09-01  
Companions: [Architecture](ARCHITECTURE.md), [Evaluation](EVALUATION_PLAN.md),
[Tuning](TUNING_PLAN.md)

## Decision

Rebuild the runnable MVP path from a clean environment; do not rewrite the
validated domain core from scratch.

The repository already has the intended architecture and extensive coverage.
The shortest safe rebuild is to preserve its contracts, assemble and prove one
vertical path at a time, and change code only when a focused failing check
exposes a real gap. A blank rewrite would recreate the same ControlPlan,
resolution, safety, execution, verification, and evaluation boundaries with
more risk and no demonstrated benefit.

## Goal

From a clean checkout, a Mac repeatedly completes:

```text
wake -> capture -> Home Assistant Assist STT -> SaySo ConversationEntity
     -> persistent WebSocket -> ControlPlan -> resolve/validate/safety
     -> caller-context HA execution -> state verification
     -> Assist TTS or local earcon -> Mac playback
```

The rebuild is complete only when this path controls an allowlisted physical
device and the core, safety, follow-up, and basic evaluation gates remain
green.

## Current baseline

- The production path is implemented, but the physical Mac demo has not been
  run.
- `uv run pytest -q`: 541 passed (committed API schema fixture includes
  conversation and prepare message types; Task 1 complete).
- `uv run pytest -q evals`: 217 passed.
- The RMS/energy wake detector is a prototype, but it is sufficient for the
  first physical demo.
- `context.json` and `evals/reports/all.report.json` are untracked. Preserve
  them and never commit `context.json`.

## Preserve these boundaries

- Home Assistant owns entity truth, exposure, permissions, caller context,
  service execution, and state verification.
- The model emits a typed semantic `ControlPlan`, never a Home Assistant service
  call or entity-ID authorization decision.
- Invalid, ambiguous, hidden, incapable, unsupported, or unsafe plans cannot
  mutate state.
- The integration keeps the initiating HA `Context` locally and sends no
  serialized authorization context to the server.
- Production conversation traffic uses the persistent WebSocket. The text HTTP
  endpoint remains evaluation/compatibility only.
- Action success requires observed state verification.
- Core, safety, follow-up, metrics, and basic eval coverage stays intact.

## Reuse instead of rebuilding

- `sayso-server`: ControlPlan schema/parser, graph, candidate retrieval,
  resolver, ambiguity, capability, safety, response policy, and telemetry.
- `custom_components/sayso`: graph snapshot/deltas, exposure, permissions,
  action mapping, and state verification.
- `sayso-satellite`: Assist client, PCM capture, wake session, response mapping,
  and playback adapters.
- `evals`: schemas, authored basic corpora, dry-run gate, metrics, runner, and
  reports.

## Explicitly deferred

- Phonetic wake-word replacement; add it after the energy detector proves the
  loop and false wakes become the measured bottleneck.
- Fine-tuning; follow `TUNING_PLAN.md` only after the physical voice path and
  frozen eval gate work.
- Live Home-LLM 270M bake-off, broader model benchmarking, larger corpora,
  generalized multi-satellite support, streaming optimization, and polished
  diagnostics.
- New frameworks, dependency changes, parallel authority paths, or a second
  home graph.

## Execution rules

Each numbered unit is a 5–15 minute stop point. Next stop is Task 2.
After every unit, report the red and green checks and ask, `Task N done.
Continue?`

For validation-only units, do not edit code when the focused check passes. If a
check fails and a code change is needed:

1. Add or identify one colocated `test_*.py` assertion that reproduces the
   failure.
2. Run it and show the failure.
3. Patch the shared root cause with the smallest implementation change.
4. Run the focused test and show it passing.
5. Run the phase regression set before continuing.

Never create or modify a `tests/` directory, weaken assertions, or fix a failed
test by changing its expected behavior. Fixture regeneration is allowed when
the runtime contract is already correct and the fixture is stale.

## Phase 1 — Restore a reproducible baseline

### Task 1 — Synchronize the committed protocol schema (5–10 min) — done

Completed. The committed fixture `evals/fixtures/sayso_api_v1.schema.json` now
includes conversation and prepare message types;
`test_sayso_api_v1_json_schema_matches_committed_fixture` passes, and
`uv run pytest -q` reports 541 passed.

Historical steps (already executed):

- Test first: schema drift was exposed by
  `test_sayso_api_v1_json_schema_matches_committed_fixture`.
- Minimal implementation: regenerated only
  `evals/fixtures/sayso_api_v1.schema.json` from the current
  `SaySoEnvelope.model_json_schema()` without changing the envelope or removing
  the implemented conversation/prepare message types.
- Verify: focused schema test, then `uv run pytest -q` and
  `uv run pytest -q evals`.

### Task 2 — Prove clean dependency and import setup (10–15 min)

- Test first: in a disposable environment, run `uv sync --frozen` followed by
  imports for `sayso_server` and `sayso_satellite`; record the first failing
  command if either fails.
- Minimal implementation: none when the lockfile is reproducible. If it fails,
  add one focused import/startup regression and change only the incorrect
  package metadata or lock entry; do not upgrade unrelated dependencies.
- Verify: repeat the frozen sync/import check and run the package import tests.

### Task 3 — Prove the deterministic control core (10 min)

- Test first: run the existing parser, schema, candidate, resolver, ambiguity,
  capability, safety, scope, ControlPlan, and orchestrator tests.
- Minimal implementation: none when green. For any failure, add the smallest
  missing colocated regression for that input and fix the common parser or
  validator path rather than individual callers.
- Verify: rerun the focused failure and the complete deterministic-core set.

Phase gate: all automated tests are green from the locked environment, and no
validated core behavior was rewritten merely for structural cleanliness.

## Phase 2 — Reassemble the resident server boundary

### Task 4 — Start the server with one resident runtime (10–15 min)

- Test first: run the existing app/main/runtime/readiness tests, including the
  assertion that model readiness is separate from liveness.
- Minimal implementation: reuse the current composition root. If startup fails,
  fix only the missing wiring or configuration validation; do not introduce a
  factory or alternate server path.
- Verify: start the server from the clean environment, confirm liveness, and
  confirm readiness fails closed until the graph and model are ready.

### Task 5 — Rebuild the HA graph handshake (10–15 min)

- Test first: run the envelope, gateway, home-graph, snapshot, delta, exposure,
  and reconnect tests; then connect the integration and observe graph readiness.
- Minimal implementation: none when the existing snapshot/delta path works. If
  it fails, repair the single serializer, envelope handler, or graph-store path
  responsible; do not add a shadow graph.
- Verify: readiness changes to graph-ready after one snapshot, reconnect rebuilds
  the graph, and hidden entities remain absent.

### Task 6 — Prove prepare and no-action conversation transport (10–15 min)

- Test first: run prepare, conversation correlation, timeout, disconnect, and
  no-action gateway/coordinator tests; then send one harmless state query.
- Minimal implementation: repair only the persistent WebSocket request/future
  path if needed. Keep per-turn HTTP out of `ConversationEntity`.
- Verify: `async_prepare()` fails closed when unavailable, one request receives
  one matching response, and no action request is emitted for a query.

Phase gate: a clean server/integration start reaches connected, graph-ready,
and model-ready state, and a harmless conversation completes over one
WebSocket.

## Phase 3 — Reassemble caller-authorized execution

### Task 7 — Prove source device and area resolution (10 min)

- Test first: run conversation/coordinator tests for HA device registry lookup,
  missing source IDs, stale areas, and explicit versus area-relative targets.
- Minimal implementation: fix only HA registry resolution or payload mapping if
  a check fails. Never restore an implicit `macbook` or living-room fallback.
- Verify: a known Mac device supplies its real HA area; a missing/unknown source
  stays missing and an area-relative command clarifies instead of acting.

### Task 8 — Prove caller context and permission enforcement (10–15 min)

- Test first: run result-correlation and permission tests for concurrent
  contexts, missing context, denied entities, cancellation, timeout, and
  disconnect cleanup.
- Minimal implementation: repair the existing correlation-ID context store or
  terminal cleanup path only. Never serialize HA `Context` into a WebSocket
  payload.
- Verify: the exact initiating context reaches `hass.services.async_call`, no
  service is called without it, and every terminal path empties retained state.

### Task 9 — Prove bounded action mapping and state verification (10–15 min)

- Test first: run action-mapping, capability, service, and state-verification
  tests for accepted, failed, rejected, unchanged, and timed-out actions.
- Minimal implementation: fix the shared semantic-action mapping or verifier if
  needed. Do not permit model-provided domains, services, or arbitrary data.
- Verify: an allowlisted explicit light command maps to one bounded HA service
  call and only reports success after the expected state is observed.

### Task 10 — Prove ambiguity and safety fail closed (10 min)

- Test first: run ambiguity, safety, exclusions, unsupported, pronoun/follow-up,
  and multi-target atomicity tests.
- Minimal implementation: fix the common resolver/validator barrier if any
  scenario emits an action. Do not special-case only the named phrase.
- Verify: ambiguous, hidden, incapable, malformed, unsupported, and partially
  invalid multi-target requests produce zero HA service calls.

Phase gate: an explicit action is caller-authorized and state-verified, while
all named refusal cases remain no-action.

## Phase 4 — Reassemble the Mac voice boundary

### Task 11 — Prove deterministic Assist replay through response parsing (10–15 min)

- Test first: run satellite Assist event-sequence and response tests using the
  committed PCM fixture, including failed STT and malformed/missing TTS output.
- Minimal implementation: repair only the existing Assist event parser or stage
  request. Keep Home Assistant responsible for STT and TTS.
- Verify: `--audio-file` reaches the SaySo agent, returns the expected intent
  result, and yields a TTS media reference or intentional earcon.

### Task 12 — Prove authenticated playback (10 min)

- Test first: run playback tests for authenticated media fetch, MIME handling,
  native player failure, and local earcon generation.
- Minimal implementation: repair the current URL/fetch/player adapter only; add
  no media framework.
- Verify: one real HA TTS response plays once through `afplay`, and an action
  earcon is audible without speaking the placeholder text.

### Task 13 — Prove bounded live microphone capture (10–15 min)

- Test first: run microphone/capture tests for 16 kHz mono PCM16, pre-roll,
  bounded utterance capture, invalid format, process failure, and cleanup.
- Minimal implementation: repair the existing macOS/ffmpeg source or capture
  composition only if the fake-to-live contract breaks.
- Verify: record one bounded live utterance, confirm non-empty correctly aligned
  PCM, and replay it successfully through Assist.

### Task 14 — Prove wake and continuous loop recovery (10–15 min)

- Test first: run wake/loop tests for threshold hits, no detection, retained
  pre-roll, one complete turn, exception recovery, and clean interruption.
- Minimal implementation: tune the existing threshold/hit calibration or fix
  the loop cleanup path. Do not replace the energy engine yet.
- Verify: one wake completes capture -> Assist -> playback and returns to
  listening; a no-wake interval produces no Assist request.

Phase gate: file replay, push-to-talk/live capture, wake, TTS/earcon playback,
and loop recovery all work on the target Mac.

## Phase 5 — Physical MVP acceptance

### Task 15 — Run one explicit physical-device turn (10–15 min)

- Test first: select one reversible, exposed, allowlisted light and record its
  initial HA state. The first end-to-end command is the acceptance check; if it
  fails, capture the failing stage before changing code.
- Minimal implementation: none unless the check exposes a reproducible defect.
  Any repair becomes its own approved TDD unit with a fake/local regression
  before touching the live device again.
- Verify: voice input changes only the selected device, telemetry contains the
  correlation/stage result, HA observes the expected state, and the Mac emits
  TTS or an earcon. Restore the light to its initial state through the same
  verified path.

### Task 16 — Run physical refusal and recovery turns (10–15 min)

- Test first: issue one ambiguous target, one unavailable/unknown target, and
  one valid command after a failed turn.
- Minimal implementation: none unless a case mutates state or poisons the next
  turn. A defect gets one colocated failing regression and a root-cause fix at
  the shared resolver, coordinator, or cleanup path.
- Verify: refusal cases make zero service calls, the valid follow-up succeeds,
  pending futures/contexts are empty, and the loop keeps listening.

### Task 17 — Run the basic evaluation gate (10–15 min)

- Test first: run the authored core, safety, follow-up, and language-noise
  corpus checks plus the dry-run execution safety gate.
- Minimal implementation: fix runtime behavior, not expected datasets or metric
  assertions. Do not expand corpora or start a model bake-off in this rebuild.
- Verify: the eval run is reproducible, live execution remains opt-in and
  allowlisted, false-execution cases remain zero-action, and a report is
  generated without committing local report output.

### Task 18 — Record the proven operator path (10 min)

- Test first: have a second clean shell follow only the documented commands;
  record any missing prerequisite, option, or readiness check.
- Minimal implementation: update `README.md` with the shortest working server,
  integration, satellite, and verification command sequence. Reference the
  architecture/eval docs instead of duplicating them; include no secrets or
  `context.json`.
- Verify: links resolve, commands parse, `git diff --check` passes, and the final
  full and eval suites are green.

## Final acceptance checklist

- [ ] Frozen dependency sync and package imports work from a clean environment.
- [ ] Main and eval test suites are green.
- [ ] Server readiness requires connection, graph, and resident model readiness.
- [ ] `ConversationEntity` uses the persistent WebSocket, not per-turn HTTP.
- [ ] No implicit Mac/living-room origin exists in the production Assist path.
- [ ] HA caller context stays inside HA and is required for execution.
- [ ] ControlPlan validation, ambiguity, capability, exposure, permission, and
      atomic multi-target barriers remain intact.
- [ ] Action-result waits are bounded/correlated and terminal state is cleaned.
- [ ] Successful actions are state-verified before success speech/earcon.
- [ ] Mac wake -> capture -> Assist -> action -> playback repeats after errors.
- [ ] One reversible real device succeeds; physical refusal cases do not act.
- [ ] Basic authored evals and dry-run safety gates remain runnable.
- [ ] No tuning, broad benchmark, new framework, or generalized satellite work
      entered the rebuild.

Plan ready. Approve to proceed.
