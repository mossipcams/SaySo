# SaySo architecture boundary closure plan

## Decision

Keep SaySo's current architecture. SaySo already has the intended control
architecture: Home Assistant Assist owns the voice turn, the model produces one
bounded semantic `ControlPlan`, deterministic code resolves and validates it,
and Home Assistant executes and verifies the resulting action.

Do not add another HomeGraph, planner, generic tool loop, or model-generated
service-call path. The remaining work closes authority and transport gaps and
completes the physical voice path.

All nine units below are in scope. Their order is the implementation order, not
a deferral list. Project workflow still requires one numbered TDD task at a
time, with approval before implementation and confirmation before advancing.

## Architectural invariants

- Home Assistant Assist owns audio transport, STT, turn handling, conversation
  routing, and TTS generation.
- The SaySo server remains a separate, resident model and evaluation boundary.
- Home Assistant is authoritative for entities, exposure, permissions, caller
  context, areas, service execution, and resulting state.
- The server receives only the planning context it needs. It never receives or
  reconstructs Home Assistant authorization context.
- The model emits one semantic `ControlPlan`; it does not choose Home Assistant
  tools, services, or entity IDs.
- Existing ambiguity, capability, safety, execution, verification, metrics, and
  eval barriers remain intact.
- Production conversation turns use one authenticated persistent WebSocket.
  `POST /api/v1/text` remains only for evaluation and compatibility.

## Target turn flow

```text
Satellite
  -> Home Assistant Assist pipeline
  -> SaySo ConversationEntity
  -> conversation_request over the existing WebSocket
  -> SaySo model and deterministic ControlPlan path
  -> action_request over the same WebSocket
  -> HA authorization, validation, execution, and state verification
  -> action_result over the same WebSocket
  -> conversation_response over the same WebSocket
  -> Assist TTS
  -> satellite playback
```

## 1. Use one persistent WebSocket

### 1.1 Add conversation protocol messages

- Test: `conversation_request` and `conversation_response` validate through the
  existing envelope and preserve their correlation IDs.
- Code: add both message types to the existing server and integration protocol
  enums/constants.
- Verify: run the focused envelope and protocol tests.

### 1.2 Handle conversation requests in the server gateway

- Test: an inbound `conversation_request` invokes the existing text controller
  and produces exactly one correlated `conversation_response`.
- Test: malformed input and controller failure return a correlated fail-closed
  error without an action request.
- Code: route the request through the current controller, graph, ControlPlan,
  resolver, validation, action, verification, and response-policy path.
- Verify: run gateway tests for no-action and action turns.

### 1.3 Correlate responses in the HA coordinator

- Test: concurrent conversation requests receive only their matching responses.
- Test: timeout, disconnect, and coordinator shutdown fail every pending request
  and remove it from pending state.
- Code: add a pending `asyncio.Future` per conversation correlation ID and one
  `async_request_conversation()` method on the existing coordinator.
- Verify: run coordinator concurrency, timeout, and reconnect tests.

### 1.4 Move `ConversationEntity` off HTTP

- Test: `_async_handle_message()` calls the coordinator and never posts to
  `/api/v1/text`.
- Test: invalid, timed-out, or disconnected responses retain the existing
  no-action error behavior.
- Code: replace the entity's per-turn HTTP dependency with the coordinator
  request method.
- Verify: run conversation, coordinator, gateway, and full test suites.

Acceptance:

- A native Assist turn uses one WebSocket in both directions.
- Conversation, action, result, and response messages remain correlated.
- `POST /api/v1/text` is not called by the Home Assistant integration.

## 2. Preserve the initiating Home Assistant context

### 2.1 Retain context inside the integration

- Test: starting a turn stores the exact `user_input.context` object under the
  turn correlation ID.
- Test: concurrent turns keep distinct contexts.
- Code: store `correlation_id -> Context` in the HA coordinator before sending
  `conversation_request`.
- Verify: run focused coordinator context tests.

### 2.2 Attach context to service execution

- Test: a correlated `action_request` calls
  `hass.services.async_call(..., context=request_context, blocking=True)`.
- Test: an action with no active matching context is rejected and never calls a
  Home Assistant service.
- Code: look up the locally retained context by the turn correlation ID and pass
  it to the existing service caller. Do not put context fields in any WebSocket
  payload.
- Verify: run permission, action-mapping, and state-verification tests.

### 2.3 Expire context on every terminal path

- Test: completion, clarification, error, timeout, disconnect, and cancellation
  all remove the retained context.
- Code: perform cleanup in the coordinator's terminal `finally` path and when a
  connection closes.
- Verify: assert the context store is empty after every terminal-path test.

Acceptance:

- HA service calls retain the initiating caller and conversation context.
- Home Assistant `Context` never leaves Home Assistant.
- Stale or missing context cannot authorize an action.

## 3. Derive the origin area from Home Assistant

### 3.1 Resolve the source device and area

- Test: `ConversationInput.device_id` resolves through HA's device registry to
  its current area ID.
- Test: a valid Assist satellite/device identifier resolves to the same HA
  device and area.
- Test: a device with no area produces no fabricated origin.
- Code: resolve the source in the integration and include only the resolved HA
  `area_id` and stable source identifier in `conversation_request`.
- Verify: run conversation tests against fake HA device and area registries.

### 3.2 Validate the supplied area in the server

- Test: the server accepts a resolved area present in the current HomeGraph and
  rejects a stale or unknown area.
- Code: use the HA-supplied area directly for bounded planning; remove production
  dependence on the server's environment-based satellite-to-area mapping.
- Verify: run gateway, runtime, scope, and resolver tests.

### 3.3 Handle text chat without a source area

- Test: area-neutral explicit targets still resolve safely.
- Test: an area-relative request such as "turn off the lights" returns an area
  clarification when there is no device origin or valid conversation referent.
- Code: represent missing origin explicitly and use the existing clarification
  path instead of assigning a default room.
- Verify: run conversation, ambiguity, follow-up, and no-action tests.

Acceptance:

- Room-aware behavior comes from HA's current device registry.
- Text chat never silently inherits the living room.

## 4. Remove the implicit `macbook` fallback

### 4.1 Remove the production fallback

- Test: missing `satellite_id`/`device_id` is preserved as missing and never
  becomes `macbook`.
- Code: delete `user_input.satellite_id or "macbook"` from the production entity
  path. Permit a prototype default only behind explicit development
  configuration.
- Verify: search production code for implicit `macbook` fallback and run
  conversation tests.

### 4.2 Wire a stable device ID through the Mac client

- Test: `--device-id` and `SAYSO_HA_DEVICE_ID` reach the existing
  `assist_pipeline/run` `device_id` field, with the CLI value taking precedence.
- Test: absence leaves `device_id` unset.
- Code: expose the already-supported `device_id` argument in the satellite CLI
  and environment configuration. Do not create a general configuration system.
- Verify: run satellite CLI and Assist client tests, then perform one HA registry
  lookup with the configured Mac device ID.

Acceptance:

- The temporary Mac prototype has a stable HA device identity.
- No normal turn invents a satellite or living-room origin.

## 5. Replace action-result busy polling with futures

### 5.1 Add terminal action-result futures

- Test: accepted results remain interim; completed, failed, rejected, unchanged,
  or verification-timeout results resolve the matching request future.
- Test: results for one request cannot resolve another request.
- Code: replace the session's polling loop with one `asyncio.Future` per action
  request while retaining the existing result classification.
- Verify: run HA WebSocket client and result-correlation tests.

### 5.2 Bound and clean up waits

- Test: timeout, disconnect, cancellation, and session detach resolve or cancel
  pending futures and remove their entries.
- Code: use bounded `asyncio.wait_for()` waits and centralized terminal cleanup;
  remove the `await asyncio.sleep(0)` polling loop.
- Verify: run reconnect, timeout, and full orchestration tests.

Acceptance:

- No conversation or action result path busy-polls.
- Pending waits cannot leak across reconnects or turns.

## 6. Complete the Assist response audio path

### 6.1 Run Assist through TTS

- Test: the satellite requests `end_stage: "tts"` and accepts the ordered
  intent/TTS/run-end event sequence.
- Test: missing, malformed, or failed TTS output ends with a clear error rather
  than claiming playback.
- Code: extend the existing Assist client to parse the TTS output while retaining
  the transcript and intent result.
- Verify: run satellite Assist event-sequence tests.

### 6.2 Play HA-generated audio on the Mac

- Test: a returned TTS media reference is fetched with the required HA
  authentication and passed once to an injected audio player.
- Test: playback failure is reported and does not become a successful spoken
  turn.
- Code: add one Mac playback path for HA-generated response audio, using the
  native player where possible and keeping the player injectable for tests.
- Verify: run response tests and a manual speaker check against the configured HA
  TTS pipeline.

### 6.3 Preserve response semantics

- Test: clarification speech, errors, normal speech, and the current earcon
  response all produce the intended audible output; `\a` is not converted into
  spoken "Done." when an actual earcon is available.
- Code: map deterministic response policy output to HA speech or a local earcon
  without sending tool results back through the model.
- Verify: run the response-policy matrix and one end-to-end audible turn.

Acceptance:

- The pipeline no longer ends at `intent` and prints text as its final response.
- A successful physical turn produces audible HA TTS or an actual earcon.

## 7. Add a real wake engine and streaming microphone capture

### 7.1 Add live Mac microphone input

- Test: a fake live source yields fixed 16 kHz mono PCM16 chunks into the existing
  pre-roll and turn-capture code.
- Test: input failure, invalid format, and shutdown close the source cleanly.
- Code: add one Mac microphone source and CLI live mode. Keep prerecorded
  `--audio-file` as a deterministic test/debug mode.
- Verify: run capture tests and record one bounded live utterance.

### 7.2 Implement one concrete wake engine

- Test: the engine triggers only after the configured wake threshold and feeds
  the existing `WakeWordSession` without losing pre-roll.
- Test: false/no detection causes no Assist turn.
- Code: implement one configurable wake-engine adapter behind the existing
  `WakeWordEngine` protocol. Do not add generalized multi-satellite support.
- Verify: run fixture-based wake tests, then a live wake/no-wake smoke test.

### 7.3 Run the continuous satellite loop

- Test: wake -> capture -> Assist -> playback completes one turn and returns to
  listening; exceptions do not execute a partial turn.
- Code: connect the concrete mic, wake session, Assist client, and response player
  in a bounded loop with clean interruption.
- Verify: run the loop with fakes and then the first physical-device demo.

Acceptance:

- The executable can listen, wake, capture live speech, invoke Assist, and play
  the response repeatedly.
- File replay and manual capture remain available for deterministic diagnosis.

## 8. Implement `ConversationEntity.async_prepare()`

### 8.1 Add WebSocket readiness verification

- Test: a prepare request reports connected/graph-ready/model-ready state over
  the existing WebSocket and never executes an action.
- Code: expose the server's existing readiness state through a small correlated
  WebSocket prepare exchange.
- Verify: run readiness and gateway tests.

### 8.2 Prepare before transcript handling

- Test: `async_prepare(language)` succeeds only when the coordinator is connected
  and the server confirms the model is ready; timeout or negative readiness
  fails without accepting a transcript.
- Code: implement the Home Assistant method using the coordinator readiness
  exchange. Reuse the resident runtime; do not reload the model per turn.
- Verify: run conversation preparation, reconnect, and cold-start tests.

Acceptance:

- Model connectivity/readiness is checked before transcript handling.
- Integration reloads do not force the resident server model to reload.

## 9. Benchmark EOS-to-action against Home-LLM 270M

### 9.1 Add comparable benchmark cases

- Test: the benchmark dataset contains warm, cold, current-area, named-area,
  ambiguous-target, and multi-target cases with reviewed expected outcomes.
- Code: reuse the existing eval schema, corpus loader, metrics, and dry-run safety
  defaults; add only the minimal comparison cases and adapter configuration.
- Verify: run dataset/schema validation and hand-check the scored fixture.

### 9.2 Measure the same boundaries

- Test: both models report EOS-to-plan, EOS-to-action-request, and verified
  EOS-to-action latency with identical start/stop definitions.
- Code: record cold model readiness separately from warm turn latency and retain
  existing false-execution, wrong-device, resolution, and clarification scores.
- Verify: run seeded repeated trials and confirm complete timing fields.

### 9.3 Produce the comparison report

- Test: report generation rejects missing cases, unequal run counts, live
  actuation without an allowlist, and mixed timing definitions.
- Code: generate one reproducible report comparing SaySo's 230M model with the
  pinned Home-LLM 270M model across all six scenarios.
- Verify: rerun from a clean process and compare the generated summary with raw
  JSONL records.

Acceptance:

- The comparison covers accuracy, safety, warm latency, cold readiness, and
  verified action latency—not merely model token generation.
- Live execution remains opt-in and allowlisted.

## Final end-to-end gate

The plan is complete only when a Mac can repeatedly perform:

```text
wake
  -> live capture
  -> HA Assist STT
  -> SaySo ConversationEntity
  -> one correlated WebSocket turn
  -> ControlPlan
  -> deterministic resolution/validation/safety
  -> caller-context HA execution
  -> state verification
  -> HA TTS or real earcon
  -> Mac playback
```

The final verification must also prove:

- no implicit `macbook` or living-room origin in production;
- no serialized HA authorization context;
- no per-turn HTTP call from `ConversationEntity`;
- no action-result busy polling;
- no model-generated Home Assistant service call; and
- runnable core, safety, follow-up, and benchmark evals still pass.

