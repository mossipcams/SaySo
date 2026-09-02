# SaySo architecture

Status: rebuild baseline. The production path is implemented; the physical
Mac demo has not been run. The numbered close-out is
[CLEAN_REBUILD_PLAN.md](CLEAN_REBUILD_PLAN.md).

SaySo is a Home Assistant conversation agent. Home Assistant owns the voice
session, entity truth, permissions, service execution, and state verification.
A resident SaySo server owns the model and the deterministic ControlPlan path.
A Mac satellite is the temporary smart speaker.

The repository already has the intended architecture. A rebuild assembles and
proves that path from a clean environment. It does not rewrite the validated
ControlPlan, resolution, safety, execution, verification, or evaluation
boundaries.

The complete path is:

```text
wake -> capture -> Home Assistant Assist STT -> SaySo ConversationEntity
     -> persistent WebSocket -> ControlPlan -> resolve/validate/safety
     -> caller-context HA execution -> state verification
     -> Assist TTS or local earcon -> Mac playback
```

The architecture is complete for the MVP only when this path controls an
allowlisted physical device and the core, safety, follow-up, and basic
evaluation gates remain green.

## Design goals

- Make a Mac behave like a temporary Home Assistant voice satellite.
- Keep the wake-word engine replaceable without changing conversation logic.
- Use the native Assist pipeline for STT, conversation routing, and TTS.
- Keep the model on a small, typed smart-home control vocabulary.
- Prevent model output from becoming a Home Assistant service call or an
  entity-ID authorization decision.
- Make every mutating action permissioned, explainable, and state-verified.
- Keep the first physical demo local and small: one home, one pipeline, one
  control path.

## Architectural invariants

1. Home Assistant is the source of truth for entities, areas, capabilities,
   exposure, permissions, caller context, service execution, and live state.
2. The satellite and wake-word engine never choose or execute Home Assistant
   services.
3. Assist owns STT, conversation routing, and TTS. SaySo receives a transcript
   and returns spoken response content; it does not run an independent audio
   protocol.
4. The model emits a typed semantic `ControlPlan`. It does not emit raw Home
   Assistant service calls or use entity IDs as semantic targets.
5. Deterministic code parses, resolves, validates, and authorizes a plan before
   any mutating request leaves the server.
6. Invalid, ambiguous, hidden, incapable, unsupported, or unsafe plans cannot
   mutate state. Validation is atomic: one invalid target blocks the whole
   plan.
7. Home Assistant remains the final execution boundary. It rechecks the
   request, performs the service call with the initiating Assist caller
   context, and verifies the resulting state.
8. Queries and clarifications do not invoke mutating tools.
9. The model is replaceable behind one narrow adapter. Changing the model does
   not change safety or execution rules.
10. Production conversation turns use one authenticated persistent WebSocket.
    `POST /api/v1/text` remains evaluation and compatibility only.
11. The server never receives serialized Home Assistant authorization context.
    The integration keeps the initiating HA `Context` locally.
12. Action success requires observed state verification. Unchanged, timed-out,
    rejected, or failed actions are not success.
13. The production Assist path has no implicit Mac or living-room origin.
    Missing source IDs stay missing; area-relative commands without an origin
    clarify instead of acting.

## Component boundaries

| Component | Owns | Must not own |
|---|---|---|
| Mac satellite | Mic capture, replaceable wake engine, Assist PCM upload, local TTS/earcon playback | Intent parsing, entity selection, HA service calls |
| Wake-word engine | Detecting the configured wake threshold and starting a turn | Transcription, conversation decisions, device control |
| Assist pipeline | STT, conversation ID, TTS generation | SaySo-specific planning or direct model tool execution |
| SaySo `ConversationEntity` | `async_prepare()`, one correlated WebSocket turn, mapping the response to Assist speech | Loading the model, resolving entities, calling HA services, or per-turn HTTP |
| HA coordinator | Persistent SaySo WebSocket, graph/state push, caller-context action execution and verification | Planning or model inference |
| Resident SaySo server | Model runtime, ControlPlan parse/resolve/validate/safety, correlated WebSocket messages | HA authorization context or direct HA service calls |
| Tuned SaySo model | Mapping natural language and supplied context to a typed plan | Inventing tools, choosing arbitrary entity IDs, or executing actions |
| Home Assistant | Registries, exposure, permissions, services, state changes | Depending on model output being safe |
| Evaluation and telemetry | Stage timings, outcomes, failure attribution, and regression checks | Live actuation by default |

The ConversationEntity is the only SaySo integration boundary exposed to Assist.
The model and validator run in the resident server. Neither receives authority
to call Home Assistant directly.

Reuse these packages instead of rebuilding them:

- `sayso-server`: ControlPlan schema/parser, graph, candidate retrieval,
  resolver, ambiguity, capability, safety, response policy, and telemetry.
- `custom_components/sayso`: graph snapshot/deltas, exposure, permissions,
  action mapping, and state verification.
- `sayso-satellite`: Assist client, PCM capture, wake session, response mapping,
  and playback adapters.
- `evals`: schemas, authored basic corpora, dry-run gate, metrics, runner, and
  reports.

Do not add a second home graph, a parallel authority path, a generic agent
loop, or model-provided domains, services, or arbitrary service data.

## Runtime topology

Three processes:

1. **sayso-satellite** on a Mac: `--live --wake --loop` listens, captures 16 kHz
   mono PCM16, sends it through Assist, and plays HA TTS or a local earcon.
   `--audio-file` remains deterministic debug replay.
2. **Home Assistant** with `custom_components/sayso`: registers the conversation
   agent, keeps one WebSocket to the server, executes validated actions with the
   Assist caller context, and verifies state.
3. **sayso-server**: resident model runtime, home graph from HA, ControlPlan
   path, and correlated WebSocket messages (`conversation_request`, `prepare`,
   `action_request` / `action_result`).

### Model runtime (`SAYSO_MODEL_ID`)

The resident server loads one MLX model at startup. Override the Hugging Face
model id with the `SAYSO_MODEL_ID` environment variable. When unset, the
default is `mlx-community/LFM2.5-230M-OptiQ-4bit` (see
`sayso_server.mlx_runtime.DEFAULT_MLX_MODEL_ID`).

`mlx-lm` is required on the host that runs `sayso-server` but is not in the
locked workspace dependencies. Install it separately, for example
`uv pip install mlx-lm`, before starting the server.

### Mac satellite device origin (`SAYSO_HA_DEVICE_ID`)

The Mac client may pass a Home Assistant **device registry** ID through
`SAYSO_HA_DEVICE_ID` or `--device-id`. The satellite forwards it on
`assist_pipeline/run`; Assist supplies `ConversationInput.device_id`; the
SaySo coordinator resolves it through HA's device registry to
`(source_id, area_id)` for the planning payload. There is no implicit Mac or
living-room fallback.

- **Optional** for explicit named targets (“turn on the floor lamp”).
- **Required** for area-relative origin (“turn off the lights in here”) so the
  validator receives the Mac's current area.

Obtain a device registry ID from Home Assistant (use any device assigned to the
Mac's room; entity IDs are not valid here):

1. **Device page URL** — **Settings → Devices & services**, open the device,
   copy the final path segment from `/config/devices/device/<device_id>`.
2. **Temporary automation YAML** — add a device trigger for that hardware,
   choose **Edit in YAML**, and read the `device_id` field.
3. **Advanced** — inspect `$HA_CONFIG/.storage/core.device_registry` (or your
   install's equivalent config path).

Assign the device to an area in Home Assistant; otherwise area-relative
commands clarify instead of acting.

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
  -> Assist TTS or local earcon
  -> satellite playback
```

### 1. Wake and capture

The satellite listens through a replaceable `WakeWordEngine`. The shipped
engine is an energy/RMS prototype (`EnergyThresholdWakeEngine`), not a
phonetic wake-word model. It is sufficient for the first physical demo.

A successful detection captures one bounded utterance with pre-roll. No
detection produces no Assist turn. Capture or Assist failures do not continue
a partial loop turn. After a completed or failed turn the loop returns to
listening.

The satellite sends PCM to Home Assistant Assist. Assist handles STT.

### 2. Prepare and conversation

`ConversationEntity.async_prepare()` succeeds only when the coordinator
WebSocket is connected and a correlated `prepare` / `prepare_response`
exchange reports `connected`, `graph_ready`, and `model_ready`. Timeout or
negative readiness fails closed without accepting a transcript. The resident
server model is not reloaded per turn.

The entity then sends one `conversation_request` (transcript, optional
`device_id` / `satellite_id` / `area_id` resolved from HA registries). It does
not default those to `macbook` or living room. A known Mac device supplies its
real HA area. A missing or unknown source stays missing.

There is one request and one matching response. No action request is emitted
for a query. The entity does not use per-turn HTTP.

### 3. SaySo planning

The server builds a bounded model input from the transcript, conversation
referents, the pushed home graph, and the ControlPlan schema. The model
returns one typed `ControlPlan`. Invalid or extra model output is rejected.
There is no generic agent loop and no speculative JSON repair.

### 4. Deterministic validation

The validator is the trust boundary between model output and Home Assistant.
For an action plan it validates schema and intent, resolves names/aliases/
areas against the exposed graph, applies ambiguity and capability rules, and
maps an approved semantic operation to a bounded `action_request`. Hidden
entities remain absent from the planning graph. Validation is atomic.

Queries are read-only. Clarifications store only a short-lived referent.

### 5. HA execution and verification

Only validated `action_request` messages cross back into Home Assistant. The
coordinator executes with the exact initiating Assist caller context. No
service is called without it. Action results use correlated futures
(`asyncio.wait_for`), not busy polling. Every terminal path empties retained
context and pending futures.

State verification must observe the expected change or the turn is not
success. The server turns the result into speech via response policy. Assist
generates TTS, or the satellite plays a local earcon for completed actions
that do not need spoken text.

## State and authority

| State | Authority | SaySo access |
|---|---|---|
| Entity and device registry | Home Assistant | Pushed graph for planning; HA resolves area/device for the turn |
| Exposure and permissions | Home Assistant configuration | Read-only during planning; enforced again at execution |
| Current device state | Home Assistant state machine | Read-only query and verification input |
| Conversation session | Assist conversation context | Bounded referents keyed by satellite/conversation |
| ControlPlan | SaySo model output after strict parsing | Never treated as authorization |
| Tool call | Server validator output, rechecked by HA | No direct service authority |
| Physical state change | Home Assistant and the device | Observed by verification |
| Assist caller `Context` | Home Assistant | Kept locally in the integration; never serialized to the server |

SaySo does not maintain an authoritative shadow graph. If cached planning
context is stale or unavailable, it fails closed or asks for clarification.
Reconnect rebuilds the graph from one snapshot plus deltas.

## Replaceable interfaces

- `WakeWordEngine`: emits wake events from PCM chunks.
- `MicSource` / `AudioPlayer`: Mac capture and playback adapters.
- `ConversationEntity`: Assist conversation input and result.
- `SaySoModel`: normalized transcript/context to typed model output.
- WebSocket envelope (`version`, `type`, `correlation_id`, `payload`): the only
  production conversation transport.

Wake-word engines and model runtimes can be swapped independently. The
ControlPlan, validator, HA tool contract, and evaluation cases remain stable
across those substitutions.

## Security and failure behavior

All external input is validated at its boundary. Model output is untrusted
input. Credentials and local transport security follow Home Assistant's
configuration.

| Failure | Behavior |
|---|---|
| Wake engine unavailable / no detection | No Assist turn; loop keeps listening |
| STT/Assist failure | No model call; concise error response |
| Prepare timeout or not ready | No transcript accepted |
| Invalid model output | No-action; record a parse/schema failure |
| Ambiguous target | Clarification; no tool call |
| Unsupported or incapable request | No-action or clarification; no tool call |
| Hidden/disallowed entity | Rejection at validation or HA execution; no mutation |
| Missing/unknown source on an area-relative command | Clarification; no tool call |
| HA service failure | Failed response; never claim success |
| Verification timeout/unchanged state | Failed or unconfirmed response |
| Restart or lost connection | Fail pending turns; rebuild graph; fail closed until ready |

Server readiness requires connection, graph, and resident model readiness.
Liveness is separate from readiness. Readiness fails closed until both the
graph and the model are ready.

## Observability and evaluation

Telemetry identifies a conversation turn without storing raw audio by default.
Record correlation ID, pipeline/model revisions, outcome, selected target IDs,
validation reason, execution result, and stage timings.

Measure:

```text
wake -> capture -> Assist/STT -> prepare -> retrieve -> plan -> parse/validate
     -> resolve -> authorize -> HA tool -> verify -> TTS/earcon -> playback
```

The evaluation harness uses authored cases and dry-run execution by default.
Live execution requires an explicit mode and entity allowlist. Core, safety,
follow-up, and language-noise corpora plus the dry-run execution safety gate
stay intact. False-execution cases remain zero-action.

See [EVALUATION_PLAN.md](EVALUATION_PLAN.md). When the model is tuned, follow
[TUNING_PLAN.md](TUNING_PLAN.md). Do not SFT on Home-LLM tool-call labels or on
`evals/datasets/` case IDs. Do not commit `context.json` or local eval report
output.

## Explicitly deferred

These are out of the rebuild and out of the first physical demo:

- Phonetic wake-word replacement. Add it after the energy detector proves the
  loop and false wakes become the measured bottleneck.
- Fine-tuning. Follow `TUNING_PLAN.md` only after the physical voice path and
  frozen eval gate work.
- Live Home-LLM 270M bake-off, broader model benchmarking, larger corpora,
  generalized multi-satellite support, streaming optimization, and polished
  diagnostics.
- New frameworks, dependency changes, parallel authority paths, or a second
  home graph.

The primary production path is Assist -> ConversationEntity -> one correlated
WebSocket turn -> ControlPlan -> HA execute/verify -> TTS/earcon. Direct text
HTTP and any leftover default satellite registry are compatibility or
evaluation surfaces, not alternate authorities.

## Rebuild completeness

The rebuild is the current close-out. It restores a reproducible baseline,
then reassembles the server, caller-authorized execution, and Mac voice
boundary without rewriting the validated core. Numbered units and stop points
live in [CLEAN_REBUILD_PLAN.md](CLEAN_REBUILD_PLAN.md).

The architecture is accepted when:

- Frozen dependency sync and package imports work from a clean environment.
- Main and eval test suites are green.
- Server readiness requires connection, graph, and resident model readiness.
- `ConversationEntity` uses the persistent WebSocket, not per-turn HTTP.
- No implicit Mac/living-room origin exists in the production Assist path.
- HA caller context stays inside HA and is required for execution.
- ControlPlan validation, ambiguity, capability, exposure, permission, and
  atomic multi-target barriers remain intact.
- Action-result waits are bounded/correlated and terminal state is cleaned.
- Successful actions are state-verified before success speech/earcon.
- Mac wake -> capture -> Assist -> action -> playback repeats after errors.
- One reversible real device succeeds; physical refusal cases do not act.
- Basic authored evals and dry-run safety gates remain runnable.
- No tuning, broad benchmark, new framework, or generalized satellite work
  entered the rebuild.
