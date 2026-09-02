# SaySo architecture

Status: the Assist-to-WebSocket control path is implemented in code. The first
physical Mac demo and a live Home-LLM 270M bake-off are still outstanding.

SaySo is a Home Assistant conversation agent for a local voice assistant. Home
Assistant owns the voice session, device state, permissions, tool execution,
and response delivery. A resident SaySo server owns the tuned model and the
deterministic ControlPlan path. A Mac satellite is the temporary smart speaker.

The complete path is:

```text
Mac satellite (wake → live capture)
        ↓
Home Assistant Assist (STT, conversation routing, TTS)
        ↓
SaySo ConversationEntity (prepare + one correlated WebSocket turn)
        ↓
SaySo server (ControlPlan → resolve/validate/safety)
        ↓
action_request on the same WebSocket
        ↓
HA caller-context execute + state verification
        ↓
conversation_response → Assist TTS or local earcon → Mac playback
```

## Design goals

- Make a Mac or other supported device behave like a temporary Home Assistant
  voice satellite.
- Keep wake-word detection replaceable without changing conversation logic.
- Use the native Assist pipeline for STT, conversation routing, and TTS.
- Tune SaySo for the small, typed smart-home control vocabulary that it must
  handle well.
- Prevent model output from becoming an arbitrary Home Assistant service call.
- Make every mutating action explainable, permissioned, and state-verified.
- Keep the first physical-device demo local and small: one home, one pipeline,
  and one dependable control path.

## Architectural invariants

1. Home Assistant is the source of truth for entities, areas, capabilities,
   permissions, and live state.
2. The satellite and wake-word engine never choose or execute Home Assistant
   services.
3. Assist owns STT, conversation routing, and TTS. SaySo receives a transcript
   and returns spoken response content; it does not run an independent audio
   protocol.
4. The model emits a strict semantic `ControlPlan`. It does not emit raw
   Home Assistant service calls or use entity IDs as semantic targets.
5. Deterministic code parses, resolves, validates, and authorizes a plan before
   creating any HA tool call.
6. Ambiguous, unsupported, unresolved, hidden, incapable, malformed, or unsafe
   requests produce clarification or no-action and cannot mutate state.
7. Home Assistant remains the final execution boundary. It checks the tool
   call again, performs the service call, and verifies the resulting state.
8. Queries and clarifications do not invoke mutating tools.
9. The model is replaceable behind one narrow adapter; changing the model does
   not change safety or execution rules.
10. Production conversation turns use one authenticated persistent WebSocket.
    `POST /api/v1/text` remains evaluation and compatibility only.
11. The server never receives serialized Home Assistant authorization context.
    Execution uses the Assist caller context on the Home Assistant side.

## Component boundaries

| Component | Owns | Must not own |
|---|---|---|
| Mac satellite | Mic capture, replaceable wake engine, Assist PCM upload, local TTS/earcon playback | Intent parsing, entity selection, HA service calls |
| Wake-word engine | Detecting the configured wake threshold and starting a turn | Transcription, conversation decisions, device control |
| Assist pipeline | STT, conversation ID, TTS generation | SaySo-specific planning or direct model tool execution |
| SaySo `ConversationEntity` | `async_prepare()`, forwarding the transcript over WebSocket, mapping the response to Assist speech | Loading the model, resolving entities, or calling HA services |
| HA coordinator | Persistent SaySo WebSocket, graph/state push, caller-context action execution and verification | Planning or model inference |
| Resident SaySo server | Model runtime, ControlPlan parse/resolve/validate/safety, `action_request` / `conversation_response` | HA authorization context or direct HA service calls |
| Tuned SaySo model | Mapping natural language and supplied context to a typed plan | Inventing tools, choosing arbitrary entity IDs, or executing actions |
| Home Assistant | Registries, exposure, permissions, services, state changes | Depending on model output being safe |
| Evaluation and telemetry | Stage timings, outcomes, failure attribution, and regression checks | Live actuation by default |

The ConversationEntity is the only SaySo integration boundary exposed to Assist.
The model and validator run in the resident server. Neither receives authority
to call Home Assistant directly.

## Runtime topology

Three processes:

1. **sayso-satellite** on a Mac: `--live --wake --loop` listens, captures 16 kHz
   mono PCM16, sends it through Assist, and plays HA TTS or a local earcon.
   `--audio-file` remains deterministic debug replay.
2. **Home Assistant** with `custom_components/sayso`: registers the conversation
   agent, keeps one WebSocket to the server, executes validated actions with the
   Assist caller context, and verifies state.
3. **sayso-server**: resident MLX runtime, home graph from HA, ControlPlan path,
   and correlated WebSocket messages (`conversation_request`, `prepare`,
   `action_request` / `action_result`).

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
phonetic wake-word model. A successful detection captures one utterance with
pre-roll; no detection produces no Assist turn. Capture or Assist failures do
not continue a partial loop turn.

The satellite sends PCM to Home Assistant Assist. Assist handles STT.

### 2. Prepare and conversation

`ConversationEntity.async_prepare()` succeeds only when the coordinator
WebSocket is connected and a correlated `prepare` / `prepare_response`
exchange reports `connected`, `graph_ready`, and `model_ready`. Timeout or
negative readiness fails closed without accepting a transcript. The resident
server model is not reloaded per turn.

The entity then sends one `conversation_request` (transcript, optional
`device_id` / `satellite_id` / `area_id` resolved from HA registries). It does
not default those to `macbook` or living room. Missing IDs stay `None`;
area-relative plans without an origin return clarification.

### 3. SaySo planning

The server builds a bounded model input from the transcript, conversation
referents, the pushed home graph, and the ControlPlan schema. The model
returns one typed `ControlPlan`. Invalid or extra model output is rejected.
There is no generic agent loop and no speculative JSON repair.

### 4. Deterministic validation

The validator is the trust boundary between model output and Home Assistant.
For an action plan it validates schema and intent, resolves names/aliases/
areas against the exposed graph, applies ambiguity and capability rules, and
maps an approved semantic operation to a bounded `action_request`. Validation
is atomic: one invalid target blocks the whole plan.

Queries are read-only. Clarifications store only a short-lived referent.

### 5. HA execution and verification

Only validated `action_request` messages cross back into Home Assistant. The
coordinator executes with the Assist caller context, never a reconstructed
token. Action results use correlated futures (`asyncio.wait_for`), not busy
polling. State verification must observe the expected change or the turn is
not success.

The server turns the result into speech via response policy. Assist generates
TTS, or the satellite plays a local earcon for completed actions that do not
need spoken text.

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

SaySo does not maintain an authoritative shadow graph. If cached planning
context is stale or unavailable, it fails closed or asks for clarification.

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
| HA service failure | Failed response; never claim success |
| Verification timeout/unchanged state | Failed or unconfirmed response |
| Restart or lost connection | Fail pending turns; rebuild graph; fail closed until ready |

## Observability and evaluation

Telemetry identifies a conversation turn without storing raw audio by default.
Record correlation ID, pipeline/model revisions, outcome, selected target IDs,
validation reason, execution result, and stage timings.

Measure:

```text
wake → capture → Assist/STT → prepare → retrieve → plan → parse/validate
     → resolve → authorize → HA tool → verify → TTS/earcon → playback
```

Comparison records use shared EOS boundaries: EOS-to-plan, EOS-to-action-request,
and verified EOS-to-action. Cold model readiness is separate from warm turn
latency.

The evaluation harness uses authored cases and dry-run execution by default.
Live execution requires an explicit mode and entity allowlist. Core, safety,
follow-up, language-noise, and comparison corpora are runnable. The Home-LLM
270M comparison slot is pinned as an external fixture (`runtime=external`); it
is not a live downloaded bake-off.

See [EVALUATION_PLAN.md](EVALUATION_PLAN.md). When the model is tuned, follow
[TUNING_PLAN.md](TUNING_PLAN.md). Do not SFT on Home-LLM tool-call labels.

## Implementation map

| Concern | Status |
|---|---|
| Conversation WebSocket (`conversation_request` / `conversation_response`) | Implemented |
| `ConversationEntity` off per-turn HTTP | Implemented |
| HA-supplied area/device; no implicit `macbook` origin on the Assist path | Implemented |
| Caller-context HA execute; no serialized authorization | Implemented |
| Action-result futures (no busy polling) | Implemented |
| Satellite Assist through TTS, local earcon, Mac playback | Implemented |
| Live mic, energy wake engine, `--loop` | Implemented (fakes; physical demo not run) |
| `prepare` / `async_prepare()` before transcripts | Implemented |
| Comparison corpus, EOS timings, comparison report | Implemented (fixture Home-LLM, not live 270M) |
| Phonetic wake-word model | Not implemented (RMS energy prototype) |
| Live Home-LLM 270M bake-off | Not implemented |
| Physical Mac wake → speaker demo | Not run |
| Generalized multi-satellite support | Deferred |
| Default `macbook` satellite registration on the server | Compatibility leftover; Assist WS path does not use it as origin |

The primary production path is Assist → ConversationEntity → one correlated
WebSocket turn → ControlPlan → HA execute/verify → TTS/earcon. Direct text HTTP
and the default satellite registry are compatibility or evaluation surfaces,
not alternate authorities.

The numbered close-out plan is [WEBSOCKET_CONVERSATION_PLAN.md](WEBSOCKET_CONVERSATION_PLAN.md).
