# SaySo architecture

Status: current codebase architecture in this worktree, including the
reliability and evaluation changes that are still uncommitted.

This document describes structure and runtime behavior. It is not an
implementation plan or roadmap.

SaySo is a local smart-home voice path. A Mac satellite sends text or recorded
audio to a SaySo server. The server uses a local model only to produce a
typed `ControlPlan`; deterministic code resolves and validates that plan. The
Home Assistant integration remains the only component allowed to call Home
Assistant services and verify physical state.

The MVP is successful when one Mac can control a real Home Assistant device
through this path without Home Assistant Assist:

```text
Mac satellite ── HTTP text/audio ──▶ SaySo server
                                        │
                           authenticated WebSocket
                                        │
                                        ▼
                              SaySo HA integration
                                        │
                                        ▼
                                  Home Assistant
                                        │
                                        ▼
                                physical devices
```

## Architectural invariants

- Home Assistant owns physical state and service execution.
- The server may act only through a validated `ControlPlan` and the typed HA
  WebSocket action contract.
- Invalid, ambiguous, unsupported, unresolved, hidden, or incapable targets
  become no-action or clarification outcomes before execution.
- The integration enforces exposure, permissions, capability, and state
  verification again at the execution boundary.
- Text and audio use the same controller after transcription.
- The server keeps only an in-memory Home Graph replica; no database is
  required for the MVP.
- There is no generic agent loop and no arbitrary model-generated service call.

## Components and process boundaries

| Component | Location | Boundary and responsibility |
|---|---|---|
| Workspace | `pyproject.toml` | uv workspace, shared test configuration, HA test plugin |
| SaySo server | `sayso-server/src/sayso_server/` | API ingress, graph replica, model/STT adapters, deterministic controller, HA client, telemetry |
| SaySo satellite | `sayso-satellite/src/sayso_satellite/` | Mac-side text/audio client, PCM helpers, response rendering |
| HA integration | `custom_components/sayso/` | Outbound connection, graph serialization, exposure and permissions, HA execution, verification |
| Evaluation harness | `evals/` | Authored JSONL cases, dry-run execution, metrics, failure ledger, latency and gate reports |

These are separate processes at runtime. The default commands are:

```text
python -m sayso_server
python -m sayso_satellite "turn off the corner lamp"
```

Home Assistant loads `custom_components/sayso` as an integration. There is no
Dockerfile or single-process launcher.

## Ownership and sources of truth

| State | Authoritative owner | Server/integration copy |
|---|---|---|
| Entity registry and live device state | Home Assistant | `HomeGraphStore` on the server |
| Entity exposure and action allowlists | HA config-entry options | Applied before snapshots, deltas, and actions |
| Physical action | Home Assistant service layer | Typed `action_request` from the server |
| Action result | HA integration after execution and verification | Correlated `action_result` on the server |
| ControlPlan | Parsed model output | Input to the deterministic orchestrator |
| Conversation referents | In-memory `ConversationStore` | Per-satellite state with TTL |
| Satellite-to-area mapping | Server `SatelliteRegistry` | Default `macbook → area_living_room` |
| Readiness | Server `ReadinessState` | `/api/v1/health` and `/api/v1/ready` |
| Evaluation records | JSONL files under `evals/` | Reports and failure ledger |

The server graph is usable only after a valid HA snapshot. When the HA socket
disconnects, the server clears the graph and refuses live text execution until
the integration reconnects and sends a fresh snapshot.

## Server assembly and startup

`create_aiohttp_app` in `sayso-server/src/sayso_server/app.py` is the primary
assembly point. It creates or accepts:

- a `HomeGraphStore` shared by the WebSocket gateway and text/audio handlers;
- a `HaGatewayBinding` holding the current HA session and action client;
- the default satellite registry;
- a live `OrchestratorTextController` with `ConversationStore`, optional JSONL
  telemetry, the resident model runtime, and the bound HA action client;
- an MLX Whisper runtime for the audio endpoint;
- a `ReadinessState`.

`python -m sayso_server` loads `SAYSO_TOKEN`, builds the resident
`MlxModelRuntime`, creates the app, marks the model ready after a successful
load, best-effort preloads Whisper, marks `stt_ready` only when that preload
succeeds, and starts aiohttp on `SAYSO_HOST`/`SAYSO_PORT` (defaults
`127.0.0.1:8765`). Whisper readiness is informational and does not gate
`/api/v1/ready`; aggregate readiness is `model_ready and ha_connected`.

The legacy `create_server()` remains a stdlib health/readiness server used by
older tests and config-flow probing. It is not the live text/audio assembly.

## Public contracts

All envelopes use API version 1 and a non-empty `correlation_id`.
`API_VERSION = 1` and `PROTOCOL_NAME = "sayso-api"` are defined in
`sayso-server/src/sayso_server/api.py`.

### HTTP API

| Method and path | Auth | Contract |
|---|---|---|
| `GET /api/v1/health` | Bearer | Liveness plus model, HA, and STT status |
| `GET /api/v1/ready` | Bearer | 200 only when model and HA are ready; otherwise 503 |
| `POST /api/v1/text` | Bearer | `text` envelope to `text_response` or `error` |
| `POST /api/v1/audio` | Bearer | 16 kHz mono PCM16 to `text_response` or `error` |
| `GET /api/v1/ws` | Bearer | HA integration session |

Text input:

```json
{
  "version": 1,
  "type": "text",
  "correlation_id": "...",
  "payload": {"satellite_id": "macbook", "text": "turn on the lamp"}
}
```

Audio input carries `satellite_id`, a sequence, duration, format metadata, and
base64 PCM. The satellite defaults to a 180-second request timeout because a
cold STT/model path can exceed a short HTTP timeout; `timeout=` and
`SAYSO_TIMEOUT_SECONDS` override it.

### HA WebSocket messages

The v1 message types are:

```text
hello, hello_ack, ping, pong, error,
graph_snapshot, state_delta, registry_delta,
action_request, action_result
```

Graph and action payloads use the same versioned envelope. Action requests and
results carry a request identifier so concurrent protocol messages cannot be
cross-matched.

### ControlPlan

`control_plan.py` defines a discriminated union on `outcome`:

| Outcome | Meaning |
|---|---|
| `action` | Semantic domain, target names, scope, include/exclude, state/value/mode |
| `query` | Read-only state request |
| `clarification` | The user must disambiguate or provide missing context |
| `unsupported` | The request is outside the supported action vocabulary |
| `no-action` | Invalid, unsafe, unresolved, or otherwise blocked request |

Targets remain semantic names. `models.py` rejects Home Assistant entity IDs in
model-facing target fields. Only the resolver may produce entity IDs.

## End-to-end command paths

### Text command

```text
POST /api/v1/text
  → authenticate and validate envelope
  → map satellite_id to origin area
  → refuse if graph/HA session is unavailable
  → retrieve bounded candidates
  → build model prompt
  → resident model generates JSON
  → strict parser validates ControlPlan
  → resolve follow-up, scope, names, and exclusions
  → apply ambiguity, capability, and safety barriers
  → send one typed action request through HA WebSocket
  → receive accepted/completed or failure result
  → return response policy and telemetry
```

The HTTP handler is in `text_api.py`. Model generation is in `runtime.py`,
`prompt.py`, `parser.py`, and `mlx_runtime.py`. The deterministic control path
is in `orchestrator.py` and its resolver/validation modules.

### Recorded audio command

```text
POST /api/v1/audio
  → validate 16 kHz, mono, PCM16 framing and duration
  → load/reuse resident Whisper runtime
  → transcribe PCM and record STT timing
  → call the same text controller with input_type="audio"
  → follow the text command path above
```

The audio path does not have a separate language or execution stack. It is a
transport and STT front end to the text controller.

### Query, clarification, and follow-up

- Query plans use `queries.py` and never send an action request.
- Clarification, unsupported, invalid, and blocked plans are returned through
  `response_policy.py`; control actions normally produce an earcon and queries
  or errors produce short text.
- `ConversationStore` keeps last-target and last-intent references per
  satellite with a TTL. Follow-up resolution happens before normal target
  resolution. Expired or missing references clarify instead of guessing.

## Deterministic control path

The model supplies intent and semantic names. It does not select or execute a
raw entity ID. The server performs the following bounded stages:

1. `candidates.py` and `scoring.py` scan the in-memory graph using normalized
   names, aliases, inferred domain, area/floor, capability, state, and
   conversation referents. This is an O(n) MVP scan.
2. `scope.py` expands current-area, named-area, floor, or all-home scope.
3. `exclusions.py` resolves included and excluded names inside the scope and
   subtracts exclusions.
4. `ambiguity.py` applies the score-margin rule. Equal plausible matches
   clarify; they do not choose arbitrarily.
5. `capability.py` rejects targets that cannot perform the requested action
   or value/range operation. Mixed invalid sets are rejected atomically.
6. `safety.py` rejects empty, hidden, unresolved-pronoun, unsupported, or
   unknown-entity plans before the HA client is called.
7. `orchestrator.py` sends a typed request and classifies correlated results.

The current execution ceiling is one entity: the first sorted resolved ID is
requested. Multi-entity fan-out is deliberately not part of the live path.
Climate service mappings exist in the integration, but orchestrator emission
of climate temperature/mode remains incomplete.

Invalid model output is strict: parse or schema failure becomes
`NoActionPlan(reason="model_output_invalid")`. There is no speculative JSON
repair, unique-name bypass, or generic tool loop. The live LFM prompt is kept
small, with at most one retrieved candidate for the 230M checkpoint.

## Home Graph synchronization

The HA integration is the WebSocket client; the server listens on
`/api/v1/ws`.

### Integration lifecycle

`SaySoConnectionCoordinator`:

1. Connects to the configured server URL with `ws`/`wss` and Bearer auth.
2. Sends `hello` and waits for `hello_ack`.
3. Sends a filtered, serialized `graph_snapshot` and starts heartbeat pings.
4. Listens for `action_request` messages and dispatches approved actions.
5. Streams one `state_delta` or `registry_delta` per change.
6. Returns accepted, rejected, failed, or completed action results.
7. Reconnects with bounded exponential backoff after disconnect.

The integration serializes floors, areas, devices, entities, scenes, and
scripts. Entity area is `entry.area_id` or, when absent, the linked device
area. Exposure filtering occurs before snapshot and delta transmission.

### Server lifecycle

`gateway.py` authenticates the socket, requires a valid first `hello`, and
returns `hello_ack`. It then:

- replaces the shared graph on a valid snapshot;
- accepts only next-sequence state and registry deltas for the same `home_id`;
- responds to pings and records action results;
- marks HA ready only after a snapshot is applied;
- detaches and clears the graph on socket exit.

`HomeGraphStore` performs atomic snapshot replacement and rejects stale,
out-of-order, or wrong-home deltas without mutating the current graph.

## Execution boundary inside Home Assistant

The integration is authoritative even when the server is correct:

```text
action_request
  → request shape and entity validation
  → exposure/domain/action/capability permission checks
  → accepted result
  → fixed semantic action → HA domain/service/data mapping
  → service call
  → state_changed observation or timeout
  → completed / failed action_result
```

The integration never imports Home Assistant Assist conversation handling and
does not run a language model. Its `state_verification.py` distinguishes a
changed state, unchanged state, and timeout. Service exceptions become failed
results.

## Security and failure barriers

The current deployment uses one shared Bearer secret. It is configured as
`SAYSO_TOKEN` for the server and stored in the HA config entry and satellite
environment. WebSocket, text, and audio action surfaces use constant-time
Bearer comparison. Health/readiness use the existing health helper.

There is no user identity, per-satellite secret, mTLS, or cloud dependency.
The security model is therefore appropriate to a trusted local network, not a
publicly exposed server.

| Failure | Result |
|---|---|
| Missing/invalid request | HTTP error; no controller execution |
| Unknown satellite or area | Refusal; no action |
| No graph or disconnected HA socket | Refusal; no action |
| Invalid model JSON/schema | `no-action` / `model_output_invalid` |
| Ambiguous or incapable target | Clarification or no-action; no request |
| Hidden/disallowed entity | Server barrier or integration rejection |
| HA service exception | `failed` / `execution_failed` |
| Verification timeout | `failed` / `state_verification_timeout` |
| Server restart | Model/graph/session/readiness reset; HA reconnects and resyncs |
| HA restart/disconnect | Server clears graph and refuses until fresh snapshot |

Telemetry is JSONL when `SAYSO_TELEMETRY_PATH` is set. It records stage timing,
model metadata, input type, outcome, and identifiers without raw audio. The
failure-ledger stage vocabulary is:

```text
stt → retrieve → plan → parse → resolve → safety → request → verify
```

Evaluation executor/schema failures use the additional `schema` stage.

## Satellite and voice scope

The satellite is intentionally thin:

| Module | Responsibility |
|---|---|
| `client.py` | Build/send versioned text and PCM audio requests |
| `capture.py` | PCM framing, duration, push-to-talk/pre-roll helpers, fixtures |
| `response.py` | Earcon or short response rendering |
| `__main__.py` | Text CLI and `--audio-file` CLI |

Recorded-file voice is implemented. Live microphone capture, wake word, VAD,
continuous listening, TTS playback, echo cancellation, and barge-in are not.
The server does not play audio; the satellite renders the response policy.

## Evaluation architecture

`evals/` uses authored JSONL cases against a fixed synthetic graph. The
controller dry-run executor runs `FakeModelRuntime`, the same resolver and
validators, and `FakeHaClient`; it records no live HA execution. The default
runner is resumable, seeded, writes timing/config metadata, and is protected by
an `--execute` plus explicit entity-allowlist gate for any live executor.

The committed corpus currently includes core, safety, language-noise, and
follow-up cases. The scorer reports exact ControlPlan accuracy, candidate
recall, exact target resolution, wrong-device rate, false execution,
clarification, query/follow-up accuracy, and stage latency. The ledger assigns
each failure to one pipeline stage and preserves case IDs. The expansion gate
fails closed on false execution, wrong-device actions, missing timing samples,
or unclassified schema crashes.

The LFM configuration is reproducible and live MLX execution is optional.
Comparative model adapters and external baseline integrations are evaluation
components, not production dependencies.

## Current implementation boundaries

The supported runtime topology is one Mac satellite, one SaySo server, and one
Home Assistant integration session. Conversation state and the server graph
are in memory and reset on process restart. The orchestrator currently sends
one resolved entity per action, and climate temperature/mode plan emission is
incomplete.

The server and satellite do not provide live microphone capture, wake word,
VAD, continuous listening, server TTS, audio playback, a database, persistent
graph storage, a public-network identity model, or a packaged multi-process
launcher. These are boundaries of the current codebase, not alternate runtime
paths.

The supervised text path has exercised the full chain:

```text
Mac satellite → SaySo server → ControlPlan → HA integration → Home Assistant
→ physical plug-lamp state change
```

The working path includes ChatML few-shot wrapping for
`mlx-community/LFM2.5-230M-OptiQ-4bit`, switch-as-light retrieval, whole-home
retry for a named target missed in the origin area, HA permissions, and state
verification. MLX and Whisper packages are optional runtime imports rather than
fully pinned server package dependencies.

## Remaining plan and deferred work

This document describes structure and runtime behavior only. For what to build
next, use the companion plans below rather than treating gaps in this file as an
unordered backlog.

| Document | Scope |
|---|---|
| [MVP reliability and evaluation plan](MVP_RELIABILITY_AND_EVALUATION_PLAN.md) | Voice-path eval and reliability baseline: honest telemetry, failure ledger, latency reporting, expansion gate, and recorded-audio controller parity |
| [Architecture implementation plan](ARCHITECTURE_IMPLEMENTATION_PLAN.md) | Numbered-unit execution pointer for architecture-aligned MVP assembly and integration work |

Voice-path eval and reliability work lives in
`MVP_RELIABILITY_AND_EVALUATION_PLAN.md`. Numbered-unit execution order lives
in `ARCHITECTURE_IMPLEMENTATION_PLAN.md`.

The following items are explicitly deferred until the supervised text path,
recorded-file voice, and reliability baseline are trustworthy. They are not
current remaining work:

| Deferred item | Notes |
|---|---|
| Home-FunctionGemma bake-off | Comparative model evaluation; not a production dependency |
| Home-LLM bake-off | Comparative model evaluation; not a production dependency |
| Alexa+ bake-off | Comparative model evaluation; not a production dependency |
| Wake word, VAD, hands-free listening | Live microphone capture and continuous listening; recorded-file PCM voice is in scope today |

## Source map

| Concern | Primary files |
|---|---|
| App and readiness | `sayso-server/src/sayso_server/app.py`, `__main__.py`, `readiness.py` |
| HTTP text/audio | `text_api.py`, `audio_api.py`, `stt.py`, `mlx_stt.py` |
| Model and parsing | `runtime.py`, `mlx_runtime.py`, `prompt.py`, `parser.py` |
| Plan and safety | `control_plan.py`, `orchestrator.py`, `resolver.py`, `ambiguity.py`, `capability.py`, `safety.py`, `queries.py`, `followups.py` |
| Graph and gateway | `home_graph.py`, `graph_store.py`, `graph.py`, `gateway.py`, `session.py` |
| HA transport | `ha_ws_client.py`, `ha_client.py`, `custom_components/sayso/coordinator.py` |
| HA serialization/execution | `snapshot.py`, `deltas.py`, `exposure.py`, `permissions.py`, `action_mapping.py`, `state_verification.py` |
| Satellite | `sayso-satellite/src/sayso_satellite/` |
| Evaluation | `evals/schema.py`, `executor.py`, `runner.py`, `metrics.py`, `ledger.py`, `latency.py`, `report.py`, `gate.py` |
