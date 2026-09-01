# SaySo architecture (as implemented)

This document describes the system **as it exists in this worktree**, not the
architecture implied by `docs/MVP_PLAN.md`. File paths are the source of truth.

**Snapshot.** Last committed product work is at `d1f03dc` (LFM JSON-only
generation instruction). **Uncommitted WIP** in this worktree:
`sayso-server/src/sayso_server/mlx_runtime.py`, `prompt.py`, `runtime.py`, and
matching tests — MLX chat-template few-shot wrapping for prompts built by
`build_lfm_prompt`, plus `GENERATION_INSTRUCTION` / `extract_lfm_prompt_user_json`.
Eval corpora through task 43 (follow-up, 70 cases) and tasks 44–45
(metric scorer, benchmark runner) are committed under `evals/`.

There is **no Dockerfile** and no single packaged launcher that starts Home
Assistant, the server, and the satellite together. Three separate processes
can be run manually: `python -m sayso_server`, the HA integration (when
installed), and `python -m sayso_satellite`.

**First physical-device demo status:** composition exists (shared graph,
live `action_request`, text client, plan pipeline), but a supervised
Mac → server → HA → device run **does not reliably succeed** today. Live
HA-connected text with the default `mlx-community/LFM2.5-230M-OptiQ-4bit`
checkpoint often ends in `model_output_invalid` after parsing, before
execution reaches Home Assistant.

---

## Component boundaries and responsibilities

Three packages plus an eval tree. They run in separate processes unless a test
fixtures them together.

| Component | Path | Responsibility today |
|---|---|---|
| Workspace | `pyproject.toml` | uv workspace named `sayso`; pytest paths; HA custom-component plugin |
| SaySo Server | `sayso-server/src/sayso_server/` | ControlPlan types, Home Graph store, retrieval/resolution/safety/orchestrator, HTTP/WS/audio surfaces, fake and MLX runtimes, STT contract |
| SaySo Satellite | `sayso-satellite/src/sayso_satellite/` | HTTP client for `/api/v1/text` and `/api/v1/audio`, push-to-talk capture helpers, response rendering |
| HA integration | `custom_components/sayso/` | Config entry, outbound WS to the server, Home Graph snapshot/deltas, inbound action execution, state verification, diagnostics |
| Evals | `evals/` | Fixture graph, JSONL corpora, `EvalCase` schema, corpus generators, metric scorer, benchmark runner |

`sayso-server` declares `aiohttp` and `pydantic` (`sayso-server/pyproject.toml`).
`mlx-lm`, `mlx-whisper`, and `numpy` are imported at runtime by MLX adapters
but are not package dependencies; the process entrypoint fails fast when
`mlx-lm` is missing.

---

## Runtime topology

Intended topology from `docs/MVP_PLAN.md`:

```text
Mac satellite ── text/audio ──▶ SaySo Server
                                    │
                         authenticated WebSocket
                                    │
                                    ▼
                         SaySo HA Integration
                                    │
                                    ▼
                              Home Assistant
```

**Implemented topology** when all three run:

```text
python -m sayso_satellite "turn on the lamp"
        │  POST /api/v1/text  (Bearer + satellite_id)
        ▼
create_aiohttp_app  (python -m sayso_server)
        ├── GET  /api/v1/health, /api/v1/ready
        ├── POST /api/v1/text   → OrchestratorTextController (default live wiring)
        ├── POST /api/v1/audio  → VoicePipelineController (STT → same text path)
        └── GET  /api/v1/ws     → handle_ha_connection on shared HomeGraphStore

SaySoConnectionCoordinator  (HA integration)
        │  outbound WS + Bearer token
        ▼
handle_ha_connection  →  graph_snapshot / state_delta / registry_delta
        │  action_request  ◀── HaWsActionClient (queued on session, flushed on receive loop)
        └── action_result  ──▶  session.record_action_result
```

`create_aiohttp_app` (`sayso-server/src/sayso_server/app.py`) is the primary
HTTP/WebSocket assembly. It registers default satellites (`macbook →
area_living_room`), attaches a shared `HomeGraphStore`, wires
`create_live_text_controller` with `BoundHaWsActionClient`, and binds the HA
WebSocket session through `HaGatewayBinding`.

`create_server()` in the same module is a legacy stdlib `ThreadingHTTPServer`
that only serves health and readiness. Config flow probes `GET /api/v1/health`
(`custom_components/sayso/config_flow.py`).

Process entrypoints:

- **Server:** `python -m sayso_server` (`sayso-server/src/sayso_server/__main__.py`)
  — loads `SAYSO_TOKEN`, builds resident MLX runtime via
  `build_mlx_runtime_for_server`, runs aiohttp on `SAYSO_HOST`/`SAYSO_PORT`
  (default `127.0.0.1:8765`).
- **Satellite:** `python -m sayso_satellite` — sends text or `--audio-file`
  PCM to the server using `SAYSO_SERVER_URL` and `SAYSO_TOKEN`.

---

## End-to-end command and response flow

### Live path (text → plan → HA WebSocket → device)

1. Satellite or curl POSTs a versioned text envelope to `/api/v1/text`
   (`sayso-server/src/sayso_server/text_api.py`).
2. `SatelliteRegistry.resolve_area_id` maps `satellite_id` to an area in the
   shared graph; refuses when graph missing or HA socket detached
   (`text_execution_refusal`).
3. `OrchestratorTextController.handle` / `handle_async` calls
   `compose_plan_generation` (`runtime.py`): retrieve candidates →
   `build_lfm_prompt` → `ModelRuntime.generate(prompt)` →
   `parse_model_output`.
4. `execute_control_plan` / `execute_control_plan_async` resolves entities
   (including ambiguity via `resolve_action_entities`), validates safety,
   sends one `action_request` through `BoundHaWsActionClient`, waits for
   correlated `action_result` payloads, classifies outcome
   (`orchestrator.py`).
5. Gateway receive loop records `action_result` on the session; outbound
   `action_request` envelopes drain on the next loop iteration
   (`gateway.py`, `ha_ws_client.py`).
6. HA integration executes the service call, verifies state, returns
   `completed` / `failed` (`custom_components/sayso/coordinator.py`).
7. Response includes `response_mode` / `response_content` from
   `resolve_response_policy` (earcon for completed actions).

### Audio path

`POST /api/v1/audio` validates 16 kHz mono PCM16, runs
`VoicePipelineController` (default: `MlxWhisperSttRuntime` → same text
controller), returns a `text_response` envelope (`audio_api.py`).

### Test-only shortcuts

- `FakeHaClient` records requests without a WebSocket (`ha_client.py`).
- `FakeModelRuntime.generate` parses the built LFM prompt JSON and returns a
  deterministic query-shaped JSON string for tests (`runtime.py`).
- HA integration tests inject `action_request` onto a fake WebSocket.

### What still fails in practice

Mac utterance → reliable ControlPlan → physical device change is blocked by
model output quality on the 230M checkpoint, not by missing glue between
server modules. Invalid JSON or schema drift becomes
`NoActionPlan(reason="model_output_invalid")` and never reaches HA.

---

## SaySo Server responsibilities

Implemented as library modules under `sayso-server/src/sayso_server/`:

- **Contracts:** `control_plan.py`, `models.py`, `envelope.py`, `messages.py`,
  `protocol.py`, `api.py`, `schema.py`
- **HTTP/WS assembly:** `app.py`, `const.py`, `auth.py`, `health.py`,
  `readiness.py`, `text_api.py`, `audio_api.py`, `gateway.py`, `session.py`
- **Home Graph:** `home_graph.py`, `graph_store.py`, `graph.py`, `deltas.py`
- **Language → plan:** `runtime.py`, `mlx_runtime.py`, `parser.py`, `prompt.py`
- **Deterministic control:** `candidates.py`, `scoring.py`, `normalize.py`,
  `ambiguity.py`, `scope.py`, `exclusions.py`, `resolver.py`, `capability.py`,
  `safety.py`, `queries.py`, `followups.py`, `orchestrator.py`, `results.py`
- **Live HA client:** `ha_client.py`, `ha_ws_client.py`
- **Voice:** `stt.py`, `mlx_stt.py`
- **Session helpers:** `conversation.py`, `satellites.py`, `telemetry.py`,
  `response_policy.py`

The server does **not** currently: persist state across restart, load config
from disk beyond environment variables, run STT/TTS playback on the server
host, or expose wake word / continuous listening.

---

## Satellite responsibilities

| Module | Role |
|---|---|
| `client.py` | Build/send text and audio envelopes; env: `SAYSO_TOKEN`, `SAYSO_SERVER_URL` |
| `__main__.py` | CLI: positional text or `--audio-file` PCM path |
| `capture.py` | Push-to-talk buffer, pre-roll, fixture mic source (no live CoreAudio mic driver) |
| `response.py` | Render earcon (`\a`) or short text from `text_response` policy fields |

Default `satellite_id` is `macbook` (matches server `DEFAULT_SATELLITE_ID`).
The satellite does not register itself over the network; server
`register_default_satellites` pre-seeds the registry at app construction.

---

## Home Assistant integration responsibilities

`custom_components/sayso/` is a `service` integration (`manifest.json`,
`iot_class: local_push`).

| Piece | File | What it does |
|---|---|---|
| Setup / teardown | `__init__.py` | Creates coordinator, forwards `binary_sensor`, registers `sayso.sync_home_graph` |
| Config | `config_flow.py` | URL + token; probes `/api/v1/health`; unique_id is the **URL** |
| Options | `config_flow.py` | Domain/action allowlists; exposure mode all / area / entity |
| Outbound WS | `coordinator.py` | Connect, hello, snapshot, deltas, heartbeat ping, action handling, bounded reconnect |
| Snapshot | `snapshot.py` | Floors, areas, devices, entities, scenes, scripts from HA registries |
| Deltas | `deltas.py` | One entity state or registry change per message |
| Exposure | `exposure.py` | Filter before snapshot/delta |
| Permissions | `permissions.py` | Exposure, domain match, allowlists, capability kind |
| Action map | `action_mapping.py` | Semantic action → HA domain/service/data |
| Results | `results.py` | `accepted` / `rejected` / `failed` / `completed` payloads |
| Verify | `state_verification.py` | Wait for `state_changed` or timeout |
| Device | `binary_sensor.py` | `SaySo Voice Assistant` device + connection entity |
| Diagnostics | `diagnostics.py` | Health, exposure counts, protocol; redacts `token` |
| Strings | `strings.json`, `translations/en.json`, `services.yaml` | UI copy and `sync_home_graph` |

The integration does **not** run a language model or Home Assistant Assist
conversation agent. It dials the SaySo server outbound.

---

## Direction and lifecycle of the Server ↔ Home Assistant connection

**Direction:** Home Assistant is the WebSocket **client**. SaySo Server listens
on `GET /api/v1/ws` (`WS_PATH` in `sayso-server/src/sayso_server/const.py` and
`custom_components/sayso/const.py`).

**Lifecycle (integration),** `SaySoConnectionCoordinator`:

1. `async_start` launches `_run` with reconnect (1s initial, factor 2, max 30s).
2. Convert configured HTTP(S) URL → `ws(s)://…/api/v1/ws`.
3. Connect with `Authorization: Bearer <token>`.
4. Send `hello`; wait for `hello_ack`; else close and raise.
5. Set `connected = True`; send `graph_snapshot`; start ping every 30s.
6. Receive loop: on `action_request`, dispatch; on socket close, exit.
7. `async_stop` cancels the runner, unsubscribes HA bus listeners, clears
   `connected`.
8. `sayso.sync_home_graph` pushes a fresh snapshot if the socket is up.

**Lifecycle (server),** `handle_ha_connection` (`gateway.py`):

1. Constant-time Bearer check; close on failure.
2. First message must be a valid `SaySoEnvelope` of type `hello`; else close.
3. Send `hello_ack` with the same `correlation_id`.
4. Attach `HaSession` to the **shared** `HomeGraphStore` passed from
   `create_aiohttp_app`; call `on_session_started` so `HaGatewayBinding` holds
   the live session and WebSocket.
5. Loop: accept `graph_snapshot` (replace), `state_delta`, `registry_delta`
   (sequence +1, same `home_id`); respond to `ping` with `pong`; record
   `action_result` on the session. Invalid JSON is skipped, not closed.
6. On exit: `HaGatewayBinding.detach`, `graph_store.clear()`,
   `readiness.set_ha_connected(False)`.

Readiness sets `ha_connected=True` only after a valid `graph_snapshot` is
applied (not merely after `hello_ack`). `model_ready` is never set by
`python -m sayso_server` today; it stays false unless a test or caller invokes
`ReadinessState.set_model_ready`.

---

## Authentication and permission model

### Shared secret

A single Bearer token:

- Config entry data: `url`, `token` (`custom_components/sayso/const.py`).
- Server: `SAYSO_TOKEN` env var via `load_server_token`.
- Unique id of the HA entry is the **URL**, not the token.

`bearer_token_valid` uses `hmac.compare_digest` for WS and POST `/api/v1/text`
and `/api/v1/audio`. `GET /api/v1/health` and `/api/v1/ready` use
`health_status`, which compares with `!=` (not constant-time).

Empty allowlists mean **allow all** listed option domains/actions.

### Two permission layers

1. **Server (deterministic controller):** ControlPlan validation, resolver
   empty-set / hidden-entity / pronoun / capability barriers. Blocks sending
   via safety outcome before `send_action_request`.
2. **Integration (authoritative execution):** exposure, domain match,
   allowlists, capability kind, then HA service call. Blocks calling Home
   Assistant even when the server sends a request.

There is no per-user identity, no satellite token distinct from the HA token,
and no mTLS.

---

## Home Graph ownership and synchronization

**Owner of physical truth:** Home Assistant registries and `hass.states`.

**Owner of the server’s working copy:** one shared `HomeGraphStore` inside the
aiohttp app. HA WebSocket ingest and `/api/v1/text` resolution read the same
object. Replace is atomic; deltas must be `sequence == current + 1` and the
same `home_id`.

**Producer:** `build_home_graph_snapshot` (`custom_components/sayso/snapshot.py`).
`home_id` is the HA config entry id. Sequence is a coordinator counter
starting at 0. Exposure filtering happens before serialize.

**Consumer:** `OrchestratorTextController` and `SatelliteRegistry.resolve_area_id`
read `app["graph_store"].snapshot`. When the HA socket drops, the gateway clears
the store and `text_execution_refusal` returns `ha_disconnected` or `no_graph`.

Area matching for the default Mac satellite accepts HA area ids or normalized
area names/aliases (`satellites.py`).

---

## ControlPlan schema and lifecycle

Discriminated union on `outcome` (`sayso-server/src/sayso_server/control_plan.py`):

| Outcome | Type | Role |
|---|---|---|
| `action` | `ActionPlan` | domain, optional scope, semantic `targets` / `include` / `exclude`, `state` / `value` / `mode` |
| `query` | `QueryPlan` | same targeting + optional `attribute` |
| `clarification` | `ClarificationPlan` | `reason` |
| `unsupported` | `UnsupportedPlan` | `reason` |
| `no-action` | `NoActionPlan` | `reason` |

`SemanticName` rejects Home Assistant `domain.object_id` strings (`models.py`).

**Lifecycle on the live text path:**

1. `compose_plan_generation` builds an LFM prompt via `build_lfm_prompt`
   (origin, conversation names/aliases, retrieved candidates — not raw entity
   IDs, not the full graph).
2. `ModelRuntime.generate(prompt)` returns raw text. `MlxModelRuntime` wraps
   the prompt in a chat template with a fixed few-shot example
   (`mlx_runtime.py`; uncommitted refinements add `GENERATION_INSTRUCTION`).
3. `parse_model_output` (`parser.py`): JSON or fenced JSON; invalid payloads
   become `NoActionPlan(reason="model_output_invalid")`. No speculative repair.
4. Orchestrator consumes the plan. Non-action outcomes become `NO_ACTION`
   after barriers. Queries go to `evaluate_query` and never call HA.
5. Successful actions may record last-target / last-intent on
   `ConversationStore` when one is wired.

---

## Entity retrieval, resolution, validation, execution, and verification

```text
utterance
  └─ compose_plan_generation
        retrieve_candidates → build_lfm_prompt → runtime.generate → parse_model_output
  └─ execute_control_plan[_async](plan, snapshot, …)
        resolve_follow_up          (when ConversationStore provided)
        resolve_action_entities    (scope + include/exclude + ambiguity margin)
        evaluate_safety_barrier
        send_action_request        (ONE entity: sorted(ids)[0])
        collect/take_action_results
        classify_action_results
```

| Stage | Module | Notes |
|---|---|---|
| Retrieval | `candidates.py`, `scoring.py` | O(n) scan; token/domain/area/floor/alias/capability/referent scores |
| Ambiguity | `ambiguity.py`, `resolver.py` | Score-margin rule inside `resolve_action_entities` on the orchestrator path |
| Scope | `scope.py` | `current_area`, `named_area`, `floor`, `all` |
| Include/exclude | `exclusions.py` | Name match inside scope; subtract exclusions |
| Capability | `capability.py` | Atomic reject of mixed invalid sets |
| Safety | `safety.py` | Barriers before send |
| Query | `queries.py` | Door open/closed; any/all lights on → yes/no |
| Execute | `orchestrator.py` | **First sorted entity only** (documented ceiling) |
| Verify (server) | `classify_action_results` | Expects `accepted` then `completed` |
| Verify (HA) | `state_verification.py` | `changed` / `unchanged` complete; timeout → `failed` |

Climate `set_temperature` / `mode` is mapped in HA (`action_mapping.py`) but
not emitted by the orchestrator’s `_semantic_action` today.

---

## Conversation-state ownership

`ConversationStore` (`conversation.py`) is in-memory, per `satellite_id`, with
TTL. Default live wiring in `create_live_text_controller` does **not** attach
one; follow-up pronoun resolution only runs when callers inject a store.

There is no persistence across process restart.

---

## Voice pipeline ownership

| Piece | Status |
|---|---|
| PCM transport (`/api/v1/audio`) | Implemented |
| Resident MLX Whisper STT adapter | Implemented (`mlx_stt.py`); optional install |
| VoicePipelineController → text path | Implemented |
| Push-to-talk capture helpers | Implemented in satellite (`capture.py`); no live mic CLI |
| Wake word, VAD, continuous loop | Not implemented |
| Server-side TTS / `say` / `afplay` | Not implemented (satellite renders earcon/text only) |

Telemetry `input_type` remains literal `"text"` in records even when the
request arrived via audio (`telemetry.py`).

---

## Public interfaces and versioned contracts

`API_VERSION = 1`, `PROTOCOL_NAME = "sayso-api"` (`api.py`).

### HTTP (aiohttp app)

| Method | Path | Auth | Behavior |
|---|---|---|---|
| GET | `/api/v1/health` | Bearer | Liveness 200 + `{status, liveness, model_ready, ha_connected}` |
| GET | `/api/v1/ready` | Bearer | 200 iff `model_ready and ha_connected`, else 503 |
| POST | `/api/v1/text` | Bearer | `TextRequestEnvelope` → `text_response` or `error` |
| POST | `/api/v1/audio` | Bearer | PCM envelope → `text_response` (voice pipeline) or `error` |
| GET | `/api/v1/ws` | Bearer | HA session |

Text request: `{version: 1, type: "text", correlation_id, payload: {satellite_id, text}}`.
Text response adds `response_mode`, `response_content`. Error codes include
`invalid_request`, `unknown_satellite`, `unknown_area`, `no_graph`,
`ha_disconnected`, `not_configured`.

### WebSocket envelope

Server `MessageType` (`messages.py`): `hello`, `hello_ack`, `ping`, `pong`,
`error`, `graph_snapshot`, `state_delta`, `registry_delta`, `action_request`,
`action_result`. All are in `MESSAGE_TYPES_V1`.

HA and server use the same envelope shape for graph and action messages.

JSON Schema fixtures: `evals/fixtures/sayso_api_v1.schema.json`,
`envelope.valid.json`, `home_graph.json`.

---

## State ownership and sources of truth

| State | Source of truth | Replica |
|---|---|---|
| Device/entity registry and HA state | Home Assistant | Shared `HomeGraphStore` on the server |
| Exposure / allowlists | Config entry options | Read at snapshot/delta/action time |
| WS connected | Coordinator `connected` | Binary sensor `SaySo Voice Assistant` / Connection |
| Server token | `SAYSO_TOKEN` env / caller | HA entry `data["token"]` |
| ControlPlan | Model output after parser | Orchestrator input |
| Conversation referents | `ConversationStore` (optional) | Prompt serialization of names/aliases only |
| Satellite → area | `SatelliteRegistry` + env override | Pre-registered at app start |
| Readiness | `ReadinessState` | Health/ready JSON |
| Eval cases | JSONL under `evals/datasets/` | Loaded by `evals/schema.py`, `evals/corpus.py`, `evals/runner.py` |

There is no database.

---

## Failure handling and restart behavior

| Event | What happens |
|---|---|
| Bad HA token / non-hello first message | Server closes WS; integration reconnects with backoff |
| HA disconnect | Integration `connected=False`; server clears graph, detaches binding, `ha_connected=False`; text path refuses with `ha_disconnected` / `no_graph` |
| Stale delta | `HomeGraphStore.apply_*` returns False; graph unchanged |
| Invalid ControlPlan JSON | `no-action` / `model_output_invalid` |
| Safety barrier | `ExecutionCategory.NO_ACTION`; no `send_action_request` |
| HA permission reject | `action_result` `rejected` before service call |
| Service exception | `failed` / `execution_failed` |
| State verification timeout | `failed` / `state_verification_timeout` (5s live) |
| Missing `accepted`+`completed` order | `incomplete_results` or `misordered_results` |
| Server process restart | Graph, satellites registry defaults, and readiness flags reset |
| HA restart | Coordinator reconnects; new snapshot after hello |
| Integration unload | Stop coordinator, remove service, unload platform |

`sayso.sync_home_graph` is the operator resync. Explicit “graph stale after
long disconnect” semantics beyond store clear + refusal are minimal.

---

## Current test boundaries

Colocated `test_*.py` only. Root `conftest.py` loads
`pytest_homeassistant_custom_component`. `pyproject.toml` `testpaths`:
`custom_components`, `sayso-server/src/sayso_server`,
`sayso-satellite/src/sayso_satellite`. There is no `tests/` directory.
`evals/` tests are run by path.

| Area | Tests | Bound |
|---|---|---|
| Satellite | `test_import.py`, `test_client.py`, `test_capture.py`, `test_main.py`, `test_response.py` | HTTP client, capture, CLI |
| ControlPlan / envelope | `test_control_plan.py`, `test_models.py`, `test_schema.py`, `test_envelope.py`, `test_messages.py`, … | Validation |
| Runtime / prompt | `test_runtime.py`, `test_mlx_runtime.py`, `test_parser.py`, `test_prompt.py` | `compose_plan_generation`, MLX chat template |
| Gateway / app | `test_gateway.py`, `test_app.py` | Shared graph, ping/pong, live wiring |
| Orchestrator / safety | `test_orchestrator.py`, `test_ambiguity.py`, … | Fake HA client and fixture graphs |
| Text / audio / telemetry | `test_text_api.py`, `test_audio_api.py`, `test_telemetry.py`, … | aiohttp test client |
| HA integration | `test_coordinator.py`, `test_snapshot.py`, … | pytest-homeassistant fakes |
| Evals | `test_schema.py`, corpus tests, `test_metrics.py`, `test_runner.py` | JSONL shape, scorer, dry-run runner — not live model or HA |

No integration test opens a real Mac mic, real MLX checkpoint, or real
Z-Wave/Zigbee device by default.

---

## Status taxonomy

### 1. Implemented now

- ControlPlan and SaySo v1 envelope types; JSON Schema helpers; full
  `MessageType` including `action_request` / `action_result`
- Shared `HomeGraphStore` wired from HA WebSocket into the text/audio path
- HA config flow, options, outbound WS, hello, snapshot, deltas, ping/pong,
  reconnect, `sync_home_graph`, action execution, state verification
- Live `HaWsActionClient` / `BoundHaWsActionClient` and async result collection
- `compose_plan_generation` + `build_lfm_prompt` on the default text controller
- Candidate retrieval, scope, include/exclude, ambiguity in resolver,
  capability validator, safety barriers, query evaluator, orchestrator
- Process entrypoints: `python -m sayso_server`, `python -m sayso_satellite`
- Default satellite registration; Mac text/audio HTTP client; response policy
- POST `/api/v1/audio` voice pipeline (STT → text controller)
- Strict parser → `model_output_invalid`
- Eval schema; core (120), safety (100), language-noise (150), follow-up (70)
  corpora; metric scorer; benchmark runner (dry-run executor)
- `MlxModelRuntime` and `MlxWhisperSttRuntime` load-once adapters
- `aiohttp` declared as a server dependency
- `AGENTS.md` voice-path priority

### 2. Partially implemented

- **Readiness:** `model_ready` never flipped on MLX load in `__main__`; `/ready`
  stays 503 even when the model and HA graph are usable
- **Model quality:** 230M LFM checkpoint frequently yields
  `model_output_invalid` on real-home text commands
- **Conversation store:** not attached in default `create_live_text_controller`
- **Orchestrator:** one entity per action; climate temperature/mode not mapped
- **Satellite voice:** capture helpers and PCM file upload; no live microphone
  driver or wake word
- **MLX deps:** runtime imports optional packages not listed in `pyproject.toml`
- **Auth:** constant-time on WS/text/audio; not on health/ready
- **Legacy HTTP server:** `create_server()` ThreadingHTTPServer still exists
- **Telemetry:** audio path not distinguished in `input_type`
- **Graph lifecycle:** store cleared on HA disconnect; no long-lived stale-graph
  flag beyond refusal codes

### 3. Planned but not implemented

From `docs/MVP_PLAN.md` / `docs/EVALUATION_PLAN.md`, absent or incomplete:

- Live model bake-offs (46–47), baseline imports (49–50), statistical report (51)
- Live dry-run allowlist gate wired to a real home (48) as default operator policy
- Wake word, VAD, endpoint state machine, continuous loop (58–64)
- macOS `say` / `afplay` playback on satellite (57) beyond earcon `\a`
- Per-satellite serialization (67), install docs (69), frozen benchmark (70)
- `training/` directory (listed in plan; not in tree)
- Home-FunctionGemma adapter, Home-LLM, Alexa+ importer
- Dockerfile / single-command stack launcher

### 4. Original requirements that conflict with the current implementation

- **Phase 3 gate “text commands operate real devices”:** wiring exists, but
  the default 230M model often fails before HA execution; demo success is not
  guaranteed.
- **Readiness equals runnable:** `model_ready` unused in startup despite MLX
  load in `__main__`.
- **Constant-time token check everywhere (task 25):** health/ready still use `!=`.
- **Satellite as hands-free smart speaker:** client + capture helpers only;
  no wake/VAD/playback stack.
- **Package deps match runtime:** MLX stack installed ad hoc by the environment.

### 5. Architectural decisions that remain unresolved

- When and how `model_ready` should flip true (load vs first successful generate)
- Whether default live wiring should attach `ConversationStore`
- Fake vs MLX vs recorded-plan runtime for demos while 230M quality is poor
- Multi-entity execution (orchestrator currently one ID)
- Climate `set_temperature` / `mode` in orchestrator vs HA mapping only
- Satellite auth vs shared HA Bearer token
- Live benchmark executor against real HA with task 48 guardrails
- Whether first demo standardizes on text-only or push-to-talk audio file

---

## Remaining plan: first physical-device demo vs deferrable

`AGENTS.md` asks to finish Mac → SaySo → Home Assistant → physical device
without cutting ControlPlan validation, ambiguity handling, integration
execution, verification, metrics, or basic evals.

A **first demonstration** means one Mac-originated command changes one real HA
entity through SaySo, with safety barriers still able to no-op. Voice wake and
model bake-offs are not required for that definition.

**The demo is not done.** Composition is largely in place; the blocking gap is
reliable ControlPlan generation from the resident 230M checkpoint on real-home
utterances.

### Required (still blocking or fragile)

1. **Reliable plan generation** on real commands (prompt/template/checkpoint
   work, or an interim runtime) so live text does not routinely end in
   `model_output_invalid`.
2. **`model_ready` wiring** so health/readiness reflect MLX load (or readiness
   semantics change to match what operators need).
3. **Operator runbook** for three processes (server env, HA integration config,
   satellite curl/CLI) — not packaged as one launcher.
4. **Optional but valuable for follow-ups:** wire `ConversationStore` into
   default live controller.
5. **Package-level safety that already exists** must stay: ControlPlan
   validation, capability/safety barriers, HA permission + state verification.
   Do not bypass them for the demo.

Optional for a *voice* demonstration of the same path: live mic capture on the
satellite, resident Whisper quality checks, and playback beyond earcon (tasks
53–57 family).

Task **48** (dry-run default + execute allowlist) is required before pointing
evals at a live home; it is not required for a supervised first demo on a
dedicated HA.

### Can be deferred without compromising safety or the core architecture

| Tasks | Why deferrable |
|---|---|
| **46–47, 49–51** | Extensive model benchmarking and baseline imports (`AGENTS.md` defer) |
| **58–64** | Wake/VAD/continuous loop — after push-to-talk or text demo |
| **67** | Per-satellite serialization — one Mac, one command |
| **68–70** | Security audit extras, install docs, frozen comparison report |
| Expanding corpora past committed sets | Large eval datasets |
| Generalized multi-satellite, fine-tuning, streaming, polished diagnostics | Explicitly deferred in `AGENTS.md` |
| Orchestrator multi-entity fan-out | Ceiling: first sorted id |
| Climate mode mapping | Not on the “turn on the lamp” path |

### Do not defer (even if “already coded”)

These must remain on the live path:

- ControlPlan validation and `model_output_invalid`
- Safety + capability barriers
- HA exposure/permissions and state verification
- Ambiguity rule when multiple equally scored devices match a single name
- Basic eval schema, corpora, scorer, and dry-run runner as regression fixtures
