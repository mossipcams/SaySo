# SaySo architecture (as implemented at step 43)

This document describes the system **as it exists in this worktree**, not the
architecture implied by `docs/MVP_PLAN.md`. File paths are the source of truth.

**Snapshot.** Last committed product work is task 42 (language-noise corpus)
plus `AGENTS.md`, at `cb53137`. Tasks 1–38 (libraries and tests) and 39–42
(eval schema and corpora) are committed. Task 43 (follow-up corpus) is
**uncommitted eval-only WIP**: `evals/test_followup_corpus.py`,
`evals/datasets/followup.jsonl` (70 cases), and edits to `evals/corpus.py`.
It does not change runtime behavior.

There is no process entrypoint, no Dockerfile, and no running composition that
wires the server, a Mac satellite, and Home Assistant together.

---

## Component boundaries and responsibilities

Three packages plus an eval tree. They do not share a runtime process.

| Component | Path | Responsibility today |
|---|---|---|
| Workspace | `pyproject.toml` | uv workspace named `sayso`; pytest paths; HA custom-component plugin |
| SaySo Server | `sayso-server/src/sayso_server/` | ControlPlan types, Home Graph store, retrieval/resolution/safety/orchestrator, HTTP/WS surfaces, fake and MLX runtimes |
| SaySo Satellite | `sayso-satellite/src/sayso_satellite/` | Importable package with a version string |
| HA integration | `custom_components/sayso/` | Config entry, outbound WS to the server, Home Graph snapshot/deltas, inbound action execution, state verification, diagnostics |
| Evals | `evals/` | Fixture graph, JSONL corpora, `EvalCase` schema, corpus generators |

`sayso-server` declares only `pydantic` (`sayso-server/pyproject.toml`).
`aiohttp` and `mlx-lm` are imported by server modules but are not package
dependencies. Tests obtain `aiohttp` transitively via
`pytest-homeassistant-custom-component` at the workspace root.

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

**Implemented topology:** two independently testable halves. They are not
joined in a live process.

```text
pytest / aiohttp test client
        │
        ▼
sayso_server.app.create_aiohttp_app
        ├── GET  /api/v1/health, /api/v1/ready
        ├── POST /api/v1/text     (503 unless a TextController is injected)
        └── GET  /api/v1/ws       (HA handshake + graph ingest on a session-local store)

Home Assistant (if the custom component is installed)
        │  outbound WS + Bearer token
        ▼
SaySoConnectionCoordinator  ──hello──▶  handle_ha_connection
        │  graph_snapshot / state_delta / registry_delta
        │  listens for action_request  (never sent by the server today)
        └── executes HA services when a test or future peer injects action_request
```

`create_server()` in `sayso-server/src/sayso_server/app.py` is a second,
stdlib `ThreadingHTTPServer` that only serves health and readiness. Config
flow probes `GET /api/v1/health` (`custom_components/sayso/config_flow.py`).
Nothing in-repo starts either HTTP server as an application.

The Mac satellite does not open sockets, capture audio, or call the text API.

---

## End-to-end command and response flow

### Path that exists in tests (text → plan → fake HA)

1. Caller constructs `OrchestratorTextController` with a `ModelRuntime`,
   `ActionRequestClient`, and `HomeGraphStore`
   (`sayso-server/src/sayso_server/text_api.py`).
2. `runtime.generate_plan(text)` emits a validated ControlPlan.
3. `execute_control_plan()` resolves, validates, optionally sends one action
   request, and classifies results
   (`sayso-server/src/sayso_server/orchestrator.py`).
4. `FakeHaClient` records the request and returns queued results
   (`sayso-server/src/sayso_server/ha_client.py`). No WebSocket is involved.

`FakeModelRuntime.generate_plan` always returns a **query** plan with
`domain="light"` (`sayso-server/src/sayso_server/runtime.py`). It cannot
produce an action for “turn on the lights.”

### Path that exists in HA tests (injected `action_request` → physical service)

1. `SaySoConnectionCoordinator._receive_loop` accepts an envelope whose
   `type` is `action_request` (`custom_components/sayso/coordinator.py`).
2. Permission check → `accepted` result → `map_action_to_ha_service` →
   `hass.services.async_call` → state verification → `completed` / `failed`.
3. Tests inject the request onto a fake WebSocket. The server never sends
   this message type (`sayso-server/src/sayso_server/test_messages.py`
   asserts `action_request` / `action_result` are **not** in `MESSAGE_TYPES_V1`).

### Path that does not exist

Mac utterance → STT → LFM prompt with retrieved candidates → ControlPlan →
shared Home Graph → `action_request` on the HA WebSocket → device state
change → text/TTS response.

`create_aiohttp_app` also does **not** attach `OrchestratorTextController`
unless the caller passes `text_controller=`. Default POST `/api/v1/text`
returns HTTP 503 `not_configured`.

---

## SaySo Server responsibilities

Implemented as library modules under `sayso-server/src/sayso_server/`:

- **Contracts:** `control_plan.py`, `models.py`, `envelope.py`, `messages.py`,
  `protocol.py`, `api.py`, `schema.py`
- **HTTP/WS assembly:** `app.py`, `const.py`, `auth.py`, `health.py`,
  `readiness.py`, `text_api.py`, `gateway.py`, `session.py`
- **Home Graph:** `home_graph.py`, `graph_store.py`, `graph.py`, `deltas.py`
- **Language → plan:** `runtime.py`, `mlx_runtime.py`, `parser.py`, `prompt.py`
- **Deterministic control:** `candidates.py`, `scoring.py`, `normalize.py`,
  `ambiguity.py`, `scope.py`, `exclusions.py`, `resolver.py`, `capability.py`,
  `safety.py`, `queries.py`, `followups.py`, `orchestrator.py`, `results.py`
- **Session helpers:** `conversation.py`, `satellites.py`, `ha_client.py`,
  `telemetry.py`

The server does **not** currently: persist state, load config, register
satellites from disk, send `action_request`, apply HA graph updates into the
store used by `/api/v1/text`, run STT/TTS, or expose a CLI.

---

## Satellite responsibilities

`sayso-satellite/src/sayso_satellite/__init__.py` exports `__version__ = "0.1.0"`.
`sayso-satellite/src/sayso_satellite/test_import.py` only checks that import.

`SatelliteRegistry` lives on the **server**
(`sayso-server/src/sayso_server/satellites.py`). It is an in-memory map of
`satellite_id → area_id` used by the text API to reject unknown satellites
and areas. Nothing registers `macbook → living_room` at process start.

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

The integration does **not** run a language model, Assist, or conversation
agent. It does not open an inbound server port; it dials the SaySo server.

---

## Direction and lifecycle of the Server ↔ Home Assistant connection

**Direction:** Home Assistant is the WebSocket **client**. SaySo Server is the
**listener** on `GET /api/v1/ws` (`WS_PATH` in
`sayso-server/src/sayso_server/const.py` and
`custom_components/sayso/const.py`).

**Lifecycle (integration),** `SaySoConnectionCoordinator`:

1. `async_start` launches `_run` with reconnect
   (`RECONNECT_INITIAL_DELAY` 1s, factor 2, max 30s).
2. Convert configured HTTP(S) URL → `ws(s)://…/api/v1/ws`.
3. Connect with `Authorization: Bearer <token>`.
4. Send `hello`; wait for `hello_ack`; else close and raise.
5. Set `connected = True`; send `graph_snapshot`; start ping every 30s.
6. Receive loop: on `action_request`, dispatch; on socket close, exit.
7. `async_stop` cancels the runner, unsubscribes HA bus listeners, clears
   `connected`.
8. `sayso.sync_home_graph` pushes a fresh snapshot if the socket is up.

**Lifecycle (server),** `handle_ha_connection`
(`sayso-server/src/sayso_server/gateway.py`):

1. Constant-time Bearer check; close on failure.
2. First message must be a valid `SaySoEnvelope` of type `hello`; else close.
3. Send `hello_ack` with the same `correlation_id`.
4. Loop: accept `graph_snapshot` (replace), `state_delta`, `registry_delta`
   (sequence +1, same `home_id`). Invalid JSON is skipped, not closed.
5. `ping` is a registered message type but is **not** handled (no `pong`).
6. `HaSession` holds a **private** `HomeGraphStore`. `create_aiohttp_app`
   discards the returned session. The app-level `graph_store` used by
   `/api/v1/text` is never updated from HA.

Readiness marks HA connected when the server **sends** `hello_ack`, and
clears it on WebSocket close (`app.py` `_ReadinessTrackingGatewayWebSocket`).
`model_ready` is never set by any startup path; it stays false unless a test
calls `ReadinessState.set_model_ready`.

---

## Authentication and permission model

### Shared secret

A single Bearer token:

- Config entry data: `url`, `token` (`custom_components/sayso/const.py`).
- Server: passed into `create_aiohttp_app` / `create_server`.
- Unique id of the HA entry is the **URL**, not the token
  (`config_flow.py` `async_set_unique_id(url)`).

`bearer_token_valid` uses `hmac.compare_digest`
(`sayso-server/src/sayso_server/auth.py`) for WS and POST `/api/v1/text`.

`GET /api/v1/health` and `/api/v1/ready` use `health_status`, which compares
with `!=` (`sayso-server/src/sayso_server/health.py`). That is not
constant-time.

Empty allowlists mean **allow all** listed option domains/actions
(`validate_action_permission` in `permissions.py`).

### Two permission layers

1. **Server (deterministic controller):** ControlPlan validation, resolver
   empty-set / hidden-entity / pronoun / capability barriers
   (`safety.py`, `capability.py`). Blocks *sending* a request via
   `FakeHaClient`.
2. **Integration (authoritative execution):** exposure, domain match,
   domain/action allowlists, capability kind, then HA service call
   (`permissions.py`, `action_mapping.py`). Blocks *calling* Home Assistant.

There is no per-user identity, no satellite token distinct from the HA token,
and no mTLS.

---

## Home Graph ownership and synchronization

**Owner of physical truth:** Home Assistant registries and `hass.states`.

**Owner of the server’s working copy:** `HomeGraphSnapshot`
(`sayso-server/src/sayso_server/home_graph.py`) inside a `HomeGraphStore`
(`graph_store.py`). Replace is atomic. Deltas must be `sequence == current + 1`
and the same `home_id`. Stale or foreign-home deltas are rejected.

**Producer:** `build_home_graph_snapshot` (`custom_components/sayso/snapshot.py`).
`home_id` is the HA **config entry id**. Sequence is a coordinator counter
starting at 0, incremented per snapshot or delta. Disabled entities are
included (see `test_snapshot.py`). Exposure filtering happens before
serialize; hidden entities do not appear and do not emit deltas.

**Consumer gap:** HA writes into the gateway session store; the text
controller reads `app["graph_store"]`. Those are different objects.

There is no “graph unavailable” flag on the server when the socket drops.
The last snapshot on a discarded `HaSession` is garbage-collected. The text
API’s store, if empty, yields `no_graph`.

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

`SemanticName` rejects Home Assistant `domain.object_id` strings
(`models.py`). Entity IDs are a resolver output, not a model output.

JSON Schema via `ControlPlan.json_schema()` / `control_plan_json_schema()`.

**Lifecycle today:**

1. Model text → `parse_model_output` (`parser.py`): JSON or fenced JSON;
   tool-call wrappers and invalid payloads become
   `NoActionPlan(reason="model_output_invalid")`. No speculative repair.
2. Or tests construct plans directly.
3. Orchestrator consumes the plan. Non-action outcomes become `NO_ACTION`
   after the safety barrier. Queries go to `evaluate_query` and never call HA.
4. Successful actions may record last-target / last-intent on
   `ConversationStore`.

`build_lfm_prompt` (`prompt.py`) serializes schema, origin, conversation
names/aliases (not raw entity IDs), and retrieved candidates. **No production
caller uses it.** `MlxModelRuntime.generate_plan` passes the raw user string
to `mlx_lm.generate`.

---

## Entity retrieval, resolution, validation, execution, and verification

These stages exist as functions. They are **not** a single wired pipeline
from HTTP to HA.

```text
utterance
  ├─ retrieve_candidates / resolve_candidates_for_request   [library; not in orchestrator]
  ├─ build_lfm_prompt + ModelRuntime.generate_plan          [not composed]
  └─ execute_control_plan(plan, snapshot, …)
        resolve_follow_up          (pronoun / “back on”)
        resolve_entity_ids         (scope + include/exclude + domain)
        evaluate_safety_barrier    (unsupported, empty, hidden, pronoun, capabilities)
        send_action_request        (ONE entity: sorted(ids)[0])
        take_action_results
        classify_action_results
```

| Stage | Module | Notes |
|---|---|---|
| Retrieval | `candidates.py`, `scoring.py` | O(n) scan of entities/scenes/scripts; token/domain/area/floor/alias/capability/referent scores |
| Ambiguity | `ambiguity.py` | Score-margin (`DEFAULT_AMBIGUITY_MARGIN`); clarification if two+ IDs within margin. **Not called by orchestrator** |
| Scope | `scope.py` | `current_area`, `named_area`, `floor`, `all` |
| Include/exclude | `exclusions.py` | Name match inside scope; subtract exclusions |
| Resolve | `resolver.py` | Combines the above; explicit `entity_ids` bypass scope |
| Capability | `capability.py` | Atomic reject of mixed invalid sets |
| Safety | `safety.py` | Barriers; `execute_if_safe` uses `HaClient.call_service` (separate from orchestrator’s `ActionRequestClient`) |
| Query | `queries.py` | Door open/closed; any/all lights on → yes/no |
| Execute | `orchestrator.py` | Maps plan to semantic action; **first sorted entity only** (commented ceiling in prior work) |
| Verify (server) | `classify_action_results` | Expects `accepted` then `completed`; flags incomplete/misordered |
| Verify (HA) | `state_verification.py` | `changed` / `unchanged` complete; timeout → `failed` |

HA mapping (`action_mapping.py`): `on`/`off`/`toggle` → `turn_*`;
`set_brightness` → `turn_on` + `brightness_pct`; `set_temperature` → climate
`set_temperature`; `scene`/`script` → `turn_on`. Orchestrator brightness
path sends `{"brightness": value}` while HA reads `request["brightness"]`
(`ACTION_PAYLOAD_BRIGHTNESS`). Those keys match. Temperature is produced by
the orchestrator only via `_semantic_action` for brightness/state/activate —
**climate `mode` / `set_temperature` is not emitted by the orchestrator.**

---

## Conversation-state ownership

`ConversationStore` (`conversation.py`) is in-memory, per `satellite_id`,
with a constructor TTL and a monotonic clock.

- `LastTarget.entity_ids` and `LastIntent`
- Cross-satellite referents do not resolve
- TTL expiry returns `None`; follow-ups then clarify (`followups.py`)

The store is optional on `execute_control_plan`.
`default_text_dependencies()` does **not** construct one.
There is no persistence across process restart.

---

## Voice pipeline ownership

**Not implemented.** No audio types, no STT, no wake word, no VAD, no TTS,
no playback, no PCM upload route.

Telemetry `input_type` is literal `"text"` only
(`sayso-server/src/sayso_server/telemetry.py`).

`MlxModelRuntime` is a resident **LLM** adapter (load-once LFM checkpoint),
not Whisper.

---

## Public interfaces and versioned contracts

`API_VERSION = 1`, `PROTOCOL_NAME = "sayso-api"` (`api.py`).

### HTTP (aiohttp app)

| Method | Path | Auth | Behavior |
|---|---|---|---|
| GET | `/api/v1/health` | Bearer | Liveness 200 + `{status, liveness, model_ready, ha_connected}` |
| GET | `/api/v1/ready` | Bearer | 200 iff `model_ready and ha_connected`, else 503 |
| POST | `/api/v1/text` | Bearer | `TextRequestEnvelope` → `text_response` or `error` |
| GET | `/api/v1/ws` | Bearer | HA session |

Text request: `{version: 1, type: "text", correlation_id, payload: {satellite_id, text}}`.
Text response: `type: "text_response"` with `category`, `reason`, `plan`,
`request_id`. Error codes: `invalid_request`, `unknown_satellite`,
`unknown_area`, `no_graph`, `not_configured`.

### WebSocket envelope

`SaySoEnvelope`: `version` literal 1, `type` ∈ `MessageType`, non-empty
`correlation_id`, `payload` dict (`envelope.py`).

Server `MessageType` (`messages.py`): `hello`, `hello_ack`, `ping`, `pong`,
`error`, `graph_snapshot`, `state_delta`, `registry_delta`.

HA additionally speaks `action_request` and `action_result` as plain JSON
with the same envelope shape (`coordinator.py`). Those types **fail**
`SaySoEnvelope` validation on the server (`test_envelope.py`).

JSON Schema fixtures: `evals/fixtures/sayso_api_v1.schema.json`,
`envelope.valid.json`, `home_graph.json`.

### HA services / options

- `sayso.sync_home_graph` (admin)
- Options: `domain_allowlist`, `action_allowlist`, `exposure_mode`,
  `area_ids`, `entity_ids`

---

## State ownership and sources of truth

| State | Source of truth | Replica |
|---|---|---|
| Device/entity registry and HA state | Home Assistant | Snapshot + deltas on the integration coordinator; intended replica is `HomeGraphStore` |
| Exposure / allowlists | Config entry options | Read at snapshot/delta/action time |
| WS connected | Coordinator `connected` | Binary sensor `SaySo Voice Assistant` / Connection |
| Server token | Caller of `create_aiohttp_app` | HA entry `data["token"]` |
| ControlPlan | Model output after parser, or test fixture | Orchestrator input |
| Conversation referents | `ConversationStore` (server memory) | Prompt serialization of names/aliases only |
| Satellite → area | `SatelliteRegistry` (server memory) | None |
| Readiness | `ReadinessState` | Health/ready JSON |
| Eval cases | JSONL under `evals/datasets/` | Loaded by `evals/schema.py` / `evals/corpus.py` |

There is no database.

---

## Failure handling and restart behavior

| Event | What happens |
|---|---|
| Bad HA token / non-hello first message | Server closes WS; integration reconnects with backoff |
| HA disconnect | Integration `connected=False`; sensor off; retry. Server readiness `ha_connected=False`. Session graph dropped |
| Stale delta | `HomeGraphStore` returns False; graph unchanged |
| Invalid ControlPlan JSON | `no-action` / `model_output_invalid` |
| Safety barrier | `ExecutionCategory.NO_ACTION`; no `send_action_request` |
| HA permission reject | `action_result` `rejected` before service call |
| Service exception | `failed` / `execution_failed` |
| No state change within timeout | `failed` / `state_verification_timeout` (timeout 5s live; 0 in tests with fake service caller) |
| Unchanged state | `completed` / `state_unchanged` |
| Missing `accepted`+`completed` order | `incomplete_results` or `misordered_results` |
| Server process restart | Nothing to restore; no graph, no satellites, model not marked ready |
| HA restart | Coordinator loop starts again; sends a new snapshot after hello. Server does not mark graph stale in the text-API store because that store was never filled |
| Integration unload | Stop coordinator, remove service, unload platform |

`sayso.sync_home_graph` is the operator resync. Tasks 65–66 (restart graph
unavailable semantics) are not implemented on the server.

---

## Current test boundaries

Colocated `test_*.py` only. Root `conftest.py` loads
`pytest_homeassistant_custom_component`. `pyproject.toml` `testpaths`:
`custom_components`, `sayso-server/src/sayso_server`,
`sayso-satellite/src/sayso_satellite`. There is no `tests/` directory.
`evals/` tests are not in `testpaths`; they are run by path.

| Area | Tests | Bound |
|---|---|---|
| Satellite | `sayso_satellite/test_import.py` | Import |
| ControlPlan / models / envelope | `test_control_plan.py`, `test_models.py`, `test_schema.py`, `test_envelope.py`, `test_messages.py`, `test_protocol.py`, `test_api.py` | Validation only |
| Runtime | `test_runtime.py`, `test_mlx_runtime.py`, `test_parser.py`, `test_prompt.py` | Fake loader; prompt content; parser fallbacks |
| Graph | `test_home_graph.py` | Snapshot replace and sequenced deltas |
| Gateway | `test_gateway.py` | Handshake, auth, aiohttp route |
| Retrieval / resolve / safety | `test_candidates.py`, `test_scope.py`, `test_exclusions.py`, `test_ambiguity.py`, `test_capability.py`, `test_safety.py`, `test_queries.py`, `test_followups.py`, `test_orchestrator.py` | In-memory fixture graphs and `FakeHaClient` |
| Text / telemetry / ready | `test_text_api.py`, `test_telemetry.py`, `test_health.py`, `test_readiness.py` | aiohttp test client; no live HA |
| Conversation | `test_conversation.py` | TTL / cross-satellite |
| HA integration | `test_manifest.py`, `test_config_flow.py`, `test_options_flow.py`, `test_coordinator.py`, `test_snapshot.py`, `test_exposure.py`, `test_deltas.py`, `test_capabilities.py`, `test_permissions.py`, `test_action_mapping.py`, `test_result_correlation.py`, `test_state_verification.py`, `test_device.py`, `test_diagnostics.py`, `test_init.py`, `test_services.py` | pytest-homeassistant fakes; fake WS |
| Evals | `evals/test_schema.py`, `test_core_corpus.py`, `test_safety_corpus.py`, `test_language_noise_corpus.py`; uncommitted `test_followup_corpus.py` | JSONL shape and resolver gold IDs — **not** model or HA runs |

No integration test opens a real Mac mic, real MLX checkpoint, or real
zwave/zigbee device.

---

## Status taxonomy

### 1. Implemented now

- ControlPlan and SaySo v1 envelope types; JSON Schema helpers
- Home Graph snapshot types; sequenced in-memory store
- HA config flow, options, outbound WS, hello, snapshot, deltas, ping,
  reconnect, `sync_home_graph`
- Exposure, permissions, action mapping, result correlation, state
  verification, connection entity, token-redacting diagnostics
- Candidate retrieval, scope, include/exclude, ambiguity **library**,
  capability validator, safety barriers, query evaluator, follow-up
  **library**, orchestrator classification
- Conversation store with TTL
- Text API **contract** (auth, satellite/area/graph gates, 503 without controller)
- Telemetry record shape and JSONL sink
- Liveness vs readiness JSON (flags exist; model_ready unused in startup)
- `MlxModelRuntime` load-once wrapper (needs `mlx-lm` installed by the environment)
- Strict parser → `model_output_invalid`
- Eval schema; core (120), safety (100), language-noise (150) corpora
- `SatelliteRegistry` in-memory API
- `AGENTS.md` voice-path priority

### 2. Partially implemented

- **HTTP app:** routes exist; no main, two server classes, text controller not
  auto-wired, readiness `model_ready` never set in production code
- **HA graph ingest:** works on `HaSession.graph`, not on the store the text
  API reads
- **Action path:** HA can execute `action_request`; server cannot send it;
  `FakeHaClient` stands in
- **Model path:** runtime interface + MLX + prompt builder + parser exist but
  are not composed; fake runtime always queries
- **Orchestrator:** plan→resolve→validate→request→verify for **one** entity;
  climate temperature/mode not mapped
- **Ambiguity / retrieval:** implemented, unused by `execute_control_plan`
- **Follow-ups:** used when a `ConversationStore` is passed; default text
  dependencies omit it
- **Auth:** constant-time on WS/text; not on health/ready
- **Satellite:** server registry only; no client, no persistence, no handshake
- **Task 43 corpus:** generator + 70 JSONL rows + tests, uncommitted
- **Diagnostics / health:** present; health compare is not constant-time;
  diagnostics are functional rather than polished

### 3. Planned but not implemented

From `docs/MVP_PLAN.md` / `docs/EVALUATION_PLAN.md`, absent in code:

- 43 committed follow-up corpus (WIP only)
- 44 metric scorer, 45 benchmark runner, 46–47 model bake-offs, 48 live
  dry-run allowlist, 49–50 baseline imports, 51 statistical report
- 52–57 push-to-talk satellite (registration client, 16 kHz PCM, capture,
  Whisper, audio→text, response policy)
- 58–64 wake word, VAD, endpoint SM, continuous loop, `say`/`afplay`, 20-command smoke
- 65–70 restart graph-unavailable, concurrency, security audit extras,
  install docs, frozen benchmark
- `training/` directory (listed in the plan; not in the tree)
- Home-FunctionGemma adapter, Home-LLM, Alexa+ importer

### 4. Original requirements that conflict with the current implementation

- **Phase 3 gate:** “Text commands operate real Home Assistant devices.”
  Text never reaches the integration socket. `MESSAGE_TYPES_V1` still treats
  `action_request` as future (`test_messages.py`).
- **Single process topology:** plan assumes one server serving satellite HTTP
  and HA WS with a shared graph. The app constructs two graph stores and
  drops the HA session.
- **LFM prompt contains candidates only, not the full graph:** `build_lfm_prompt`
  does that, but the MLX runtime prompts with raw user text.
- **No generic tool loop / only ControlPlan:** parser enforces this; the live
  generate path does not feed the schema into the model.
- **Constant-time token check (task 25):** WS and text API only; health uses
  `!=`.
- **aiohttp as the server** (`MVP_PLAN` minimal choices): implemented as a
  library function; stdlib `ThreadingHTTPServer` also exists; neither is a
  product command; `aiohttp` is not a `sayso-server` dependency.
- **Satellite as Mac smart speaker:** package is a stub.
- **Eval action_request on the wire vs envelope tests:** HA and server
  disagree on whether `action_request` is a v1 type.

### 5. Architectural decisions that remain unresolved

- Whether `action_request` / `action_result` join `MessageType` or stay a
  side channel of untyped JSON
- How the gateway session store becomes the text-API `HomeGraphStore` (one
  store, copy-on-snapshot, or pub/sub)
- Who constructs `OrchestratorTextController` (config file, env, HA-driven)
- When `model_ready` flips true (MLX `load()` success vs first generate)
- Fake vs MLX vs a recorded-plan runtime for the first physical demo
- Multi-entity execution (orchestrator currently one ID)
- Whether ambiguity runs before the model (candidate list) and/or after
  (resolver), and how that interacts with include/exclude
- Climate `set_temperature` / `mode` in the orchestrator vs HA mapping
- Satellite auth vs the HA shared Bearer token
- Ping/pong: integration sends `ping`; server neither answers nor requires it
- Package dependencies for `aiohttp` and `mlx-lm`
- Whether first E2E demo is **text-from-Mac** or **push-to-talk voice**

---

## Remaining plan: first physical-device demo vs deferrable

`AGENTS.md` asks to finish Mac → SaySo → Home Assistant → physical device
without cutting ControlPlan validation, ambiguity, integration execution,
verification, metrics, or basic evals.

A **first demonstration** is: one Mac-originated command changes one real HA
entity through SaySo, with safety barriers still able to no-op. Voice wake
and bake-offs are not required for that.

### Required (or blocking glue not numbered as a remaining task)

These are missing **composition** of code that already exists, plus the
smallest remaining units that create a client and a process.

1. **Shared Home Graph.** Apply HA snapshots/deltas into the store
   `POST /api/v1/text` reads. Without this, the text API cannot resolve
   real entities (`no_graph` or a stale fixture).
2. **Live `ActionRequestClient` on the HA WebSocket.** Register
   `action_request` / `action_result` on the server contract (today’s tests
   forbid that) and wait for correlated results. Integration execution and
   verification already work when that message arrives.
3. **Process entrypoint** that loads a token, optional MLX, `ReadinessState`,
   `SatelliteRegistry`, and `create_aiohttp_app` with a real
   `OrchestratorTextController`.
4. **Compose plan generation:** `retrieve_candidates` → `build_lfm_prompt` →
   runtime → `parse_model_output`. Keep the parser’s no-repair rule.
5. **Put ambiguity on the live path** (task 30 library). A unique-name
   command can demo without it; leaving it unwired *cuts* a safety
   requirement the plan and `AGENTS.md` both keep.
6. **Satellite registration for one Mac** (task 52). `SatelliteRegistry`
   exists; the satellite package must send `satellite_id` with text (curl
   is enough for a first demo; a tiny client is enough for “Mac”).
7. **Declare `aiohttp` (and `mlx-lm` if using LFM)** so the server can run
   outside the HA test extra.
8. **Task 66-class stale-graph guard** (or equivalent): if HA disconnects,
   refuse execution against an empty or frozen graph. The integration
   already resends a snapshot after reconnect (partial 65).
9. **Package-level safety that already exists** must stay: ControlPlan
   validation, capability/safety barriers, HA permission + state
   verification. Do not bypass them for the demo.

Optional for a *voice* demonstration of the same path, still required for
“Mac as speaker”: tasks **53–56** (PCM upload, capture, resident Whisper,
audio → existing text controller). Task **57** (earcon vs short TTS) can be
a print/beep for the first run.

Task **48** (dry-run default + `--execute` allowlist) is required before
pointing evals at a live home; it is not required for a supervised first
demo on a dedicated HA. Do not disable integration permission checks.

### Can be deferred without compromising safety or the core architecture

Does not block a single-command physical demo and does not remove the
safety libraries:

| Tasks | Why deferrable |
|---|---|
| **43** follow-up corpus | Product follow-up resolver exists; JSONL is eval coverage |
| **44–45** scorer / runner | Metrics stay as a later keep; not needed to flip a light |
| **46–47, 49–51** | Extensive model benchmarking and baseline imports (`AGENTS.md` defer) |
| **58–64** | Wake/VAD/continuous loop — after push-to-talk or even after text demo |
| **67** | Per-satellite serialization — one Mac, one command |
| **68** extras | Token redaction already exists; size limits/docs can wait |
| **69–70** | Install docs and frozen comparison report |
| Expanding corpora past core+safety (+ noise already authored) | Large eval datasets |
| Generalized multi-satellite, fine-tuning hooks, streaming, polished diagnostics | Explicitly deferred in `AGENTS.md` |
| Orchestrator multi-entity fan-out | Ceiling: first sorted id. First demo can target one light |
| Climate mode mapping | Not on the critical “turn on the lamp” path |

### Do not defer (even if they are “already coded”)

These are implemented as modules but **omitting them from the live path
would cut architecture**, not save time:

- ControlPlan validation and `model_output_invalid`
- Safety + capability barriers
- HA exposure/permissions and state verification
- Ambiguity rule once more than one equally scored device exists
- Basic eval schema and core/safety cases as regression fixtures (running
  LFM vs Gemma bake-offs can wait)
