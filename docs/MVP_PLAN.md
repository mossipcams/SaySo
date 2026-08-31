# SaySo MVP Implementation Plan

Status: Approved planning baseline  
Companion: [Evaluation and Benchmark Plan](EVALUATION_PLAN.md)

## Goal

Build SaySo as a fully local, Alexa-style smart-home voice assistant. A small
local model performs language understanding, a deterministic controller turns
its output into safe actions, and Home Assistant remains authoritative for
home state and physical execution.

The MVP succeeds when a Mac can act as a temporary smart speaker and reliably
control real Home Assistant devices through the first-class SaySo custom
integration without using Home Assistant Assist for language understanding.

## Architecture

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
                                    │
                                    ▼
                            Physical devices
```

The Home Assistant integration opens an outbound authenticated connection to
SaySo Server. It sends an initial permitted Home Graph snapshot, streams
deltas, receives typed action requests, enforces permissions, executes through
Home Assistant, and returns correlated results.

This keeps the model outside Home Assistant, prevents arbitrary service-call
generation, avoids Home Assistant Assist, and lets either process restart
independently.

## Minimal implementation choices

- Python throughout.
- `aiohttp` for SaySo Server HTTP and WebSocket endpoints.
- Pydantic for ControlPlan and SaySo API validation and JSON Schema generation.
- MLX-LM as the first model runtime behind one narrow runtime interface.
- `mlx-community/LFM2.5-230M-OptiQ-4bit` as the initial Mac checkpoint.
- In-memory dictionaries and an O(n) candidate scan for the MVP Home Graph.
- JSONL telemetry using monotonic clocks; no external observability service.
- macOS `say` and `afplay` for initial TTS and playback.
- Colocated `test_*.py` files; do not create or modify a `tests/` directory.
- No generic tool loop. The model can emit only a validated ControlPlan.

## Repository target

```text
SaySo/
├── pyproject.toml
├── sayso-server/
│   └── src/sayso_server/
├── sayso-satellite/
│   └── src/sayso_satellite/
├── custom_components/
│   └── sayso/
├── evals/
│   ├── fixtures/
│   ├── datasets/
│   └── reports/
├── training/
│   └── README.md
└── docs/
```

## Required execution workflow

Every numbered task is a 5–15 minute TDD unit:

1. Add the smallest failing test outside a `tests/` directory.
2. Run it and record the expected failure.
3. Add the minimal implementation.
4. Run the focused test and relevant regression checks.
5. Stop and request approval before starting the next task.

## Phase 1 — Text-to-ControlPlan foundation

| Task | Failing test first | Minimal implementation | Verification |
|---|---|---|---|
| 1. Repository skeleton | Imports fail for server and satellite packages | Component directories, minimal package metadata, check command | Clean install and smoke test pass |
| 2. ControlPlan schema | Missing intent, invalid state/value pairs, and entity IDs in semantic targets are accepted | Typed action/query/clarification/unsupported/no-action outcomes | Valid examples round-trip; unsafe examples fail |
| 3. SaySo API v1 envelope | Unknown versions/types and missing correlation IDs are accepted | Versioned message envelope and generated JSON Schema | Contract fixtures validate |
| 4. Home Graph types | Representative snapshot cannot load | Floor, area, device, entity, scene, script, capability, and state types | Fixture round-trips without field loss |
| 5. Conversation state | Expired and cross-satellite references resolve | Per-satellite structured state with configurable TTL | Active reference works; stale reference does not |
| 6. Model runtime contract | Fake runtime cannot expose token, latency, and model metadata | Narrow `load()` and `generate_plan()` interface | Fake runtime proves interchangeability |
| 7. LFM prompt builder | Prompt contains the full graph or raw entity IDs | Prompt with schema, origin, structured state, and retrieved candidates only | No unrelated entities appear |
| 8. Model-output parser | Malformed JSON or tool calls pass | Strict extraction and ControlPlan validation with no speculative repair | Invalid output becomes `no_action/model_output_invalid` |
| 9. Resident MLX runtime | Two generations load the model twice | Load LFM once at startup and retain it | Load count remains one; warm metrics are emitted |

### Phase 1 gate

Representative text produces valid ControlPlans without Home Assistant access.

## Phase 2 — First-class Home Assistant integration

Minimum component files:

```text
custom_components/sayso/
├── __init__.py
├── manifest.json
├── config_flow.py
├── const.py
├── coordinator.py
├── diagnostics.py
├── services.yaml
└── translations/
```

Add a connection-status entity to create the `SaySo Voice Assistant` device.

| Task | Failing test first | Minimal implementation | Verification |
|---|---|---|---|
| 10. Integration manifest | Manifest validation fails | `sayso` service integration metadata and config-flow declaration | Manifest validation passes |
| 11. Initial config flow | Bad URL/token creates an entry | URL/token fields and connection probe | Invalid credentials show an error; valid entry succeeds |
| 12. Options flow | Permissions and exposure cannot change | Domain/action allowlist plus all/area/entity exposure modes | Options persist and reload |
| 13. Connection coordinator | Disconnect leaves stale connected state | Authenticated outbound WebSocket, heartbeat, bounded backoff | Disconnect/reconnect transitions pass |
| 14. Registry snapshot | Snapshot omits floors, aliases, disabled, or excluded entities | Registry-to-contract serializer | Exact expected snapshot is produced |
| 15. Capabilities and actions | Unsupported attributes become capabilities | Conservative mapping for power, brightness, temperature, query, scene, and script | Only executable capabilities are exposed |
| 16. Incremental updates | State change sends a full snapshot | State and registry delta messages | One entity change yields one delta |
| 17. Exposure enforcement | Hidden entity leaks into a snapshot | Filter before serialization | Hidden entities never reach the server |
| 18. Action permission enforcement | Disallowed request reaches Home Assistant | Entity, domain, action, and capability checks inside the integration | Rejection occurs before service execution |
| 19. Typed action mapping | Semantic actions produce incorrect service payloads | Fixed mapping for on/off/toggle, brightness, thermostat, scene, and script | Exact service and payload assertions pass |
| 20. Result correlation | Concurrent results cross-match | Request IDs and typed accepted/rejected/failed/completed results | Concurrent fake requests correlate correctly |
| 21. State verification | Accepted call is reported successful without a state change | Await relevant state event with timeout | Changed, unchanged, and timed-out outcomes differ |
| 22. Device and health entity | Integration creates no SaySo device/status | `SaySo Voice Assistant` device and connection entity | Device registry assertion passes |
| 23. Diagnostics and redaction | Token appears in diagnostics | Health, exposure counts, protocol status, and secret redaction | Secret scan passes |
| 24. Service and translations | Home Assistant metadata validation fails | `sayso.sync_home_graph`, English strings, unload/reload handling | Component validation and reload pass |

### Phase 2 gate

A synthetic server connects, receives a graph, requests a permitted action,
and observes a verified result. No Assist APIs are imported.

## Phase 3 — Deterministic server control path

| Task | Failing test first | Minimal implementation | Verification |
|---|---|---|---|
| 25. HA session gateway | Wrong token/version establishes a session | Constant-time token check and v1 handshake | Invalid clients close; valid client receives hello |
| 26. Home Graph updates | Out-of-order/stale deltas corrupt state | Atomic snapshot replacement and sequenced deltas | Reconnect/resync restores expected graph |
| 27. Candidate retrieval | Relevant alias/current-room target is missed | Token normalization plus domain, area, floor, alias, capability, state, and referent scoring | Gold target appears in top candidates |
| 28. Scope resolver | Current area, named area, floor, or explicit target resolves incorrectly | Deterministic scope expansion | Exact entity-set assertions pass |
| 29. Include/exclude resolver | Negation selects the excluded device | Resolve names inside scope and subtract exclusions | “All except floor lamp” yields the exact set |
| 30. Ambiguity handling | Equal matches choose arbitrarily | Score-margin rule returning clarification | Ambiguous lamp request performs no action |
| 31. Capability validator | Brightness reaches a non-dimmable entity | Per-target capability and range checks | Mixed invalid target set is rejected atomically |
| 32. Safety validator | Unsupported, unresolved pronoun, empty target, or hidden entity executes | Explicit no-action barriers | Fake HA client records zero calls |
| 33. Execution orchestrator | Partial or misordered result is success | Plan → resolve → validate → request → verify pipeline | Exact success/failure category is emitted |
| 34. State queries | Query path attempts a service call | Read-only single and aggregate query evaluator | Door and “any lights on” fixtures answer correctly |
| 35. Follow-ups | “Turn it back on” loses its prior target | Structured last-target and last-intent references with TTL | Active follow-up resolves; expired one clarifies |
| 36. Text API | Invalid satellite/area reaches controller | Validated `/api/v1/text` endpoint and response envelope | HTTP contract tests pass |
| 37. Telemetry | Failure path omits mandatory fields | One JSONL record with stage timings and required metadata | Success and rejection paths satisfy schema |
| 38. Health/readiness | Server is ready before model/HA are usable | Separate liveness, model readiness, and HA connection status | Restart-state matrix passes |

### Phase 3 gate

Text commands operate real Home Assistant devices through the integration,
including exclusions, queries, ambiguity, and follow-ups.

## Phase 4 — Automated evaluation

Tasks 39–51, metric definitions, fair comparison rules, and statistical limits
are specified in [Evaluation and Benchmark Plan](EVALUATION_PLAN.md).

### Phase 4 gate

At least 300 reviewed cases run against LFM2.5 and Home-FunctionGemma with
attributable failures and reproducible latency measurements.

## Phase 5 — Push-to-talk voice

| Task | Failing test first | Minimal implementation | Verification |
|---|---|---|---|
| 52. Satellite registration | Unknown satellite lacks area context | Satellite ID, configured HA area, server URL/token | `macbook → living_room` handshake passes |
| 53. Audio transport | Corrupt format reaches STT | 16 kHz mono PCM upload with duration and sequence metadata | Recorded fixture round-trips |
| 54. Push-to-talk capture | Initial speech is clipped | Mac microphone capture and bounded pre-roll | Fixture waveform retains the leading phoneme |
| 55. Local STT | STT reloads or omits timing | Resident MLX Whisper adapter returning transcript and metrics | Known English clips meet fixture tolerance |
| 56. Voice pipeline | Transcript bypasses the existing controller | Audio → STT → existing text pipeline | Text and audio inputs resolve identically |
| 57. Response policy | Basic controls produce verbose speech | Earcon for control; short TTS for query, clarification, or error | Response-policy matrix passes |

### Phase 5 gate

Push-to-talk controls actual Home Assistant devices with complete
end-of-speech-to-result metrics.

## Phase 6 — Wake word and VAD

| Task | Failing test first | Minimal implementation | Verification |
|---|---|---|---|
| 58. Wake adapter | Audio activates STT without a wake event | Replaceable openWakeWord adapter with configurable model/threshold | Positive and negative clips classify correctly |
| 59. Wake telemetry | Detection time/false wake cannot be recorded | Wake event, latency fields, false-wake annotation | Telemetry schema passes |
| 60. VAD adapter | Silence and speech boundaries are wrong | Silero ONNX adapter with threshold and reset | Fixture boundaries fall within tolerance |
| 61. Endpoint state machine | Pause ends too early or hangs | Pre-roll, minimum speech, configurable silence, maximum timeout | Pause/noise/timeout fixtures pass |
| 62. Continuous loop | LLM/STT runs before wake | Wake → capture/VAD → upload state machine | Counters show zero STT/LLM calls before wake |
| 63. TTS/playback | Query audio blocks or persists raw audio | Server TTS interface using `say`, satellite playback using `afplay`, cleanup | Query is audible and temporary audio is removed |
| 64. Voice acceptance | No reproducible device trial exists | Scripted 20-command household smoke run | Commands, queries, clarifications, and timing pass |

### Phase 6 gate

“Hey SaySo, turn off the lights” wakes locally, ends speech locally, transcribes
locally, executes through the deterministic controller and SaySo integration,
and changes the intended physical lights.

## Phase 7 — Hardening and handoff

| Task | Failing test first | Minimal implementation | Verification |
|---|---|---|---|
| 65. Server restart | Integration never resynchronizes | Reconnect plus mandatory fresh snapshot | Kill/restart restores graph |
| 66. HA restart | Server uses stale graph while disconnected | Mark graph unavailable until a fresh snapshot | No action executes against stale state |
| 67. Concurrent requests | Commands interleave state/results | Per-satellite serialization with bounded global concurrency | Concurrency fixture passes |
| 68. Security audit | Secret or entity leak fixture fails | Token redaction, size limits, timeouts, private-LAN guidance | Secret and oversized-message checks pass |
| 69. Installation docs | Clean-machine checklist fails | Mac, server, integration, satellite, model, and troubleshooting docs | Fresh installation succeeds |
| 70. Final benchmark | Report lacks required evidence | Frozen model/config/dataset and comparison report | MVP gates and metrics are reproducible |

## MVP acceptance gates

- Real commands travel through LFM2.5 → ControlPlan → resolver → SaySo
  integration → Home Assistant → physical device.
- Invalid, ambiguous, unsupported, and unresolved requests execute nothing.
- Production code does not import or call Home Assistant Assist or conversation
  intent handling.
- Model and STT remain resident during normal operation.
- The wake/VAD loop performs no continuous STT or LLM inference.
- Every interaction produces required structured telemetry without raw audio.
- Evaluation contains at least 300 reviewed cases and can scale beyond 1,000.
- LFM2.5 and Home-FunctionGemma receive the same candidates, schema, corpus,
  and hardware conditions where technically possible.
- Home-LLM and Alexa+ comparisons distinguish measurements from inference.
- Median and p95 EOS-to-HA execution are reported before performance claims.

## Explicitly deferred

- Generic agent/tool frameworks
- Vector databases or graph databases
- Replicated Home Assistant state storage
- Dedicated or multiple hardware satellites
- Streaming STT or TTS
- Acoustic echo cancellation and barge-in
- Model training or fine-tuning
- Music, web, calendar, email, and cloud services

Add these only when evaluation shows a measured MVP limitation.
