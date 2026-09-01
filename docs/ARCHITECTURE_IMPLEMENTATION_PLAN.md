# SaySo architecture implementation plan

Status: architecture-aligned plan for the current worktree.

Companions: [architecture](ARCHITECTURE.md), [evaluation](EVALUATION_PLAN.md),
[reliability](MVP_RELIABILITY_AND_EVALUATION_PLAN.md), and
[next steps](NEXT.md).

## Goal

Make one Mac a trustworthy local smart-home voice client:

```text
Mac text/audio → SaySo server → typed ControlPlan → deterministic controller
→ authenticated HA WebSocket → HA permission/execution → verified device state
```

The MVP does not use Home Assistant Assist, arbitrary tool calls, a database,
cloud services, or a generic agent loop. Home Assistant owns physical truth;
the server owns only an in-memory graph replica and the control decision before
the integration execution boundary.

## Definition of done

The MVP is done when all of these are true:

- A configured Mac command changes the intended real Home Assistant device.
- Recorded PCM audio reaches the same controller as text.
- Invalid, ambiguous, unsupported, unresolved, hidden, or incapable requests
  execute nothing.
- The HA integration rechecks exposure, permissions, capability, and state.
- Successful actions return correlated, verified results.
- Text and audio interactions emit honest structured telemetry.
- The committed eval corpora run through a non-live controller executor with a
  failure ledger and latency report.
- The expansion gate passes with zero false execution and zero wrong-device
  actions on the scored dry-run set.

Wake word, VAD, hands-free listening, second-model comparisons, and a larger
benchmark are post-MVP work gated on this baseline.

## Non-negotiable architecture

- Three runtime boundaries: satellite, server, and HA integration.
- HTTP `/api/v1/text` and `/api/v1/audio` are satellite ingress points.
- HA is the outbound WebSocket client; the server listens on `/api/v1/ws`.
- API and WebSocket envelopes are version 1 and carry correlation IDs.
- The model emits only semantic `ControlPlan` JSON. It never emits entity IDs
  or raw HA service calls.
- Strict parsing converts malformed model output to
  `no-action/model_output_invalid`; do not add speculative JSON repair.
- Resolution, ambiguity, capability, and safety checks run before an action
  request.
- HA integration checks permissions again, maps a fixed semantic action to an
  HA service, and verifies state before returning `completed`.
- The server clears its graph when HA disconnects and refuses execution until a
  fresh snapshot arrives.
- The current deliberate ceiling is one resolved entity per action.
- Python, colocated `test_*.py`, no `tests/` directory, no `context.json`.

## Current baseline

The supervised text path is complete:

```text
Mac satellite → SaySo server → LFM ControlPlan → resolver/safety
→ HA integration → Home Assistant → physical plug-lamp state change
```

The current worktree already contains:

- v1 contracts, schema fixtures, graph snapshots, and sequenced deltas;
- resident MLX LFM runtime and best-effort resident Whisper preload;
- strict ControlPlan parsing and deterministic retrieval/resolution;
- scope, include/exclude, ambiguity, capability, safety, query, and follow-up
  handling;
- HA config/options, exposure, permissions, fixed action mappings, reconnect,
  diagnostics, resync, and state verification;
- text and recorded PCM endpoints using one text controller;
- JSONL telemetry, committed corpora, dry-run execution, metrics, failure
  ledger, latency/report helpers, and an expansion gate.

The supervised demo used ChatML few-shot prompting with
`mlx-community/LFM2.5-230M-OptiQ-4bit`, switch-as-light retrieval, and a
whole-home named-target retry when the origin-area candidate set missed.

## Execution workflow

Every implementation unit is a small TDD change:

1. Add the smallest failing colocated test.
2. Run it and record the failure.
3. Add the minimal implementation.
4. Run the focused test and relevant regressions.
5. Update the status and stop before starting another unit unless continuation
   has been explicitly authorized.

Documentation-only changes use link/path checks instead of adding tests. Never
weaken an existing assertion to make a unit pass.

## Workstream 1 — Reliability baseline

Source of truth: [MVP reliability and evaluation plan](MVP_RELIABILITY_AND_EVALUATION_PLAN.md).
Complete this workstream before expanding the corpus or starting wake/VAD.

| Unit | Failing test | Minimal implementation | Verification |
|---|---|---|---|
| R1. Live conversation state | Default live controller has no `ConversationStore` | Attach the existing per-satellite TTL store in app wiring | `test_app.py`, `test_main.py` |
| R2. Audio telemetry identity | Audio records as `input_type="text"` | Pass `input_type="audio"` through the existing controller contract | `test_telemetry.py`, `test_audio_api.py` |
| R3. Separate STT readiness | Readiness has no STT status | Add `stt_ready`; keep aggregate readiness as model + HA only | `test_readiness.py`, `test_main.py` |
| R4. Recorded fixture path | PCM fixture never reaches the controller | Exercise Fake STT → same text controller, without a live mic | `test_audio_api.py` |
| L1. Failure fields | Eval records cannot identify failure stage/reason | Add optional ledger fields and fixed stage vocabulary | `evals/test_metrics.py`, `test_schema.py` |
| L2. Failure classifier | Parse and retrieval failures collapse together | Classify each failed case once by pipeline stage | `evals/test_ledger.py` |
| L3. Ledger summary | No deterministic per-stage case ledger | Add counts, reasons, and sorted case IDs | `evals/test_ledger.py` |
| B1. Controller dry run | Dry-run records contain no plan or targets | Run FakeModelRuntime, resolver, and FakeHaClient; never live HA | `evals/test_executor.py`, `test_runner.py` |
| B2. Eval CLI | `python -m evals` is unavailable | Add corpus selection, output path, and safety flags | `evals/test_main.py`; run the CLI manually |
| B3. Frozen baseline | Fake-runtime behavior has no golden numbers | Check in a small baseline slice and assert metric numerators | `evals/test_baseline.py` |
| M1. Run metadata | Runs omit model, seed, quantization, or warmup | Add frozen `BenchmarkConfig` and JSONL header | `evals/test_config.py` |
| M2. Stage timing | Rows contain only total latency | Persist stage timings, token counts, and model ID | `evals/test_runner.py`, `test_executor.py` |
| M3. Optional MLX run | Missing MLX aborts fake-runtime evaluation | Make live MLX executor opt-in; keep CI fake-only | `evals/test_mlx_executor.py` |
| F1. Live telemetry sink | App discards interaction telemetry | Open JSONL sink only when `SAYSO_TELEMETRY_PATH` is set | server app and telemetry tests |
| F2. STT timing | Audio has no `stt` stage | Time STT and keep text-path STT at zero | telemetry and audio tests |
| P1. Latency report | Timed rows cannot produce median/p95 | Add nearest-rank warm latency report | `evals/test_latency.py` |
| P2. Evaluation report | Report mixes cold starts and hides sample size | Combine score, ledger, latency, and config metadata | `evals/test_report.py` |
| G1. Expansion gate | Corpus expansion can proceed after unsafe results | Fail closed on false execution, wrong device, missing timing, or schema crashes | `evals/test_gate.py` |
| G2. Architecture alignment | Architecture doc treats reliability as undifferentiated future work | Keep this plan as the reliability execution pointer | Link/path check |

### Workstream 1 gate

Run the committed corpora with `controller_dry_run_executor`. The run must
produce non-empty plans/targets where expected, classify failures, report warm
latency, and pass the expansion gate. No live Home Assistant action is allowed
without both `--execute` and an explicit entity allowlist.

## Workstream 2 — Recorded-file voice demonstration

This workstream proves the already-built audio transport against a configured
home. It does not add a live microphone loop.

| Unit | Failing check | Implementation | Verification |
|---|---|---|---|
| V1. HA naming | A household name maps to multiple lamps | Rename or alias the intended HA entity; do not alter model prompting | Named command changes one entity; equal aliases still clarify |
| V2. Effective areas | Device-backed entities lack their HA area | Keep `entry.area_id or device.area_id` in snapshot serialization | Snapshot and live graph show correct areas |
| V3. PCM command | Audio file path has no physical-device check | Post the committed 16 kHz mono PCM fixture through `--audio-file` | Same verified result category as equivalent text |
| V4. Operator runbook | Three processes require undocumented manual setup | Document server env, HA config, satellite env, readiness, and resync | Fresh operator can run health → connect → command |

### Workstream 2 gate

The recorded clip reaches STT, the shared text controller, the HA integration,
and state verification. A transcript mismatch is reported as such; a timeout
or silent fallback is not accepted as success.

## Workstream 3 — Production hardening

Do these after the reliability and recorded-file gates, and only where the
failure ledger or a live trial demonstrates the need.

| Unit | Failure to reproduce | Minimal implementation | Verification |
|---|---|---|---|
| H1. Server restart | Reconnect does not restore a usable graph | Require fresh snapshot after server restart | Restart matrix and graph resync test |
| H2. HA restart | Stale state remains actionable | Keep graph unavailable until fresh snapshot | No action while disconnected; action after resync |
| H3. Request serialization | Same satellite commands interleave | Serialize per-satellite work with bounded concurrency | Concurrent request fixture |
| H4. Security boundary | Secret, oversized, or malformed input leaks through | Redaction, size limits, timeouts, and private-LAN guidance | Security regression checks |
| H5. Package/runtime alignment | Clean install lacks imported MLX/STT stack | Pin/document optional runtime dependencies only if the install trial requires it | Clean-machine installation check |
| H6. Execution capability | Required use case needs more than one entity or climate emission | Add the smallest tested capability extension | New safety and verification fixtures |

The one-entity execution ceiling and incomplete climate plan emission remain
valid until a measured use case requires changing them.

## Workstream 4 — Hands-free voice, deferred

Start only after Workstream 1 passes, Workstream 2 is demonstrated, and the
project constraints explicitly authorize it.

1. Add a replaceable wake-word adapter with threshold configuration.
2. Add local VAD and endpointing with pre-roll, minimum speech, silence, and
   maximum-duration limits.
3. Add a satellite state machine that guarantees no STT or model call before
   wake.
4. Add response playback and temporary-audio cleanup on the satellite.
5. Run a scripted household acceptance trial with timing and false-wake data.

Keep wake, VAD, endpointing, and playback outside the server’s deterministic
control path. They feed the existing PCM/audio endpoint.

## Workstream 5 — Comparative benchmarks, deferred

The first benchmark is the reproducible LFM/fake-runtime baseline from
Workstream 1. Do not add another production model or change the checkpoint to
solve a model-output-quality issue.

After the expansion gate and explicit authorization:

- run optional live LFM measurements with fixed model/runtime metadata;
- add a Home-FunctionGemma adapter only if it consumes the same candidates and
  returns the same ControlPlan contract;
- import Home-LLM and Alexa+ observations as evaluation-only comparisons;
- report parser failures, wrong-device rate, false execution, category scores,
  cold/warm latency, and sample sizes separately.

No comparison may weaken safety barriers or turn a merely executable action
into a correct one.

## Acceptance checks

Before calling the MVP complete, verify:

- `GET /api/v1/health` is live and reports model, HA, and STT status.
- `GET /api/v1/ready` is 200 only after model readiness and a valid HA graph.
- Text and recorded audio return through the same response-policy contract.
- A malformed plan, ambiguous name, unsupported capability, hidden entity,
  stale graph, and disconnected HA session all result in no execution.
- A valid action returns `accepted` followed by verified `completed` or a
  classified failure.
- Telemetry contains the required fields, stage timings, and no raw audio.
- The committed eval corpora run deterministically, resume without duplicate
  case IDs, and produce a failure ledger and latency report.
- The expansion gate passes before any new corpus, wake/VAD work, or second
  model is started.

## Explicitly deferred

- Generic agent/tool frameworks
- Vector or graph databases
- Persistent replicated HA state
- Multiple/generalized satellites
- Streaming STT/TTS, echo cancellation, and barge-in
- Model training or fine-tuning
- Music, web, calendar, email, and cloud services
- Large corpus expansion before the reliability gate
- Home-LLM/Alexa+ claims before paired measurement
- Packaged Docker or single-command multi-process launcher

Add a deferred item only when a measured MVP limitation requires it.
