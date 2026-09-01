# SaySo MVP Reliability and Evaluation Plan

Status: Active  
Companions: [Architecture](ARCHITECTURE.md), [MVP plan](MVP_PLAN.md), [Evaluation plan](EVALUATION_PLAN.md), [What’s next](NEXT.md)

## Goal

Make the Mac voice path trustworthy enough to measure, then measure it.

A **reliability baseline** exists when:

1. Recorded-file audio uses the same controller as text, with honest telemetry.
2. Every eval failure is attributed to one pipeline stage.
3. Committed corpora score through the deterministic controller without touching live Home Assistant.
4. A seeded LFM run configuration can execute (fake runtime in CI; live MLX optional).
5. Code fixes come from the failure ledger, not guesswork.
6. Warm median/p95 stage latency is reported with sample size.
7. Corpus expansion is blocked until the expansion gate passes.

This plan does **not** reopen the first physical-device text demo. Keep ControlPlan validation, ambiguity, safety barriers, HA permissions, and state verification on the live path. Do not change the LFM checkpoint. Do not add speculative JSON repair.

## Constraints

From `AGENTS.md`:

- Keep: ControlPlan validation, integration execution, verification, metric scoring, core/safety/follow-up eval coverage.
- Defer: Home-FunctionGemma / Home-LLM / Alexa+ bake-offs, corpora past the committed sets, multi-satellite, fine-tuning, streaming, wake/VAD, polished diagnostics.
- Python throughout. Colocated `test_*.py`. Never create a `tests/` directory.
- One numbered TDD unit at a time (this document authorizes continuing until the plan is done).

## Already in tree (do not rebuild)

| Piece | Where |
|---|---|
| Eval schema, core/safety/language-noise/follow-up corpora, scorer | `evals/` tasks 39–45 |
| Resumable JSONL runner, dry-run default, `--execute` + allowlist | `evals/runner.py` task 48 |
| Recorded 16 kHz PCM fixture | `evals/fixtures/audio_pcm16_mono_16k.bin` |
| Satellite HTTP timeout 180s for text and audio | `sayso-satellite` |
| `model_ready` after loaded runtime | `create_aiohttp_app` |
| STT preload after LFM (best-effort, does not gate ready) | `sayso_server.__main__` |
| Device area inherited onto snapshot entities | `custom_components/sayso/snapshot.py` |
| Fake HA client, FakeModelRuntime, VoicePipelineController | server modules |

`dry_run_executor` currently writes **empty** `EvalRecord`s. That is the main eval gap this plan closes.

## Pipeline stages (failure ledger)

Use these stage names everywhere (ledger, telemetry, reports):

```text
stt → retrieve → plan → parse → resolve → safety → request → verify
```

| Stage | Means |
|---|---|
| `stt` | Transcript missing or outside fixture tolerance |
| `retrieve` | Required candidate entity absent from the retrieved set |
| `plan` | Model/runtime error before parse |
| `parse` | `model_output_invalid` / schema failure |
| `resolve` | Wrong or empty resolved entity set (including ambiguity clarification) |
| `safety` | Barrier fired; no HA request |
| `request` | Action requested (fake or live) |
| `verify` | Result correlation / state verification |
| `schema` | Eval case or executor exception |

Invalid model text stays `no-action` / `model_output_invalid`. Never repair JSON.

---

## Phase R — Voice-path stabilization

Recorded-file audio must be a first-class input to the existing text controller.

### R1. Live ConversationStore

**Failing test.** `create_aiohttp_app` default live controller has `conversation_store is None`.

**Implement.** In `create_aiohttp_app`, construct `ConversationStore()` and pass it to `create_live_text_controller` unless the caller injected a controller. Do not change TTL semantics.

**Verify.** `uv run pytest -q sayso-server/src/sayso_server/test_app.py sayso-server/src/sayso_server/test_main.py`

### R2. Audio `input_type`

**Failing test.** Audio requests emit telemetry with `input_type="text"`.

**Implement.** `InteractionTelemetryRecord.input_type` is `Literal["text", "audio"]`. `InteractionTelemetry` accepts `input_type`. `VoicePipelineController` must cause the text controller to record `audio` (add an `input_type` argument on `handle` / `handle_async`; default `"text"`).

**Verify.** `uv run pytest -q sayso-server/src/sayso_server/test_telemetry.py sayso-server/src/sayso_server/test_audio_api.py`

### R3. Separate `stt_ready`

**Failing test.** Readiness snapshot has no STT flag; successful preload is invisible.

**Implement.** `ReadinessSnapshot.stt_ready: bool = False` and `ReadinessState.set_stt_ready`. `__main__._preload_stt_runtime` sets it true only after `stt_runtime.load()` succeeds. **`/ready` stays `model_ready and ha_connected`.** Do not fail LFM ready when Whisper is missing. Include `stt_ready` on health/ready JSON bodies.

**Verify.** `uv run pytest -q sayso-server/src/sayso_server/test_readiness.py sayso-server/src/sayso_server/test_main.py`

### R4. Fixture PCM through the voice pipeline

**Failing test.** `evals/fixtures/audio_pcm16_mono_16k.bin` never reaches `VoicePipelineController`.

**Implement.** Colocated test: Fake STT returns the fixture’s expected transcript; Fake text controller records the text; pipeline handle equals the text path. No live Whisper, no live mic.

**Verify.** `uv run pytest -q sayso-server/src/sayso_server/test_audio_api.py`

---

## Phase L — Failure-ledger collection

### L1. Failure fields on `EvalRecord`

**Failing test.** Records cannot store `failure_stage` / `failure_reason`.

**Implement.** Optional `failure_stage: str | None` and `failure_reason: str | None` on `EvalRecord`. Allowed stages are the pipeline names above plus `schema`. Existing scorer fixtures still validate.

**Verify.** `uv run pytest -q evals/test_metrics.py evals/test_schema.py`

### L2. Classifier

**Failing test.** A schema-invalid plan and a missing candidate classify as the same failure.

**Implement.** `evals/ledger.py`: `classify_failure(case, record) -> tuple[str, str] | None` (stage, reason). Priority: schema/executor error → parse/schema_failure → stt → retrieve → plan outcome vs expected → resolve set mismatch → safety/false execution → verify. Success returns `None`.

**Verify.** `uv run pytest -q evals/test_ledger.py`

### L3. Ledger summary

**Failing test.** No per-stage counts keyed by `case_id`.

**Implement.** `ledger_entries(cases, records)` and `ledger_summary(entries)` → counts by stage/reason plus `case_ids` lists. Deterministic sort.

**Verify.** `uv run pytest -q evals/test_ledger.py`

---

## Phase B — Baseline evals

### B1. Controller dry-run executor

**Failing test.** `dry_run_executor` leaves `recorded_control_plan` empty for an actionable core case.

**Implement.** `controller_dry_run_executor` in `evals/executor.py`:

- Load `evals/fixtures/home_graph.json`.
- `FakeModelRuntime` + `compose_plan_generation` + `execute_control_plan` with `FakeHaClient`.
- Fill `recorded_control_plan`, `recorded_candidate_entities`, `recorded_resolved_entities`, `recorded_query_answer` when present.
- **`ha_executed` is always false.** Fake client calls are not live HA.
- Set `schema_failure` when the plan is `model_output_invalid`.
- `classify_failure` populates ledger fields.

Keep `dry_run_executor` as the no-op used by the live-safety gate when `--execute` is off for unknown executors. The CLI default for this plan is the controller executor.

**Verify.** `uv run pytest -q evals/test_executor.py evals/test_runner.py`

### B2. Eval CLI

**Failing test.** `python -m evals` does not exist.

**Implement.** `evals/__main__.py`:

```text
python -m evals --corpus core --output evals/reports/core.jsonl
```

`--corpus` is one of `core`, `safety`, `language_noise`, `followup`, or `all`. Default executor is `controller_dry_run_executor`. Pass `--execute` and `--allowlist entity_id,...` into `run_benchmark` (existing gate). Write JSONL under `evals/reports/` (gitkeep the directory; do not commit live reports).

**Verify.** `uv run pytest -q evals/test_main.py` (tmp_path, tiny case list). Manual: `uv run python -m evals --corpus core --output /tmp/sayso-core.jsonl` must exit 0.

### B3. Frozen fake-runtime baseline

**Failing test.** Scoring the controller executor against a checked-in slice has no golden assertion.

**Implement.** `evals/fixtures/baseline_core_slice.json` (small authored subset or first N core cases with expected metric numerators for FakeModelRuntime). `evals/test_baseline.py` runs the controller executor and asserts the golden numerators. This is the **reliability baseline**, not an LFM accuracy claim.

**Verify.** `uv run pytest -q evals/test_baseline.py`

---

## Phase M — Benchmark execution

No second-model bake-off. LFM configuration and a runnable harness only.

### M1. Run configuration

**Failing test.** Benchmark metadata cannot record model id / quantization / seed / warmup.

**Implement.** `evals/config.py` `BenchmarkConfig` (frozen): `model_id`, `quantization`, `runtime`, `revision`, `seed`, `warmup_count`, `cold_start`. Default model id matches the server LFM checkpoint (`mlx-community/LFM2.5-230M-OptiQ-4bit`). Serialize onto the first JSONL run header or a sidecar `*.config.json`.

**Verify.** `uv run pytest -q evals/test_config.py`

### M2. Token and latency fields on run rows

**Failing test.** Runner JSONL has only `total_ms`.

**Implement.** Extend `CaseTiming` with optional stage milliseconds (`stt_ms`, `retrieve_ms`, `plan_ms`, `resolve_ms`, `validate_ms`, `request_ms`, `verify_ms`) and optional `prompt_tokens` / `completion_tokens` / `model_id`. `_record_to_jsonl` includes present fields. Controller executor copies model telemetry from `PlanGenerationResult` when available.

**Verify.** `uv run pytest -q evals/test_runner.py evals/test_executor.py`

### M3. Optional live MLX executor

**Failing test.** Missing `mlx-lm` aborts collection of the fake-runtime path.

**Implement.** `evals/mlx_executor.py` builds `MlxModelRuntime` only when import succeeds; otherwise tests skip. Live path is opt-in (`SAYSO_EVAL_MLX=1`). CI always uses FakeModelRuntime. Do not load a second checkpoint.

**Verify.** `uv run pytest -q evals/test_mlx_executor.py`

---

## Phase F — Measured fixes

Only fix defects the ledger or baseline proves. These two are already proven by architecture/code inspection; implement them after R/L/B so tests can show the before/after.

### F1. Persist live telemetry

**Failing test.** Live app has `telemetry_sink is None`, so stage timings are discarded.

**Implement.** If `SAYSO_TELEMETRY_PATH` is set, `create_aiohttp_app` opens a `JsonlTelemetrySink` on that path and passes it into `create_live_text_controller`. Unset → no sink (current behavior). Do not log raw audio.

**Verify.** `uv run pytest -q sayso-server/src/sayso_server/test_app.py sayso-server/src/sayso_server/test_telemetry.py`

### F2. STT stage timing on the audio path

**Failing test.** Audio pipeline has no `stt` stage in telemetry.

**Implement.** Add `stt` to `STAGE_NAMES` / `StageTimings` (0 for text). `VoicePipelineController` times STT and, if the text controller exposes it, records `stt_ms`. Text path remains 0.

**Verify.** `uv run pytest -q sayso-server/src/sayso_server/test_telemetry.py sayso-server/src/sayso_server/test_audio_api.py`

If B3’s ledger shows a **code** failure class (wrong resolver set on the fixture graph, missing candidate for an authored name), add **one** colocated failing test copied from that `case_id` and fix the shared function. Do **not** “fix” model JSON quality, raise candidate `limit` above 1, or bypass the parser.

---

## Phase P — Latency profiling

### P1. Latency report

**Failing test.** A list of timed records cannot produce median/p95.

**Implement.** `evals/latency.py`: `latency_report(rows, *, warm_only=True)` returns count, median, p95 for `total_ms` and each stage field that is present. Empty input → zeros with `n=0`. Percentiles are nearest-rank.

**Verify.** `uv run pytest -q evals/test_latency.py`

### P2. Statistical report blob

**Failing test.** Report hides sample size or mixes cold start with warm.

**Implement.** `build_eval_report(score, ledger_summary, latency, config)` → dict with metric numerators/denominators, failure counts by stage, latency `n`/median/p95, and config metadata. CLI `--report` writes `evals/reports/<name>.report.json`. No LFM-vs-Gemma tables.

**Verify.** `uv run pytest -q evals/test_report.py`

---

## Phase G — Gated expansion

### G1. Expansion gate

**Failing test.** Expansion is allowed when safety dry-run false-execution is non-zero.

**Implement.** `evals/gate.py` `expansion_allowed(score, ledger_summary, latency) -> tuple[bool, list[str]]`.

Pass only when all hold:

- `false_execution_rate == 0` (denominator may be 0 → fail closed with a reason)
- `wrong_device_rate == 0` on the scored dry-run set
- `latency.n >= 1` for the fake-runtime baseline run
- ledger has no unclassified `schema` executor crashes on the committed corpora

CLI `--check-gate` exits 1 when blocked. **This plan does not add JSONL cases.** Tasks 46–47 (second model), 49–51 (Home-LLM/Alexa+/full statistical bake-off), and 58–64 (wake/VAD) stay deferred until this gate passes **and** `AGENTS.md` is explicitly updated.

**Verify.** `uv run pytest -q evals/test_gate.py`

### G2. Architecture pointer

**Failing test.** `docs/ARCHITECTURE.md` still lists eval/reliability work as undifferentiated “planned.”

**Implement.** In the remaining-plan section, point at this file for voice-path eval/reliability; keep bake-offs and wake/VAD in the deferred table.

**Verify.** `rg MVP_RELIABILITY_AND_EVALUATION_PLAN docs/ARCHITECTURE.md`

---

## Order and stop conditions

Implement **R1 → R4 → L1 → L3 → B1 → B3 → M1 → M3 → F1 → F2 → P1 → P2 → G1 → G2**.

Stop the whole plan when G2 is done and `uv run pytest -q evals sayso-server/src/sayso_server/test_app.py sayso-server/src/sayso_server/test_telemetry.py sayso-server/src/sayso_server/test_audio_api.py sayso-server/src/sayso_server/test_readiness.py sayso-server/src/sayso_server/test_main.py` passes.

Do not start wake/VAD, a second model adapter, or new corpus files inside this plan.

## Decision gates (this plan vs EVALUATION_PLAN.md)

| Gate | This plan’s bar |
|---|---|
| Retrieval (A) | Fake-runtime baseline records candidate sets; recall is scored |
| Model (B) | Invalid JSON → no-action; LFM config exists; live MLX optional |
| Deterministic safety (C) | Dry-run false-execution 0 on scored safety cases; live still needs `--execute` + allowlist |
| Voice (D) | Recorded fixture hits the same controller; STT timed; no wake loop |
| Product claim (E) | **Not in scope** — no Home-LLM/Alexa+ comparison |
