# Plan: embedded CPU inference in the SaySo integration

**Status:** implemented and verified on both architectures.
**Gate:** `scripts/verify_embedded_backend.sh` — run it before trusting this plan.

Goal: run LFM2.5-230M on the CPU inside the SaySo integration via
llama-cpp-python, so a local install needs no server, URL, port, or API key.
The external OpenAI-compatible backend stays as an advanced fallback.

---

## 1. Dependency gate: result

Verified against the real Home Assistant container images at 2026.8.3, both
architectures, with `scripts/verify_embedded_backend.sh`.

| | amd64 | arm64 |
|---|---|---|
| Image | `ghcr.io/home-assistant/amd64-homeassistant` | `ghcr.io/home-assistant/aarch64-homeassistant` |
| Python / libc | 3.14.6 / musl 1 | 3.14.6 / musl 1 |
| Model load | 501 ms | 111 ms |
| `create_chat_completion` | 1060 ms | 551 ms |
| `general.architecture` | `lfm2` | `lfm2` |
| Result | PASS | PASS |

amd64 timings are under emulation on an arm64 host, so they are an upper bound,
not a benchmark. Model was `LFM2.5-350M-Q4_K_M` (218 MB) as a stand-in for the
230M fine-tune.

### 1.1 The dependency cannot be a manifest requirement

`llama-cpp-python` publishes **sdist only** to PyPI — zero wheels, every
version. The HA container is Alpine/musl with **no compiler** (`gcc`, `cc`,
`cmake`, `make` all absent), so the source build fails at CMake:

```
CMake Error: Could not find the compiler specified in the environment variable CC: gcc
```

Putting `llama-cpp-python` in `manifest.json` `requirements` therefore breaks
setup on HAOS. The verification script keeps this as a control case so the
failure mode stays visible.

### 1.2 What does work

The maintainer publishes prebuilt wheels at
`https://abetlen.github.io/llama-cpp-python/whl/cpu`, including
`musllinux_1_2_x86_64` and `musllinux_1_2_aarch64` — exactly the two HAOS
targets. Since 0.3.33 they are tagged `py3-none-<platform>` (ctypes bindings, no
CPython ABI), so they install on Python 3.14 without per-version rebuilds.

Install must use `--no-deps`. A plain install resolves `numpy` 2.3.2 → 2.5.2,
replacing Home Assistant's pinned numpy for every other integration. With
`--no-deps` (plus `diskcache`) numpy is untouched and the model still loads and
runs — verified on both arches.

### 1.3 Finding that changed the design: no tool-call parsing

The wheels bundle `libllama`/`libggml` only, **not** llama.cpp's
`common/chat.cpp`. `common_chat_parse` is absent from the shared library.

Consequence: `llama-server --jinja`, which the current external backend talks
to, parses LFM2 tool calls into OpenAI `tool_calls` server-side. The Python
bindings do not. With the model's own chat template the call arrives as
assistant **text**:

```
[HassTurnOn(name="kitchen lights", area="kitchen")]
```

`structured_tool_calls=False`. Everything downstream in `conversation.py`
consumes `ChatCompletionResult.tool_calls`, so the embedded engine must parse
that text itself.

Rejected alternatives:

- `chat_format="chatml-function-calling"` does return structured calls, but it
  discards the model's native template. The SaySo fine-tune is trained on the
  LFM2.5 template (`docs/TRAINING_PLAN.md` §3), so this changes the prompt the
  model was trained against.
- Forced `tool_choice={"type": "function", ...}` returns structured calls but
  cannot decline to call a tool, and produced wrong arguments in the probe.

**This is not new work.** `training/evals/lfm_python_parse.py` already parses
exactly this format, apostrophe-safe. It moves into the integration and gets
reused.

This is also a small win: `docs/TRAINING_PLAN.md` §4 records that llama-server's
structured `tool_calls` truncate names like `O'Malley's` and `Kids'` — a known
serving bug. Parsing the raw text with the apostrophe-safe parser avoids it.

---

## 2. Scope

In scope:

- `SaySoInferenceEngine` abstraction with embedded and external implementations.
- Embedded backend: keep the GGUF resident, run off the event loop.
- First-run wheel install and GGUF download into persistent `/config` storage.
- Config flow defaults to local with no connection fields; external moves behind
  an advanced toggle.
- Startup, shutdown, model-load-failure, and inference-error handling.

Explicitly not in scope (per `AGENTS.md`):

- Tool/context generation, schema compilation, routing, validation, correction
  retries, boundary diagnostics, tracing, HA action execution. The engine swap
  is the only change to the request path.
- Streaming, GPU offload, multi-model support, `ARCHITECTURE.md` topology beyond
  the model-hosting row.

---

## 3. Files

New:

| File | Responsibility |
|---|---|
| `custom_components/sayso/inference.py` | `SaySoInferenceEngine` protocol; `EmbeddedEngine`; `ExternalEngine` wrapping the existing `LlamaCppClient` |
| `custom_components/sayso/lfm_parse.py` | Apostrophe-safe LFM2 tool-call parser, moved from `training/evals/lfm_python_parse.py` |
| `custom_components/sayso/model_store.py` | Wheel install + GGUF download/verify under `/config/sayso/` |
| `tests/test_inference.py` | Engine contract, parser, error mapping |

Changed:

| File | Change |
|---|---|
| `__init__.py` | Build the engine, own the executor, shut both down on unload |
| `conversation.py` | `runtime.client.chat_completion(...)` → `runtime.engine.async_chat_completion(...)`. Nothing else. |
| `config_flow.py` | Local-first; URL/key fields behind an advanced external option |
| `const.py` | Backend, model path, thread and context constants |
| `manifest.json` | Requirements stay `[]` — see §1.1 |
| `training/evals/lfm_python_parse.py` | Re-export from the integration so the eval scorer and runtime cannot drift |

`ARCHITECTURE.md` needs one edit: the "Model hosting and lifecycle" ownership row
and invariant 2 currently say llama.cpp is user-managed. Embedded inference moves
hosting into the integration. That is an ownership change, so the document is
updated — narrowly, those two places.

---

## 4. Design

### Engine boundary

```python
class SaySoInferenceEngine(Protocol):
    async def async_chat_completion(
        self, messages, *, tools, temperature, max_tokens
    ) -> ChatCompletionResult: ...
    async def async_start(self) -> None: ...
    async def async_shutdown(self) -> None: ...
```

`ChatCompletionResult` is unchanged, so `conversation.py` keeps its existing
tool-call validation, correction, and boundary handling untouched.

### Threading

One `ThreadPoolExecutor(max_workers=1, thread_name_prefix="sayso_inference")`
owned by the config entry. `Llama` is not thread-safe and a single 230M model
saturates the CPU it is given; a single worker serialises turns and keeps the
KV cache coherent. Inference runs via `hass.async_add_executor_job` against that
executor, never the shared HA pool — a 1-second inference must not occupy a
worker Home Assistant needs.

Threads default to `min(4, os.cpu_count())`, overridable, since HA shares the box.

### Model storage

`/config/sayso/models/<filename>.gguf`, created if missing. `/config` persists
across HA updates; site-packages does not, so the wheel is re-checked at every
startup and the model is downloaded once.

Download is resumable-by-restart: write to `.part`, verify size and SHA-256
against a manifest, then atomic rename. A partial file never loads.

### Failure handling

| Failure | Behaviour |
|---|---|
| Wheel install fails | Entry setup raises `ConfigEntryNotReady`; retried on HA's backoff |
| Model download fails | `ConfigEntryNotReady` with the HTTP cause |
| Checksum mismatch | Delete the file, fail loudly; do not load unverified weights |
| `Llama(...)` raises | `ConfigEntryNotReady`, model file left for inspection |
| Inference raises | Map to `SaySoInvalidResponseError` / `SaySoTimeoutError` so existing boundary diagnostics classify it unchanged |
| Unload | Executor shut down with `wait=True`, then `del llm` to free the model |

Inference timeout uses `asyncio.wait_for`. A llama.cpp call already running
cannot be cancelled mid-token, so the turn returns a timeout error while the
worker finishes; the single-worker queue naturally backpressures the next turn.

---

## 5. Open questions

1. **Default GGUF — resolved 2026-09-15.** Published as GitHub Release
   `model-v1`, asset `SaySo-Gauntlet-v1-Q8_0.gguf` (246.6 MB), sha256 pinned in
   `const.py`. This is Run 013 step-2500, the artifact already serving port 8080
   — a statistical tie with the prior champion, shipped because it is what runs,
   not because it won. Weights cannot be committed: every useful quant exceeds
   GitHub's 100 MB file limit, Git LFS pointers break HACS installs, and a binary
   in git history is permanent. `scripts/publish_model.sh` uploads a new asset
   and rewrites the three constants in one step.

2. **Requirement 1 ("easiest to process info, simple and fast")** overlaps the
   routing and schema-subset work that `AGENTS.md` protects. Needs to be stated
   as a measurable target against `evals/` before anything changes there.

---

## 6. Verification

1. `scripts/verify_embedded_backend.sh` — dependency gate, both arches.
2. `pytest tests/ custom_components/sayso` — existing suite must stay green;
   the engine swap is behind the same `ChatCompletionResult`.
3. `tests/test_inference.py` — parser round-trips the recorded LFM2 outputs
   including apostrophe names; engine maps load and inference failures to the
   existing exception types.
4. `python -m evals.runner` — offline eval scores must not regress against
   `evals/baselines/current.json`.
5. Manual: HAOS install with no llama.cpp server running; confirm setup
   downloads the model, a voice command executes, and reload/unload frees memory.
