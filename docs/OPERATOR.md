# SaySo operator runbook

This runbook covers the three runtime processes that make up a local SaySo
deployment: the **SaySo server**, the **Home Assistant integration**, and the
**Mac satellite**. A fresh operator can follow it to bring up the stack and run
**health → HA connect → command** without tribal knowledge.

See also: [Architecture](ARCHITECTURE.md) for design detail and invariants.

## Three processes

| Process | How it runs | Role |
|---|---|---|
| SaySo server | `python -m sayso_server` | HTTP API, local model, Home Graph replica, HA WebSocket gateway |
| HA integration | Loaded by Home Assistant from `custom_components/sayso/` | Outbound WebSocket client, graph sync, action execution, state verification |
| Mac satellite | `python -m sayso_satellite …` | Sends text or recorded PCM to the server; renders the response policy |

These are separate processes. There is no single launcher or Docker image.

```text
Mac satellite ── HTTP ──▶ SaySo server ◀── WebSocket ── HA integration ──▶ Home Assistant
```

## Prerequisites

From the repository root:

```bash
uv sync
```

`uv sync` installs core Python dependencies only (`aiohttp`, `pydantic`, etc.).
**MLX packages are not pinned in `pyproject.toml`.** Install them separately
into the same environment as optional runtime packages:

```bash
uv pip install mlx-lm mlx-whisper
```

| Package | Required for | If missing |
|---|---|---|
| `mlx-lm` | Server startup and text commands (local LFM) | `python -m sayso_server` exits with an install hint |
| `mlx-whisper` | Recorded-audio STT (`POST /api/v1/audio`) | Server still starts; `stt_ready` stays `false`; audio returns a classified **no-action** response (HTTP **200**, not **500**) |

Whisper preload runs at server startup after the LFM loads. A failed preload
prints a warning to stderr and leaves `stt_ready: false`. That does **not**
block `GET /api/v1/ready` — aggregate readiness remains
`model_ready AND ha_connected`.

The satellite only needs HTTP client dependencies. Home Assistant must have the SaySo integration available under
`custom_components/sayso/` (copy or symlink into your HA `config/` tree, or
install via HACS when published).

Generate or choose a shared bearer secret and configure it consistently across
all three processes (see environment variables below). **Do not commit secret
values** — set them in your shell, a local env file that is gitignored, or HA
config-entry storage.

---

## Environment variables (names only)

Document **names** here; set values locally. Never copy values from `.env` into
docs or commits.

### SaySo server

| Variable | Required | Purpose |
|---|---|---|
| `SAYSO_TOKEN` | yes | Bearer token for all HTTP and WebSocket surfaces |
| `SAYSO_HOST` | no | Bind address (default `127.0.0.1`) |
| `SAYSO_PORT` | no | Listen port (default `8765`) |
| `SAYSO_TELEMETRY_PATH` | no | When set, append JSONL interaction telemetry to this file |
| `SAYSO_SATELLITE_AREA_ID` | no | HA area id for the default `macbook` satellite (default `area_living_room`) |

### Mac satellite

| Variable | Required | Purpose |
|---|---|---|
| `SAYSO_TOKEN` | yes | Same bearer token as the server |
| `SAYSO_SERVER_URL` | no | Server base URL (default `http://127.0.0.1:8765`) |
| `SAYSO_TIMEOUT_SECONDS` | no | HTTP timeout in seconds for text and audio (default `180`) |

The satellite CLI also accepts `--timeout SECONDS`, which overrides the env var
for that invocation.

### Home Assistant integration

The integration stores **Server URL** and **Access token** in the config entry
(not environment variables). The token must match `SAYSO_TOKEN` on the server.

---

## 1. Start the SaySo server

```bash
export SAYSO_TOKEN=<your-secret>
# optional: SAYSO_HOST, SAYSO_PORT, SAYSO_TELEMETRY_PATH, SAYSO_SATELLITE_AREA_ID
python -m sayso_server
```

The server loads the local model, starts aiohttp on `SAYSO_HOST`/`SAYSO_PORT`,
and waits for the HA integration to connect on `/api/v1/ws`.

Until HA connects and sends a graph snapshot, live commands are refused even if
the model is loaded.

---

## 2. Configure Home Assistant

### Install the integration

Ensure `custom_components/sayso/` is on Home Assistant’s integration path, then
add **SaySo** via **Settings → Devices & services → Add integration**.

During setup you provide:

- **Server URL** — base URL of the running server (e.g. `http://127.0.0.1:8765`)
- **Access token** — must match `SAYSO_TOKEN`

Setup probes `GET /api/v1/health` with Bearer auth before creating the entry.

### Options: exposure and permissions

After setup, open **Configure** on the SaySo integration entry:

| Option | Meaning |
|---|---|
| **Allowed domains** | Empty = all domains; otherwise only listed domains (e.g. `light`, `switch`) |
| **Allowed actions** | Empty = all actions; otherwise only listed actions (e.g. `on`, `off`, `toggle`, `query`) |
| **Entity exposure** | `All entities`, `Selected areas`, or `Selected entities` |
| **Included areas** | Used when exposure mode is *Selected areas* |
| **Included entities** | Used when exposure mode is *Selected entities* |

Only exposed entities appear in the Home Graph snapshot sent to the server.
The integration re-checks exposure, domain, action, and capability at execution
time even when the server’s plan is valid.

### Manual graph sync

The integration pushes a snapshot automatically after connect. You can also call
the **Sync home graph** service (`sayso.sync_home_graph`) to force a fresh
snapshot after registry or exposure changes.

### HA naming and aliasing (one lamp, one name)

SaySo resolves **semantic names** from the Home Graph — entity friendly names
and registry aliases — not raw entity ids in user speech.

If several lamps share overlapping tokens (e.g. multiple switches whose names
contain “lamp”), a phrase like “turn off the floor lamp” correctly **asks for
clarification** instead of picking arbitrarily.

To make a named command hit **one** entity:

1. In Home Assistant, open the target entity (e.g. the physical floor lamp).
2. Either **rename** it to a unique name (e.g. *Floor Lamp*), or add a unique
   **alias** (e.g. `floor lamp`) on **that entity only**.
3. Reload the SaySo integration or call **Sync home graph** so the server
   receives an updated snapshot.

Verify:

- `python -m sayso_satellite "turn off the floor lamp"` completes on that one
  entity.
- If two entities still share the same name or alias, equal matches still
  **clarify** — that is expected safety behavior.

Do not “fix” ambiguity by changing model prompts; fix naming in HA.

---

## 3. Health and readiness

Both endpoints require Bearer auth:

```http
Authorization: Bearer <SAYSO_TOKEN>
```

### `GET /api/v1/health`

Liveness probe. Returns HTTP **200** when auth succeeds.

Example body fields:

```json
{
  "status": "ok",
  "liveness": "ok",
  "model_ready": true,
  "ha_connected": true,
  "stt_ready": false
}
```

Use this to confirm the process is up and to inspect dependency flags. Invalid
or missing auth returns **401**.

### `GET /api/v1/ready`

Readiness probe for “safe to execute live commands.”

- HTTP **200** only when `ready` is true.
- HTTP **503** when the server is up but not ready.

Body:

```json
{
  "ready": true,
  "model_ready": true,
  "ha_connected": true,
  "stt_ready": false
}
```

**Aggregate readiness:** `ready = model_ready AND ha_connected`.

- `model_ready` — local LFM loaded.
- `ha_connected` — HA WebSocket attached **and** a graph snapshot applied.
- `stt_ready` — Whisper preload succeeded; **reported but does not gate**
  `/api/v1/ready`. Expect `false` when `mlx-whisper` is not installed or
  preload failed; `/ready` can still return HTTP **200** once model and HA are
  ready. Install `mlx-whisper` (see [Prerequisites](#prerequisites)) to enable
  audio transcription.

Quick check from the shell (replace URL and token):

```bash
curl -s -H "Authorization: Bearer $SAYSO_TOKEN" \
  http://127.0.0.1:8765/api/v1/health | jq .

curl -s -o /dev/null -w "%{http_code}\n" \
  -H "Authorization: Bearer $SAYSO_TOKEN" \
  http://127.0.0.1:8765/api/v1/ready
```

Expect **200** on `/ready` only after HA has connected and synced.

---

## 4. Graph resync after disconnect or restart

The server keeps an **in-memory** Home Graph. It is authoritative for planning
only while HA is connected and the snapshot is current.

| Event | Server behavior |
|---|---|
| HA WebSocket disconnect | Graph cleared; `ha_connected` false; live execution refused |
| HA reconnect | Integration reconnects with backoff; sends fresh `graph_snapshot` |
| Server restart | Model, graph, session, and readiness reset; HA must reconnect |
| Stale / out-of-order deltas | Rejected; snapshot replacement is atomic |

Live text and audio commands return refusal codes such as `no_graph` or
`ha_disconnected` until a **fresh snapshot** arrives after reconnect.

**Operator checklist after any restart:**

1. Confirm the SaySo server process is running.
2. Confirm Home Assistant loaded the SaySo integration (no error on the entry).
3. `GET /api/v1/ready` → HTTP 200 and `ha_connected: true`.
4. Run a command.

If `/ready` stays 503 with `ha_connected: false`, check HA logs and that URL
and token match. Use **Sync home graph** after fixing exposure or registry
changes.

---

## 5. Mac satellite commands

Export the same token (and optional URL / timeout) in the satellite shell:

```bash
export SAYSO_TOKEN=<your-secret>
export SAYSO_SERVER_URL=http://127.0.0.1:8765   # optional
export SAYSO_TIMEOUT_SECONDS=180                 # optional
```

### Text command

```bash
python -m sayso_satellite "turn off the corner lamp"
```

The satellite POSTs to `/api/v1/text` with `satellite_id` default `macbook`.
Successful control actions usually render an **earcon**; queries and errors
render short text (same response policy as audio).

### Recorded PCM (file voice)

Format: **16 kHz, mono, PCM16** (raw s16le bytes, no WAV header).

Committed fixture:

```text
evals/fixtures/audio_pcm16_mono_16k.bin
```

Run:

```bash
python -m sayso_satellite --audio-file evals/fixtures/audio_pcm16_mono_16k.bin
```

This POSTs to `/api/v1/audio`, runs STT, then the **same text controller** as
CLI text. Response rendering uses the same policy.

**Without `mlx-whisper` installed:** the server accepts the audio request but
cannot transcribe. The response is HTTP **200** with `category: "no_action"`
(and a reason mentioning speech-to-text unavailability) — **not** HTTP **500**.
No Home Assistant action runs. Install `mlx-whisper` and restart the server
(or rely on lazy load after install) before expecting voice commands to execute.

**Success criteria for a physical-device check:**

- Same outcome category as the equivalent text command (e.g. `completed` /
  `state_changed` on the intended entity), **or**
- A clear **STT transcript mismatch** reported in the response — not a silent
  failure.

**Failure modes that are not success:**

- HTTP timeout (raise default timeout with `SAYSO_TIMEOUT_SECONDS` or
  `--timeout` if Whisper + model exceed the limit)
- Refusal while graph is unavailable (`no_graph`, `ha_disconnected`)
- HTTP 503 on `/ready` — do not expect commands to succeed until resync completes

Optional: compare transcript expectations using
`evals/fixtures/stt_clip.json` (`expected_transcript` field) when validating
STT quality in evals; live home commands should use utterances that match your
configured entity names.

---

## 6. End-to-end operator path

Follow this sequence on a fresh machine:

### Step A — Install and configure secrets

1. `uv sync` in the repo, then `uv pip install mlx-lm mlx-whisper` for MLX
   runtime support (see [Prerequisites](#prerequisites)).
2. Choose one bearer secret; set `SAYSO_TOKEN` for the server and satellite
   shells. Enter the same value as the HA integration **Access token**.
3. Optionally set `SAYSO_HOST`, `SAYSO_PORT`, `SAYSO_TELEMETRY_PATH`,
   `SAYSO_SATELLITE_AREA_ID`, `SAYSO_SERVER_URL`, `SAYSO_TIMEOUT_SECONDS`.

### Step B — Start server

```bash
python -m sayso_server
```

Wait for model load to finish (watch process output).

### Step C — Health (server only)

```bash
curl -s -H "Authorization: Bearer $SAYSO_TOKEN" \
  http://127.0.0.1:8765/api/v1/health
```

Expect `"liveness": "ok"`. `ha_connected` may still be `false`.

### Step D — Connect Home Assistant

1. Install `custom_components/sayso/` on HA.
2. Add integration: Server URL + Access token.
3. Configure exposure/permissions for the devices you want voice control over.
4. Name or alias one test lamp uniquely (see [HA naming](#ha-naming-and-aliasing-one-lamp-one-name)).

### Step E — Ready (server + HA)

```bash
curl -s -H "Authorization: Bearer $SAYSO_TOKEN" \
  http://127.0.0.1:8765/api/v1/ready
```

Expect HTTP **200**, `"ready": true`, `"model_ready": true`,
`"ha_connected": true`. If 503, wait for reconnect or trigger **Sync home
graph**.

### Step F — Command

Text:

```bash
python -m sayso_satellite "turn off the corner lamp"
```

Or recorded audio:

```bash
python -m sayso_satellite --audio-file evals/fixtures/audio_pcm16_mono_16k.bin
```

Confirm the target entity changed in Home Assistant (or read the satellite
response / HA log for `completed` / `state_changed`).

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Server exits immediately | Missing `SAYSO_TOKEN` | Set token and restart |
| Server exits on start | Missing `mlx-lm` | `uv pip install mlx-lm` in the project env |
| `/ready` 200 but `stt_ready: false` | `mlx-whisper` not installed or preload failed | `uv pip install mlx-whisper`; restart server; `/ready` unaffected |
| Audio returns `no_action`, not 500 | STT unavailable (Whisper missing) | Install `mlx-whisper`; expect classified no-action until then |
| `/health` 401 | Token mismatch | Align server, satellite, and HA entry |
| `/ready` 503, `ha_connected: false` | Integration not connected | Check HA entry, URL, firewall; reload integration |
| Command refused `no_graph` | Snapshot not yet applied | Wait for connect or sync home graph |
| Command refused `ha_disconnected` | WebSocket dropped | Wait for HA reconnect; verify `/ready` |
| Clarification instead of action | Ambiguous names | Rename or alias one entity in HA; resync |
| Action rejected at HA | Exposure or permission | Adjust integration options |
| Audio times out | STT + model slow | Increase `SAYSO_TIMEOUT_SECONDS` or `--timeout` |
| Audio wrong device / plan | STT transcript drift | Treat as transcript mismatch; re-record or use text |

---

## Related docs

- [Architecture](ARCHITECTURE.md) — contracts, control path, failure barriers
- [NEXT.md](NEXT.md) — post-demo tasks and naming notes
- [MVP plan](MVP_PLAN.md) — numbered implementation units
