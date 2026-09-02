# SaySo

Local, Alexa-style smart-home voice assistant. Python monorepo for
`sayso-server`, the Home Assistant integration in `custom_components/sayso`,
and the Mac `sayso-satellite` client.

Runtime topology, invariants, and component boundaries:
[Architecture](docs/ARCHITECTURE.md). Evaluation corpora, metrics, and gates:
[Evaluation plan](docs/EVALUATION_PLAN.md).

## Prerequisites

- **macOS** for live satellite capture (`ffmpeg`) and TTS playback (`afplay`).
- **Python 3.14.2+** and **[uv](https://docs.astral.sh/uv/)**.
- **`mlx-lm`** on the machine that runs `sayso-server` (Apple Silicon Mac
  recommended). Not in the locked workspace deps; install separately, e.g.
  `uv pip install mlx-lm`.
- **Home Assistant** running locally with Assist enabled and a long-lived access
  token (Profile → Security → Long-Lived Access Tokens). Do not commit tokens.
- **Home Assistant config directory** (`$HA_CONFIG`, often
  `$HOME/.homeassistant` or the path shown in your HA install docs).

## Bootstrap

From a clean checkout:

```bash
uv sync --frozen
uv run pytest -q
uv run pytest -q evals
python -c 'import sayso_server, sayso_satellite'
```

## 1. Start the server

Create a server bearer token and export it (same value is entered in the HA
integration UI):

```bash
export SAYSO_TOKEN='<server-token>'
uv run python -m sayso_server
```

Defaults: `http://127.0.0.1:8765` (`SAYSO_HOST`, `SAYSO_PORT`). Optional model
override: `SAYSO_MODEL_ID` (see [Architecture](docs/ARCHITECTURE.md)).

**Readiness before Home Assistant connects** — liveness OK, readiness not yet
(503 is expected until the integration connects and pushes a graph):

```bash
curl -fsS -H "Authorization: Bearer $SAYSO_TOKEN" \
  http://127.0.0.1:8765/api/v1/health
curl -fsS -H "Authorization: Bearer $SAYSO_TOKEN" \
  http://127.0.0.1:8765/api/v1/ready || true
```

## 2. Connect Home Assistant

Install the integration (symlink or copy), then restart Home Assistant:

```bash
export HA_CONFIG="$HOME/.homeassistant"   # adjust to your install
ln -sf "$(pwd)/custom_components/sayso" "$HA_CONFIG/custom_components/sayso"
```

In Home Assistant:

1. **Settings → Devices & services → Add integration → SaySo**
2. **Server URL:** `http://127.0.0.1:8765` (or your host/port)
3. **Access token:** the same value as `SAYSO_TOKEN`
4. **Configure options** (exposure, domain/action allowlists) before live
   device control — see integration options in the UI.
5. **Settings → Voice assistants:** create or edit an Assist pipeline that uses
   the SaySo conversation agent.

After the integration connects, readiness should report `"ready": true`:

```bash
curl -fsS -H "Authorization: Bearer $SAYSO_TOKEN" \
  http://127.0.0.1:8765/api/v1/ready
```

Also check **SaySo → Connection** (`binary_sensor`) in Home Assistant.

## 3. Run the Mac satellite

Export a Home Assistant long-lived token (not the server token):

```bash
export SAYSO_HA_TOKEN='<ha-long-lived-token>'
export SAYSO_HA_WEBSOCKET_URL='ws://127.0.0.1:8123/api/websocket'   # if not default
```

**Deterministic replay** (no microphone; verifies Assist → SaySo → response path):

```bash
uv run python -m sayso_satellite \
  --audio-file evals/fixtures/audio_pcm16_mono_16k.bin
```

**Live voice loop** (requires `ffmpeg`, registered HA device for origin area):

```bash
export SAYSO_HA_DEVICE_ID='<ha-device-id>'   # optional; CLI --device-id also works
uv run python -m sayso_satellite --live --wake --loop
```

Optional tuning: `SAYSO_WAKE_THRESHOLD`, `SAYSO_WAKE_HITS`, `SAYSO_LISTEN_MS`,
`--capture-ms`. See [Architecture](docs/ARCHITECTURE.md) for the full voice
path.

## 4. Verify

**Automated suites** (repeat after changes):

```bash
uv run pytest -q
uv run pytest -q evals
```

**Basic evaluation gate** (dry-run by default; no live actuation):

```bash
uv run python -m evals --corpus all --output /tmp/sayso-all.jsonl --report
uv run python -m evals --corpus all --output /tmp/sayso-all.jsonl --check-gate
```

Reports land under `evals/reports/`; do not commit local report output. Live
execution requires both `--execute` and `--allowlist` — see
[Evaluation plan](docs/EVALUATION_PLAN.md).

## Further reading

- [Architecture](docs/ARCHITECTURE.md)
- [Evaluation plan](docs/EVALUATION_PLAN.md)
- [WebSocket conversation plan](docs/WEBSOCKET_CONVERSATION_PLAN.md)
- [Tuning plan](docs/TUNING_PLAN.md)
- [Clean rebuild plan](docs/CLEAN_REBUILD_PLAN.md)
