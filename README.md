# SaySo

SaySo is a fully local [Home Assistant](https://www.home-assistant.io/) voice assistant. Home Assistant is authoritative for the voice pipeline, entities, context, tools, and action execution. SaySo runs [llama.cpp](https://github.com/ggerganov/llama.cpp) on the CPU inside the integration itself for language understanding and tool selection, then executes validated actions through Home Assistant’s native LLM tool API. A user-managed llama.cpp server is supported as an advanced alternative.

SaySo has two independently deployable components: the Home Assistant conversation integration and an optional reference satellite under `satellite/` for local `SaySo` wake-word detection on OHF Voice’s Linux Voice Assistant. The SaySo integration does not manage or require the bundled satellite; any compatible Home Assistant voice satellite works. Home Assistant owns STT, TTS, pipeline orchestration, and smart-home actions. See [ARCHITECTURE.md](ARCHITECTURE.md) for runtime boundaries.

## Requirements

- Home Assistant 2026.8.3 or newer
- About 1 GB of free space in `/config` and at least 1 GB of free RAM recommended
- No server, URL, port, or API key

On first setup SaySo installs a prebuilt `llama-cpp-python` wheel and downloads
**SaySo LFM v5b** (`SaySo-LFM-v5b-F16.gguf`, 462 MB, an LFM2.5-230M
fine-tune) into `/config/sayso/models/`. The download is checksum-verified.
The model stays in memory while SaySo is loaded and runs on a dedicated worker
thread, off Home Assistant’s event loop.

Prebuilt wheels exist for Home Assistant OS on amd64 and arm64. Verify your
platform before relying on it:

```bash
scripts/verify_embedded_backend.sh
```

## Install as a custom integration

### HACS

1. Add this repository as a [custom HACS integration](https://hacs.xyz/docs/faq/custom_repositories/).
2. Install **SaySo** from HACS.
3. Restart Home Assistant.

### Manual

1. Copy the `custom_components/sayso` folder into your Home Assistant `config/custom_components/` directory.
2. Restart Home Assistant.

## Configure SaySo in Home Assistant

1. Go to **Settings → Devices & services → Add integration**.
2. Search for **SaySo**.
3. Leave the URL field empty and submit. That is the whole local setup.
4. Adjust options (CPU threads, context window, system prompt, temperature,
   token limits, tool iterations, trace retention) from the **Configure** menu.

First setup downloads a few hundred megabytes, so the entry may sit in
"retrying" for a few minutes. To use your own weights instead, drop a `.gguf`
into `/config/sayso/models/` and set **Model file** in the options.

## Advanced: use an external llama.cpp server

Supply a base URL during setup to use a user-managed server instead of embedded
inference. Start `llama-server` on a host Home Assistant can reach:

```bash
llama-server \
  --model /models/model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --jinja \
  --api-key replace-with-a-local-secret
```

- `--jinja` enables the chat template required for tool calling.
- `--api-key` is optional; if set, enter the same value in the SaySo config flow.

Use a model whose template supports tool/function calling. SaySo sends requests to the OpenAI-compatible `/v1/chat/completions` endpoint. Enter the base URL (for example `http://127.0.0.1:8080/v1`) and optional API key during setup, then select the model identifier your server exposes.

## Use SaySo as the conversation agent

1. Open **Settings → Voice assistants** (or **Settings → Assist**).
2. Create or edit an assistant.
3. Set **Conversation agent** to **SaySo** (the entry title shows your model and host).
4. Expose only the entities that assistant should control.

With a standard Home Assistant voice pipeline, wake word → STT → SaySo → TTS → satellite playback stays entirely inside Home Assistant. With embedded inference there is no network call at all.

The optional reference satellite under `satellite/` uses Home Assistant’s standard voice pipeline. Linux Voice Assistant owns capture, volume normalization, WebRTC processing, and HA transport; the SaySo overlay receives the same processed PCM for LiveKit wake detection via LVA’s external wake hook. It does not perform speech recognition, language understanding, model inference, or Home Assistant actions, and it never connects directly to llama.cpp.

## Diagnostics

Download config entry diagnostics from the SaySo integration page. API keys and other configured secrets are redacted automatically.

## Interaction traces

SaySo records one trace per voice interaction: wake, audio transport, STT, context construction, model inference, tool parsing, Home Assistant action execution, response, and TTS. Traces are stored separately from `home-assistant.log` and are retrievable from **Developer tools → Actions**:

- `sayso.get_trace` — one trace with its chronological stage events.
- `sayso.list_traces` — recent interaction summaries, with `only_failures`, `error_stage`, `start_time`, and `end_time` filters.

Both return a response; use **Perform action** with *Return response* enabled.

Traces never contain audio, prompts, tool schemas, or credentials. Transcripts are stored by default and can be turned off with **Store transcribed utterances in traces** without losing timings, tool, target, or failure information. Retention defaults to 30 days or 500 interactions, whichever comes first.

To attach trace IDs to ordinary logs for troubleshooting:

```yaml
logger:
  logs:
    custom_components.sayso: debug
```

## Publishing a model

Weights are never committed: every useful quant is over GitHub's 100 MB file
limit, and HACS re-downloads the integration directory on every update. Ship the
GGUF as a Release asset instead. From the machine holding the model:

```bash
TITLE="SaySo Gauntlet v2" scripts/publish_model.sh /srv/llm/lfm/runs/<run>/<new>.gguf model-v2 --dry-run
TITLE="SaySo Gauntlet v2" scripts/publish_model.sh /srv/llm/lfm/runs/<run>/<new>.gguf model-v2
```

It uploads the asset, pins its SHA-256, and repoints `DEFAULT_MODEL_URL`,
`DEFAULT_MODEL_FILENAME`, and `DEFAULT_MODEL_SHA256` in `const.py`. Commit that
diff. The `model-v*` tag is deliberately separate from the release-please
version tags so the model and the integration version independently.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -q
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for runtime boundaries and
[docs/SAYSO_TRAINING_LIFECYCLE.md](docs/SAYSO_TRAINING_LIFECYCLE.md) for the local training and promotion workflow,
[docs/SAYSO_LFM_TRAINING_PLAN.md](docs/SAYSO_LFM_TRAINING_PLAN.md) for SaySo model training design.
Wake-word training is planned separately in
[docs/SAYSO_WAKE_WORD_TRAINING_PLAN.md](docs/SAYSO_WAKE_WORD_TRAINING_PLAN.md).
