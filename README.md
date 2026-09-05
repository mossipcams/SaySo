# SaySo

SaySo is a fully local [Home Assistant](https://www.home-assistant.io/) voice assistant. Home Assistant is authoritative for the voice pipeline, entities, context, tools, and action execution. SaySo connects its conversation agent to your user-managed [llama.cpp](https://github.com/ggerganov/llama.cpp) server for language understanding and tool selection, then executes validated actions through Home Assistant’s native LLM tool API.

SaySo has two independently deployable components: the Home Assistant conversation integration and an optional reference satellite under `satellite/` for local `SaySo` wake-word detection on OHF Voice’s Linux Voice Assistant. The SaySo integration does not manage or require the bundled satellite; any compatible Home Assistant voice satellite works. Home Assistant owns STT, TTS, pipeline orchestration, and smart-home actions. See [ARCHITECTURE.md](ARCHITECTURE.md) for runtime boundaries.

## Requirements

- Home Assistant 2026.8.3 or newer
- A running `llama-server` instance reachable from Home Assistant
- A model and chat template that support reliable OpenAI-style tool calling

SaySo does not bundle or prescribe a specific model. Choose a GGUF model whose chat template works with function calling when launched with `--jinja`.

## Install as a custom integration

### HACS

1. Add this repository as a [custom HACS integration](https://hacs.xyz/docs/faq/custom_repositories/).
2. Install **SaySo** from HACS.
3. Restart Home Assistant.

### Manual

1. Copy the `custom_components/sayso` folder into your Home Assistant `config/custom_components/` directory.
2. Restart Home Assistant.

## Configure llama.cpp

Start `llama-server` separately on a host Home Assistant can reach. Example:

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

Use a model whose template supports tool/function calling. SaySo sends requests to the OpenAI-compatible `/v1/chat/completions` endpoint.

## Configure SaySo in Home Assistant

1. Go to **Settings → Devices & services → Add integration**.
2. Search for **SaySo**.
3. Enter the llama.cpp base URL (for example `http://127.0.0.1:8080/v1`) and optional API key.
4. Select the model identifier exposed by your server.
5. Adjust options (timeout, Home Assistant LLM API, system prompt, temperature, token limits, tool iterations) from the integration’s **Configure** menu.

## Use SaySo as the conversation agent

1. Open **Settings → Voice assistants** (or **Settings → Assist**).
2. Create or edit an assistant.
3. Set **Conversation agent** to **SaySo** (the entry title shows your model and host).
4. Expose only the entities that assistant should control.

With a standard Home Assistant voice pipeline, wake word → STT → SaySo → TTS → satellite playback stays entirely inside Home Assistant except for the HTTP call to llama.cpp.

The optional reference satellite under `satellite/` uses Home Assistant’s standard voice pipeline. Linux Voice Assistant owns capture, volume normalization, WebRTC processing, and HA transport; the SaySo overlay receives the same processed PCM for LiveKit wake detection via LVA’s external wake hook. It does not perform speech recognition, language understanding, model inference, or Home Assistant actions, and it never connects directly to llama.cpp.

## Diagnostics

Download config entry diagnostics from the SaySo integration page. API keys and other configured secrets are redacted automatically.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -q
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for runtime boundaries and
[docs/TRAINING_PLAN.md](docs/TRAINING_PLAN.md) for SaySo model training design.
