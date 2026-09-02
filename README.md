# SaySo

Local, Alexa-style smart-home voice assistant. Python monorepo skeleton for
`sayso-server` and `sayso-satellite`.

## Setup

```bash
uv sync
```

## Check

```bash
uv run pytest -q
python -c 'import sayso_server, sayso_satellite'
```

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Evaluation plan](docs/EVALUATION_PLAN.md)
- [WebSocket conversation plan](docs/WEBSOCKET_CONVERSATION_PLAN.md)
- [Tuning plan](docs/TUNING_PLAN.md)
