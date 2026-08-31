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
