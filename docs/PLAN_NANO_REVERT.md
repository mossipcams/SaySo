# Plan: Remove NanoWakeWord from the satellite

Status: shipped. NanoWakeWord provider, model, and eval harness removed.
Production wake stays LiveKit living2 + mel verifier.

Follow-up: delete `test_load_config_rejects_nanowakeword_provider` (and the
matching launcher reject test if it is in the same leftover set). Keep
verifier config/launcher tests. Do not commit.

## Why

NanoWakeWord false-woke live in this room. The user asked to revert all nano
code. LiveKit living2 (`b840f51f`) and the mel verifier stay.

## Scope

Delete the optional Nano provider, its model/recipe, its LiveKit-vs-Nano
eval harness, and the runtime dependency. LiveKit remains the only
`wake_word.provider`.

Keep:

- `satellite/sayso/wake/livekit.py` and `verifier.py`
- `satellite/models/sayso.onnx`, `living2.yaml`, `sayso-verifier.npz`
- Host directory names in comments (`sayso-nanowakeword/data/…`) — those are
  where the Snowball wavs live, not runtime code

## Files

Delete:

- `satellite/sayso/wake/nanowakeword.py`
- `satellite/sayso/wake/test_nanowakeword.py`
- `satellite/models/sayso-nanowakeword.yaml`
- `satellite/models/sayso-nanowakeword.onnx`
- `satellite/eval/compare_providers.py`
- `satellite/eval/test_compare_providers.py`
- `docs/PLAN_NANOWAKEWORD.md`

Edit:

- `satellite/sayso/wake/__init__.py` — drop `NanoWakeWordProvider`
- `satellite/sayso/config.py` — `provider` is `livekit` only
- `satellite/sayso/launcher.py` — LiveKit branch only
- `satellite/sayso/cli.py` — drop nano test-wake path
- `satellite/sayso/test_config.py` / `test_launcher.py` — drop nano cases;
  reject `nanowakeword` as unknown
- `satellite/pyproject.toml` + `satellite/uv.lock` — drop `nanowakeword`
- `satellite/README.md`, `satellite/models/README.md`,
  `docs/HANDOFF_WAKE.md`, `docs/PLAN_LIVEKIT_VERIFIER.md` — strip provider
  and compare-providers text
- `ARCHITECTURE.md` — overlay is LiveKit only (ownership sentence)

Do not commit wavs or `context.json`. Do not change Pi config in this
step. Do not commit unless asked.

## Verification

```
PYTHONPATH=satellite pytest -q \
  satellite/sayso/wake/test_livekit.py \
  satellite/sayso/wake/test_verifier.py \
  satellite/sayso/test_config.py \
  satellite/sayso/test_launcher.py \
  satellite/sayso/wake/test_handoff.py \
  satellite/sayso/test_wake_gating.py
```

Grep the tree: no `nanowakeword`, `NanoWakeWord`, `NanoInterpreter`, or
`sayso-nanowakeword` except host data-path comments in `living2.yaml` /
handoff data tables.
