# Plan: Switch the living-room satellite to NanoWakeWord

Status: done. Explicit operator request to load Nano live. Do not
change Home Assistant, llama.cpp, or architecture. Do not put passwords
in the repo. Do not train.

## Outcome

On satellite `192.168.1.54`, load scale Nano `0a3c0d64` as the live wake
engine. LiveKit ONNX stays on disk. Repo default remains LiveKit; only this
Pi’s `/etc/sayso-satellite/config.yaml` flips.

## Scope

1. Pi (ops):
   - Confirm `sayso-nanowakeword.onnx` md5 `0a3c0d645c82adbb8c1d33f39cf81017`
   - Confirm `nanowakeword` importable in `/opt/sayso-satellite/.venv`
   - Backup `/etc/sayso-satellite/config.yaml` to
     `config.yaml.bak-livekit-0a3c0d64`
   - Set `wake_word.provider: nanowakeword` and `wake_word.model` to
     `/opt/sayso-satellite/models/sayso-nanowakeword.onnx`
   - Leave threshold `0.50`, phrase `SaySo`, LiveKit `sayso.onnx` untouched
   - Restart `sayso-satellite`; confirm logs load NanoInterpreter and the
     service is active
2. Repo: `docs/HANDOFF_WAKE.md` live-system block only (provider now
   nanowakeword). Do not change ARCHITECTURE.md (Nano stays an optional
   overlay engine). Do not change example defaults in satellite README
   beyond a one-line “living-room Pi is on Nano” if HANDOFF already covers it.

## Verification

- `grep provider/model` shows nanowakeword + sayso-nanowakeword.onnx
- `md5sum` Nano ONNX still `0a3c0d64…`; LiveKit `sayso.onnx` still `03e612d8…`
- `systemctl is-active sayso-satellite` is active
- Journal contains `Loaded NanoWakeWord model` (or equivalent) and no
  fail-closed missing-model error
- Rollback: restore `config.yaml.bak-livekit-0a3c0d64` and restart
