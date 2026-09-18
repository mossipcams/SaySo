# Plan: NanoWakeWord prototype for the SaySo satellite

Status: implemented (prototype). Do not replace the production LiveKit
operating point or the existing LVA external-wake path.

## Outcome

The reference satellite can detect the spoken phrase **SaySo** with a
NanoWakeWord ONNX model on the same processed 16 kHz PCM feed LVA already
sends to the overlay. A checked-in training config plus a documented command
can synthesize and train a prototype "SaySo" model. LiveKit remains the
default production provider.

## Why this is a prototype

The shipped satellite uses `livekit-wakeword` (`satellite/sayso/wake/livekit.py`)
on `/opt/sayso-satellite/models/sayso.onnx`. That path has a calibrated
threshold and known false-positive behavior (`satellite/models/README.md`).
NanoWakeWord (`arcosoph/nanowakeword`, `NanoInterpreter`) is a different
feature frontend and score distribution. A first model is not a drop-in
replacement for the living-room operating point.

Full NanoWakeWord training (thousands of Piper clips, 20k steps) is host-side
work, same rule as LiveKit: do not run it on the Pi.

## Architecture constraints (do not violate)

- Overlay still registers LVA's **external wake hook**. No second capture path.
- Satellite still does not do STT, TTS, NLU, llama.cpp, or HA actions.
- No NanoWakeWord **server**, remote verifier, API keys, or cloud path.
- Wake audio is not retained except the existing optional `mine_dir` spool.
- Must not start another request while the voice pipeline is active (existing
  suspend / `_pipeline_active` gating stays).
- Update `ARCHITECTURE.md` only to name NanoWakeWord as an optional overlay
  engine on the same hook. Ownership, transport, and invariants do not change.

## Scope

### Runtime (satellite overlay)

Add a `NanoWakeWordProvider` that implements `WakeWordProvider`
(`satellite/sayso/wake/provider.py`):

- Load via `NanoInterpreter.load_model(path)`.
- Consume int16 mono PCM at 16 kHz. Do not open the microphone.
- Expose `predict_window(window, sample_index=)` like LiveKit so
  `SaySoExternalWakeHook` / `WakeInferenceWorker` stay unchanged aside from
  typing the hook as the protocol instead of `LiveKitWakeWordProvider`.
- Map `result.score` against `wake_word.threshold`; return `Detection` with
  phrase `SaySo`. Call `interpreter.reset()` on `reset` / after fire as
  NanoWakeWord recommends.
- Fail closed if the model file or import is missing (`available` is False;
  launcher still exits).
- Keep LiveKit hop/window constants in the hook unless NanoInterpreter
  **requires** a different chunk size; if it does, keep hop/window on the
  provider and have the hook read them from the provider. Do not invent a
  second PCM pipeline.

Config:

- Allow `wake_word.provider` of `livekit` (default) or `nanowakeword`.
- Phrase remains exactly `SaySo`.
- Default install / example config stays LiveKit. NanoWakeWord is opt-in.

Launcher / CLI:

- Construct `NanoWakeWordProvider` when provider is `nanowakeword`.
- `test-wake-word` must work for both providers.

Runtime dependency: inference-only `nanowakeword`. Do **not** add
`nanowakeword[train]` to satellite runtime deps.

### Training (prototype-scale)

Add a NanoWakeWord YAML next to the LiveKit retrain config, e.g.
`satellite/models/sayso-nanowakeword.yaml`:

- `target_phrase` / positives: **SaySo** only (`fixed_phrase: "SaySo"`).
  `"say so"` and the rest of the `/seI soU/` list are hard negatives only.
- Hard negatives: reuse the existing `/seI soU/` confusable set from
  `sayso-training.yaml` (`say so`, `says so`, `said so`, …).
- Architecture: embedding-mode **DNN** (fastest prototype).
- Sample counts small enough for a host smoke run (tens to a few hundred
  clips, short `steps`). Label the file as prototype, not production FAR.
- `generate_clips` / `transform_clips` / `train_model` stages documented.
- Gitignore generated `data/`, `output/`, Piper artifacts, and large ONNX
  dumps. Do not commit tens of MB of wavs.

Document the command in `satellite/models/README.md`:

```text
pip install "nanowakeword[train]"
nanowakeword -c satellite/models/sayso-nanowakeword.yaml -G -t -T
```

If ffmpeg/Piper and time allow in this session, run a **tiny** smoke train
and keep a small exported ONNX only if it is small and clearly marked
prototype. If the environment cannot train, leave the config + command;
do not fake a production model or copy the LiveKit ONNX and call it
NanoWakeWord.

Do not train on `evals/cases/` utterances or ChatML labels.

### Tests

Colocated pytest, existing suite only:

- `satellite/sayso/wake/test_nanowakeword.py`: mock `NanoInterpreter`;
  threshold fire / no-fire, reset, missing model fail-closed, no mic open.
- `satellite/sayso/test_config.py`: accept `provider: nanowakeword`; reject
  unknown providers; keep LiveKit fixtures green.
- `satellite/sayso/test_launcher.py`: nanowakeword branch constructs the new
  provider; livekit branch unchanged.

Do not require a real trained ONNX for unit tests.

## Files to touch

- `docs/PLAN_NANOWAKEWORD.md` (this file)
- `ARCHITECTURE.md` (one overlay-engine sentence, if needed)
- `satellite/pyproject.toml`
- `satellite/models/sayso-nanowakeword.yaml` (new)
- `satellite/models/README.md`
- `satellite/README.md` (short opt-in note)
- `satellite/.gitignore` or repo gitignore for generated train artifacts
- `satellite/sayso/config.py`
- `satellite/sayso/launcher.py`
- `satellite/sayso/cli.py` (if `test-wake-word` is LiveKit-hardcoded)
- `satellite/sayso/test_config.py`
- `satellite/sayso/test_launcher.py`
- `satellite/sayso/wake/__init__.py`
- `satellite/sayso/wake/hook.py` (protocol type only, unless hop must move)
- `satellite/sayso/wake/nanowakeword.py` (new)
- `satellite/sayso/wake/test_nanowakeword.py` (new)

Out of scope: HA conversation agent, llama.cpp, evals LLM cases, LiveKit
retrain, NanoWakeWord server, replacing the default LiveKit model, Pi
training.

## Verification

1. `python -m pytest satellite/sayso/wake/test_nanowakeword.py satellite/sayso/test_config.py satellite/sayso/test_launcher.py satellite/sayso/wake/test_livekit.py satellite/sayso/wake/test_handoff.py satellite/sayso/test_wake_gating.py`
2. Existing satellite tests that import launcher/config still pass.
3. Config with `provider: nanowakeword` and a missing model fails closed.
4. Training: YAML parses; if a smoke train runs, it writes an ONNX under the
   configured output dir. If skipped, document why in the plan status /
   delegate report.

## Done when

- Opt-in `nanowakeword` provider runs on the existing external wake hook.
- LiveKit default path is unchanged.
- Prototype "SaySo" training config exists and is documented.
- Tests cover the new provider without a real model file.
