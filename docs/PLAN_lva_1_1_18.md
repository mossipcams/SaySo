# Plan: move the satellite to LVA v1.1.18

## Scope

Bump `linux-voice-assistant` from `99ee08f7b136f5a57a3da59d5f7c73e7ed62ed68` to
v1.1.18 (`b55d36af522c263eeed4e023d6ebf30da47e4710`) and fix everything that
breaks. Adopt the zero-config upstream behaviours the overlay would otherwise
miss because it replaces those methods.

Out of scope: deploying to the Pi, committing, pushing, and retuning
`post_tts_cooldown_ms` (needs the live device).

## Changes

1. **`_tts_finished` kwarg** (`satellite/sayso/events.py`). Upstream calls
   `_tts_finished(pipeline_ended=...)` on `RUN_END`. Forward it with
   `*args, **kwargs`; `test_tracing.py` has a fake that takes no kwargs, so an
   explicit `pipeline_ended=` forward would break it.
2. **Disable the pipeline-ended sound** (`satellite/sayso/launcher.py`). Pass
   `--pipeline-ended-sound ""`. It replaces the in-flight failure chime and loses
   that chime's re-arm callback.
3. **Output device name** (`satellite/sayso/launcher.py`). v1.1.18 exits at startup
   unless mpv knows the device, and mpv lists `pulse/<sink>`. Pass
   `pulse/<sink>` for both bare and already-prefixed config values, reusing the
   existing `removeprefix("pulse/")` logic from `cli.py` where practical.
4. **Mic reconnect** (`satellite/sayso/process_audio.py`). Upstream's
   `_ReconnectingRecorder._refresh_mic` replaces `self._mic` with a fresh object,
   losing the instance-level `mic.recorder` wrapper. Recorders opened after a
   refresh must still be `_ResamplingRecorder`, keeping gain, `raw_sink` and clip
   tally. The wrapper must stay tolerant of an `lva_main` without
   `_ReconnectingRecorder`, because existing tests use fakes.
5. **Patches** (`satellite/patches/`). Regenerate 0001 and 0002 against v1.1.18 so
   plain sequential `git apply` works on a shallow checkout, as in
   `.github/workflows/ci.yml`.
6. **`events.py` shim** (`satellite/patches/0002-lva-external-wake-provider.patch`).
   Add `linux_voice_assistant/events.py` re-exporting `LVAEvent` from
   `peripheral_api`, with no comments or docstring. The live Pi carries this as an
   untracked file; a repo rebuild crashes without it.
7. **Pin** (`satellite/pyproject.toml`, `satellite/uv.lock`). Only the
   `linux-voice-assistant` entry may change in the lock.
8. **README** (`satellite/README.md`). Update the patch 0002 description.

## Adopted features

- **Wake phrase on follow-ups** (`satellite/sayso/events.py`): the overlay's
  `wakeup` sets `self._wake_word_phrase = wake_word_phrase`, as upstream's does.
- **No failure chime for a lost wake race** (`satellite/sayso/events.py`): skip
  `_fail_turn` when the error `code` is `duplicate_wake_up_detected`. The trace and
  capture for that run must still finish cleanly; the wake hook must still re-arm.

## Not adopted

- `--mac-address`: needs new config surface and solves no current problem.
- Removing `install_playback_recovery`: still correct, and removing it means
  deleting existing test assertions.
- Pipeline-ended sound: conflicts with SaySo's chimes.

## Rules

- No comments or docstrings in `.py` files, including the shim.
- Do not change or delete existing tests or assertions. New test functions are
  allowed in a new `satellite/sayso/test_lva_compat.py`.
- If an existing test blocks a change, stop and report it.
- No commits, pushes, merges, rebases or branch switches.

## Verification

- New tests fail without their fix: `_tts_finished` kwarg through an
  upstream-shaped fake, launcher argv for both output device forms and the empty
  sound, recorder wrapping after `_refresh_mic`, wake phrase set, duplicate error
  plays no failure chime but re-arms.
- With `SAYSO_LVA_SRC` set, a test asserts the patched tree has
  `linux_voice_assistant/events.py`.
- Fresh `git clone --depth 1` of v1.1.18, then `git apply --verbose` of 0001 then 0002,
  exactly as CI does, succeeds.
- `SAYSO_LVA_SRC=<that tree> pytest -q satellite/sayso` passes.
- `ruff check .` and `ruff format --check .` pass.
- Scratch check outside the repo: the real `VoiceSatelliteProtocol` with SaySo handlers
  over a real asyncio socket, with only `mpv`, `webrtc_noise_gain`, `pymicro_wakeword`,
  `pyopen_wakeword`, `netifaces` and `soundcard` stubbed. A no-TTS `RUN_END` keeps the
  connection up and calls `rearm` once.
