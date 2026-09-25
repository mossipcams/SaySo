# STT VAD warm-up workaround

## Scope

Send a short silent PCM primer to Home Assistant when the satellite starts the
STT stream, wait 1.2 seconds for HA's external VAD initialization, then flush
the already-buffered microphone audio in order. This is a temporary workaround
for the observed ~1.05 second VAD start delay.

## Files

- `satellite/sayso/events.py` — insert the primer and defer the existing wake
  ring flush; cancel a pending flush when a pipeline ends early.
- `docs/STT_AUDIO_CAPTURE.md` — document that saved captures include the primer.

## Verification

- Run `python -m py_compile satellite/sayso/events.py`.
- Review the diff and confirm the VAD handoff remains buffered until warm-up
  completes.
