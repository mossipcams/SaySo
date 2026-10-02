# STT VAD input warm-up

## Status

The 1.2-second silent primer is implemented and remains the VAD warm-up. The
1.2-second wall-clock guard on top of it was removed
(`docs/PLAN_SATELLITE_LATENCY_BARGE_IN.md`, Phase 1): the buffered wake-ring
audio is now flushed immediately after the primer is sent, with the token-based
cancellation guard and underflow handling preserved. The original wall-clock
hold was a conservative duplicate of the primer, which already advances HA's
VAD by the observed ~1.05 s of processed audio. If a live wake→STT A/B shows
the command onset clipped or missing, HA is dropping pre-VAD-ready audio and
the wall-clock hold must be restored.

## Scope (original plan, historical)

The current workaround sends only 64 ms of silence, then waits 1.2 seconds
without sending audio. The next capture still shows the VAD opening around
1.05 seconds into the audio stream. Send 1.2 seconds of silent PCM up front so
Home Assistant's 10 ms VAD processing advances before the buffered microphone
audio is flushed. Keep the existing 1.2 second wall-clock guard and audio
ordering.

## Files

- `satellite/sayso/events.py` — make the silent primer 1.2 seconds; retain the
  delayed wake-ring flush and cancellation behavior.
- `satellite/sayso/test_events.py` and `satellite/sayso/test_wake_gating.py` —
  expect the full silence pre-roll before buffered command audio.
- `docs/STT_AUDIO_CAPTURE.md` — document the silence pre-roll in captures.

## Verification

- Run `python -m py_compile satellite/sayso/events.py` and Ruff checks.
- Review the diff and confirm the microphone ring remains buffered until the
  silent input pre-roll has been sent and the warm-up timer completes.
