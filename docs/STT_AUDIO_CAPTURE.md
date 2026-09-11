# STT audio capture

Status: operational. This is milestone 0 of
`docs/PLAN_AUDIO_PATH_REMEDIATION.md`.

## Why

Before tuning gain, moving the microphone, or blaming the model, there must be
an artifact that can be listened to. This records the exact post-processing PCM
handed to Home Assistant, at the single point where it leaves the satellite, so a
captured WAV is byte-identical to what Faster Whisper received. It is not a
re-recording and not a re-derivation.

## Where the tap is

`satellite` -> LVA capture -> **native-rate capture + one resample** (see
`satellite/sayso/process_audio.py`) -> WebRTC AGC/NS (fixed config) -> wake hook
-> `satellite.handle_audio` -> Home Assistant.

The tap wraps `handle_audio` in `satellite/sayso/events.py`. Recording happens
immediately before the send, so nothing between the tap and Home Assistant can
change the bytes.

## What is written

Under `stt_capture_dir` (default `/var/lib/sayso-satellite/stt_capture`):

```text
commands/<run-id>.wav      one per command, 16 kHz mono int16
commands/<run-id>.json     sidecar
failures/<run-id>_<reason>.wav   one per failed command
failures/<run-id>_<reason>.json  sidecar
```

Failure reasons: `empty_transcript`, `pipeline_error`.

The sidecar records the processing chain so a listening session can be tied
back to the exact settings that produced it: `capture_rate`, `sample_rate`,
`mic_gain_db`, `noise_suppression`, `auto_gain`, sample count, duration, peak,
RMS (linear and dBFS), clip count, transcript, and the preroll `underflow` flag.

## Configuration

```yaml
audio:
  sample_rate: 16000        # transport rate to Home Assistant (fixed)
  capture_rate: 44100       # native device rate; one deliberate resample
  mic_gain_db: 6.0          # fixed gain, applied once
  noise_suppression: 0      # stay off until a test proves it helps
  auto_gain: 0              # no AGC
  aec_gate_ms: 150          # settle delay before the mic opens
  stt_capture_enabled: true
  stt_capture_dir: /var/lib/sayso-satellite/stt_capture
```

Capture is **on by default** so the milestone artifact exists. Writes happen on a
dedicated thread with a bounded queue; a slow disk drops the oldest pending
capture rather than stalling the audio thread. Retention is bounded by count
(500 commands) and age (14 days), pruned at write time.

## Milestone 0 acceptance: listen to the WAV

1. Check the input path and level:

   ```bash
   sayso-satellite test-mic        # records at capture_rate, reports RMS/peak/clip
   ```

2. Start the satellite and speak a command from
   `satellite/benchmarks/commands.json`:

   ```bash
   sayso-satellite start
   ```

3. Find the newest command capture and play it:

   ```bash
   ls -t /var/lib/sayso-satellite/stt_capture/commands/*.wav | head -1
   ```

4. **Listen.** The WAV must clearly sound like the words spoken.

If it does, the audio path is not the articulation problem and the next lever is
the benchmark's gain recommendation. If it does not, the sidecar's `peak`,
`rms_dbfs`, and `clip_count` say whether the cause is level, and the
`underflow` flag says whether the preroll trim was wrong.

## Confirming the file equals what Home Assistant got

The tap is covered by byte-equality tests
(`satellite/sayso/test_stt_tap.py`): the WAV contents are compared directly
against the bytes passed to `handle_audio`. There is no second code path that
could produce a different file.

## Escalation

If the captured WAV sounds correct but transcription still fails on difficult
entity names, the remaining cause is reverberation or distance, not the audio
path. Steps, in order:

1. Reposition and reorient the microphone (off-axis, away from walls and the
   speaker).
2. Re-run the 20-command benchmark closer vs at normal speaking distance.
3. If distance remains the limit, move to a far-field microphone array. That
   changes capture ownership and requires an `ARCHITECTURE.md` update at that
   time.
