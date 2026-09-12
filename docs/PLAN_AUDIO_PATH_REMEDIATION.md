# Plan: Satellite audio-path remediation (native-rate capture → verified STT WAV)

Status: implemented. The numbered TDD units below are all complete and covered
by tests under `satellite/`. Operational instructions live in
`docs/STT_AUDIO_CAPTURE.md`; the benchmark is described in
`satellite/benchmarks/README.md`.

## Outcome

A standard Home Assistant voice pipeline reaches SaySo, Faster Whisper receives
audio that clearly sounds like the spoken command, and the exact post-processing
PCM sent to Home Assistant is retained so defects can be diagnosed from a WAV
instead of by ear over the live satellite.

Milestone 0 (first deliverable): **a WAV of exactly what Faster Whisper receives
that audibly matches the spoken words.** Everything else is sequenced behind it.

## What the current path actually is (verified, not assumed)

Read from upstream LVA at the pinned commit
`99ee08f` (`satellite/pyproject.toml`) plus the SaySo overlay:

- Capture: `mic.recorder(samplerate=16000, channels=n_channels, blocksize=...)`
  in `linux_voice_assistant/__main__.py::process_audio`. The 16 kHz is
  **hardcoded**. soundcard/libpulse therefore resamples the Snowball from its
  native rate implicitly, with no explicit anti-alias filter under our control.
- Volume: `mic_vol_scalar = max(0.1, min(1.0, state.mic_volume / 100.0))`,
  applied as float32 scaling on every block. `state.mic_volume` is never set by
  SaySo's launcher, so it stays LVA's default 100 → scalar `1.0` today. It is a
  hidden, blockwise, state-dependent multiply — not deterministic by
  construction.
- AGC/NS: `WebRTCProcessor` re-instantiates `AudioProcessor(agc, ns)` whenever
  preferences change and buffers a `bytearray` across calls. Re-instantiation
  drops buffered samples (`self._buffer` is not migrated), so a settings change
  injects a discontinuity. `agc`/`ns` come from `state.preferences`, i.e. from
  Home Assistant at runtime, so the effective processing is not fixed.
  `webrtc-noise-gain` exposes **only AGC and NS — no AEC**.
- Wake feed: `external_wake_feed(state, audio_chunk)` is called on the *primary
  channel after mic-volume scaling and after WebRTC* (the call site is below the
  AGC/NS block, so the wake hook sees post-WebRTC PCM). The overlay's
  `SaySoExternalWakeHook.feed_pcm` buffers that into `WakePrerollLookback` and
  feeds `WakeAudioBuffer` for inference.
- Wake→STT handoff: overlay `wakeup()` (patched in `events.py`) calls
  `hook.suspend()`, sets `_pipeline_active`, `_emit(WAKE_WORD_DETECTED)`,
  `duck()`, `_start_audio_streaming(phrase)`, then `hook.flush_preroll(self)`.
  LVA's own `_start_audio_streaming` flips `_is_streaming_audio = True`, and
  `handle_audio` only forwards when `_is_streaming_audio`.
- Playback: LVA plays the wake chime first and only starts streaming in its
  `_on_wakeup_sound_finished` callback *unless* `listen_during_wake_sound` is
  set. The overlay bypasses that and streams **immediately**, so the chime and
  any in-flight TTS are captured as command audio.

### The defects this plan exists to fix

1. **Wake→STT handoff is not sample-ordered.** Detection is computed off-thread
   from a window that may already be one hop stale. When the worker publishes a
   detection, `flush_preroll` drains whatever the lookback happens to hold and
   pushes it through `satellite.handle_audio`, while the capture thread is
   concurrently pushing the same period of audio via `handle_audio`. The two
   producers interleave on the same stream with no ordering or handoff
   boundary. Result: duplicated, reordered, or gapped audio at the STT
   boundary.
2. **Preroll is trimmed relative to flush time, not the detection boundary.**
   `flush_bytes(wake_skip_ms)` skips `wake_skip_ms` from the *end* of the ring
   at the instant of the call. Latency between the true detection sample and
   the flush call (queue hop, thread scheduling, WebRTC buffering) is silently
   included, so the trim is both wrong and variable.
3. **No AEC, and listening reopens before TTS is truly finished.** Wake
   streaming starts before the chime completes; `_tts_finished` gates re-arm
   only for the *wake* path while the wake-up sound and ducked music still leak
   into the first command audio.
4. **Mic volume, AGC, and NS are not deterministic.** `mic_volume` is an
   unset default applied as a float multiply, and `agc`/`ns` are read from HA
   preferences on every block, can change mid-stream, and re-instantiate the
   processor in a way that discards buffered samples.
5. **Implicit resample + no capture record.** The Snowball is resampled
   implicitly to 16 kHz by the audio server, and nothing writes the exact PCM
   handed to HA. There is no artifact that can be listened to.
6. **`test-mic` records at 16 kHz**, so the operator's own sanity check never
   demonstrates the native-rate signal.

## Scope

In scope:

- `satellite/sayso/` overlay code and its tests.
- One new upstream patch file under `satellite/patches/` (native-rate capture
  and explicit resample, if it cannot be done by wrapping).
- `satellite/sayso/config.py` + config validation for a fixed, deterministic
  audio profile (rate, gain, AGC, NS, AEC-gate).
- `satellite/sayso/cli.py` `test-mic` native-rate capture.
- A fixed 20-command benchmark and its runner (offline, using Faster Whisper
  through the existing pipeline shape, not a new test framework).
- Docs: this plan, `satellite/README.md` audio-path section, and a new
  `docs/STT_AUDIO_CAPTURE.md`.

Out of scope (do not touch to save time; do not let them block the voice path):

- Streaming/transport optimizations.
- Far-field mic-array support (only a documented escalation, step 6).
- Changes to the HA integration's tool/schema path.
- Repositioning guidance beyond documenting step 6.
- Expanding `evals/cases/` or training corpora.

## Files to touch

| File | Change |
|---|---|
| `satellite/sayso/wake/capture.py` (new) | Sample-ordered capture ring + explicit resampler + STT PCM tap |
| `satellite/sayso/wake/buffer.py` | `WakePrerollLookback` trims relative to a detection sample index, not flush time |
| `satellite/sayso/wake/hook.py` | Atomic handoff: detection carries the capture sample index; flush is a single ordered splice |
| `satellite/sayso/wake/livekit.py` | Surface the detection's window end sample offset (if not already available) |
| `satellite/sayso/wake/stt_capture.py` (new) | `SttAudioRecorder`: writes exact post-processing PCM per command + one WAV per failure |
| `satellite/sayso/events.py` | Gate wake streaming on TTS/chime completion; wire the STT tap |
| `satellite/sayso/config.py` | `audio.capture_rate`, `mic_gain_db`, `capture_channels`, `stt_capture_dir`, `aec_gate_ms`; validation |
| `satellite/sayso/launcher.py` | Pass the fixed audio profile; install capture tap |
| `satellite/sayso/process_audio.py` | Install native-rate capture wrapper around LVA `process_audio` |
| `satellite/patches/0003-lva-native-rate-capture.patch` (new, only if needed) | Native-rate record + explicit resample hook |
| `satellite/sayso/cli.py` | `test-mic` records native rate, reports RMS/peak, writes labelled WAVs |
| `satellite/sayso/test_capture.py` (new) | TDD unit for capture/resample |
| `satellite/sayso/test_stt_capture.py` (new) | TDD unit for the WAV tap |
| `satellite/sayso/test_voice_cycle.py` | Ordering + gating regression tests |
| `satellite/benchmarks/` (new) | Fixed 20-command benchmark + runner + manifest |
| `docs/STT_AUDIO_CAPTURE.md` (new) | How to capture, where files land, how to listen |

## Design

### 1. Sample-ordered, atomic wake→STT handoff

Introduce a single **capture timeline**: every PCM block produced by the capture
wrapper is appended to one monotonic ring (`WakeCaptureRing`) and stamped with
its absolute sample index (`end_index`). Everything else references that index.

- The wake window carries `window_end_index`.
- A detection carries `detection_index = window_end_index` (the last sample the
  classifier scored).
- The command-audio trim boundary is computed as
  `trim_index = detection_index - skip_samples`, clamped to the ring span.
- The handoff is one operation performed under the ring's lock: capture the
  flush slice `[trim_index, ring_end]`, mark the ring's STT write cursor at
  `ring_end`, and return the slice. Live blocks are then forwarded only for
  samples *after* the cursor, so each sample reaches the STT path exactly once.

This removes duplicate/reordered/gapped audio at the boundary and makes the
trim a function of the detection boundary rather than the flush wall clock.

### 2. Preroll trimmed to the real detection boundary

On the live path this is `WakeCaptureRing.flush_from(trim_index)`, reached via
`SaySoExternalWakeHook.flush_preroll`, which:
- computes `trim_index = detection_index - skip_samples`,
- emits `[max(trim_index, stt_cursor, span_start), ring_end]` under the ring
  lock, clamped to what is held and never below what the live path already
  delivered,
- distinguishes two ways the trim can fall short. Audio the ring held and then
  overwrote is a real `preroll_underflow`, flagged on the sidecar and logged as
  a warning. A trim reaching back past `ring.origin` — the anchor set by the
  most recent rearm — is a cold start, because a wake arriving within
  `wake_skip_ms` of a TTS response asks for audio that never existed. That is
  logged at debug and is not flagged, so the sidecar's `underflow` stays a
  signal rather than noise on every first command.

`WakePrerollLookback.flush_until` in `wake/buffer.py` is the standalone trim
primitive, kept independently tested but not on the live path; it reports
underflow by emitting nothing rather than by clamping.

### 3. AEC or block listening until TTS is truly finished

`webrtc-noise-gain` has no AEC, so we do not pretend to add one. Instead:

- **Gate listening.** In the `wakeup` patch, do not call `_start_audio_streaming`
  until the wake chime's `done_callback` fires, matching LVA's
  `_on_wakeup_sound_finished` semantics. Add an explicit `aec_gate_ms` settle
  delay after the chime, and require `_tts_played == False` before opening the
  mic.
- Keep the pre-open audio in the capture ring (not discarded) so step 2 can
  still flush genuine pre-wake speech.
- Document that a true AEC requires a hardware/reference channel; the LVA
  2-channel path is the only supported route and is out of scope here. The gate
  is the correct fail-safe: it is deterministic and cannot inject speaker
  bleed.

### 4. Deterministic mic volume, AGC, NS

- Add `audio.mic_gain_db` to config; convert once to a fixed linear scalar at
  startup and apply it in one place, with saturation clamping and a clipping
  counter. Do not read LVA's runtime `mic_volume` for the wake/STT path.
- Pin `audio.auto_gain` and `audio.noise_suppression` from config and *force*
  them (do not defer to HA preferences) for the command path. Default
  `noise_suppression = 0` (NS stays off until testing proves it helps) and
  `auto_gain = 0` (no AGC).
- If AGC/NS is enabled, use it in a single fixed configuration for the whole
  process lifetime so `WebRTCProcessor.update_settings` is never called
  mid-stream and no buffered samples are dropped.

### 5. Native 44.1 kHz capture, one deliberate resample to 16 kHz

- `audio.capture_rate` selects the rate requested from the device (default
  44100 for the Snowball). `audio.sample_rate` stays 16000 as the HA transport
  rate.
- Capture at `capture_rate`, run wake inference on a **single explicit
  resample** to 16 kHz, and use that same 16 kHz stream for the STT path so
  wake and STT see identical audio.
- Implementation: a polyphase resampler (numpy; no new dependency) with a
  single continuous filter state across blocks, so block boundaries stay
  continuous. No per-block `np.interp` (that is what `wake/eval.py` does for
  fixtures and is not acceptable for live capture).
- Wrap LVA `process_audio`'s recorder through a new patch only if wrapping at
  the Python level cannot change the hardcoded `samplerate=16000`; prefer
  patching `process_audio` in the overlay so upstream stays mergeable.
- Keep `audio.sample_rate == 16000` validation; add
  `capture_rate in (16000, 44100, 48000)`.

### 6. Real STT audio capture

New `SttAudioRecorder` fed only from the tap immediately before
`satellite.handle_audio`:

- Always writes the exact post-processing int16 PCM sent to HA for the command
  window to `stt_capture_dir/commands/<run_id>.wav` (16 kHz mono, matching what
  HA receives).
- On empty-transcript / pipeline-error events, additionally writes
  `stt_capture_dir/failures/<run_id>_<reason>.wav`.
- Writes a JSON sidecar with: capture rate, resample ratio, gain, AGC/NS,
  detection index, trim index, ring underflow flag, RMS/peak/clip% of the
  command window, and the STT transcript when known.
- Bounded retention by count and age, pruned at write time, off the request
  path; never blocks the audio thread (write on a dedicated writer thread with
  a single-slot queue like `WakeInferenceWorker`).
- Enabled by config; **on by default in this remediation** so the milestone
  artifact exists, with a documented disable flag.

### 7. Milestone 0 acceptance test

A scripted, non-live check that verifies the tap fidelity:

1. Feed a known synthetic utterance (or a recorded WAV) through the capture →
   resample → gain → tap path.
2. Assert the WAV the tap wrote, when decoded, is a
   same-duration, same-rate, non-clipped, monotonic-in-energy rendition of the
   input (correlation above a threshold after rate alignment) with no dropped
   or duplicated samples across block boundaries.
3. Listen: the operator plays the failure/command WAV and confirms it sounds
   like the words spoken. This is the manual half of milestone 0 and is
   recorded in `docs/STT_AUDIO_CAPTURE.md`.

## TDD units (one at a time, in order)

1. **Capture ring + explicit resampler.** Unit: `test_capture.py`
   - 44.1 kHz→16 kHz length ratio and continuity across blocks (no seam at
     block boundaries; chirp/impulse at a boundary survives).
   - Absolute sample indexing is monotonic and exact.
   - Red: write the failing test first.
2. **Sample-ordered, atomic handoff.** Unit: `test_voice_cycle.py`
   - A sample delivered to the STT path appears exactly once, in order, across
     the wake boundary.
   - Trim uses the detection index: with an injected flush delay, the emitted
     window is unchanged (this is the defect-#2 regression).
3. **TTS/chime gating.** Unit: `test_events.py` / `test_voice_cycle.py`
   - No `handle_audio` before chime completion + `aec_gate_ms`.
   - Pre-open audio is still available to the flush path.
4. **Deterministic gain/AGC/NS.** Unit: `test_config.py` + `test_capture.py`
   - Startup computes one gain scalar; preference changes never alter the
     command path; clip counter increments at saturation.
5. **STT audio capture.** Unit: `test_stt_capture.py`
   - Byte-exact equality between tap bytes and the bytes passed to
     `handle_audio`; failure WAV written on empty transcript; sidecar fields
     present; retention prunes by count and age; writer never blocks.
6. **`test-mic` native rate.** Unit: `test_cli.py`
   - Invokes the recorder at `capture_rate`, reports RMS/peak/clip, writes a
     labelled WAV.

## Benchmark: fixed 20 commands

`satellite/benchmarks/commands.json` — 20 fixed phrases, including difficult
entity names: "living room light switch", "bedroom TV", and similar
multi-word/confusable entity names, plus short and long utterances.

`satellite/benchmarks/run.py`:
- Runs **the same Faster Whisper model** over two variants of the same
  recordings:
  - `current`: implicit-resample, current processing.
  - `corrected`: native-rate capture + one deliberate resample to 16 kHz.
- Same fixed gain/AGC/NS for both, per config.
- Reports WER/CER per phrase, per-variant aggregate, and the delta.
- Also reports the measurement set the user asked for: speech RMS/peak per
  phrase, clip%, and a stable gain recommendation.
- **Tune gain, not NS**: the runner's recommended action is a gain value, and
  NS variants are not compared unless a separate experiment explicitly does so.

## Verification steps

For each TDD unit and at the end:

```bash
cd satellite
uv run python -m pytest sayso -q
```

Milestone 0 artifact check:

```bash
sayso-satellite test-mic                 # native-rate capture, RMS/peak report
sayso-satellite start                    # speak a listed benchmark command
ls /var/lib/sayso-satellite/stt_capture/commands/*.wav
```

Then play the newest command WAV and confirm it audibly matches the spoken
words. Record the result in `docs/STT_AUDIO_CAPTURE.md`.

Full offline benchmark:

```bash
python3 satellite/benchmarks/run.py --model <faster-whisper-model> \
  --commands satellite/benchmarks/commands.json
```

Also re-run the existing suites that must not regress:

```bash
python3 -m pytest satellite/sayso/wake/test_buffer.py satellite/sayso/wake/test_eval.py -q
python3 -m pytest tests -q
```

## Step 6 (only if articulation errors remain)

Documented escalation, no code in this plan:

- Reposition/orient the Snowball (off-axis, away from speaker and walls).
- Benchmark closer vs normal speaking distance using the same 20 commands.
- If distance remains limiting, move to a far-field mic array; that changes
  capture ownership and requires an `ARCHITECTURE.md` update at that time.

`ARCHITECTURE.md` is updated only if step 6 changes ownership. Steps 1–5 keep
LVA as the capture owner and SaySo as a thin overlay, so no invariant changes.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Native-rate capture needs an upstream change | Prefer overlay wrapping of `process_audio`; fall back to a third minimal patch file |
| Resampler seams | Single continuous filter state; impulse-at-boundary test in unit 1 |
| Blocking the audio thread with disk writes | Single-slot writer queue, drop-oldest, same pattern as `WakeInferenceWorker` |
| Over-trimming the wake word | Keep `wake_skip_ms`, clamp `start` to ring span, log underflow explicitly |
| Gating introduces perceived latency | Gate only on chime completion; `aec_gate_ms` small and measured in traces |
| Removing AGC makes quiet input worse | Gain is set from measured speech RMS in the benchmark, not guessed |

## Open questions for the user

1. Is the Snowball actually the configured `input_device`, and does it expose
   44.1 kHz mono through PipeWire/Pulse, or only 48 kHz?
2. Should `stt_capture_dir` default under `/var/lib/sayso-satellite/stt_capture`
   (state dir) or a user-visible path for easy listening?
3. Confirm NS stays disabled (`noise_suppression = 0`) as the starting default.

## Implementation status

| Item | Where | Tests |
|---|---|---|
| 1. Sample-ordered, atomic wake→STT handoff | `wake/capture.py` (`WakeCaptureRing`), `wake/hook.py` | `wake/test_handoff.py`, `wake/test_capture.py` |
| 2. Preroll trimmed at the detection boundary | `wake/buffer.py` (`flush_until`), `wake/hook.py` | `wake/test_preroll.py`, `wake/test_buffer.py` |
| 3. Block listening until playback is truly done | `events.py` (`_defer_until_playback_idle`, `aec_gate_ms`) | `test_wake_gating.py` |
| 4. Deterministic gain/AGC/NS | `process_audio.py` (`_ResamplingRecorder`, `_pin_audio_settings`), `config.py` | `test_process_audio.py`, `test_config.py` |
| 5. Native 44.1 kHz capture, one deliberate resample | `process_audio.py`, `wake/capture.py` (`CaptureResampler`) | `wake/test_capture.py`, `test_process_audio.py` |
| 6. Retain the exact STT PCM + failure WAVs | `wake/stt_capture.py`, `events.py` (`handle_audio` tap) | `wake/test_stt_capture.py`, `test_stt_tap.py` |
| 20-command benchmark | `benchmarks/run.py`, `benchmarks/commands.json` | `benchmarks/test_benchmark_run.py` |
| `test-mic` native-rate capture | `cli.py` | `test_cli.py` |

Resolved during implementation beyond the original plan:

- `Detection` now carries the absolute `sample_index` so the trim boundary is a
  property of the audio, not of thread scheduling.
- The wake worker forwards that index through its queue
  (`wake/worker.py`), with signature-based fallback for providers that do not
  accept it.
- The capture ring re-anchors on rearm instead of zeroing, so a stale detection
  index can never be mistaken for a fresh one.
- A capture safety net writes a failure WAV when a turn ends without `stt_end`.
