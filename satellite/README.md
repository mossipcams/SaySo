# SaySo reference satellite

Optional SaySo reference satellite for OHF-Voice/linux-voice-assistant. It uses
Home Assistant’s standard voice pipeline. The SaySo Home Assistant integration
does not manage or require this bundled satellite; any compatible Home Assistant
voice satellite works.

## Audio path

Linux Voice Assistant (LVA) owns microphone capture, volume normalization,
WebRTC audio processing, Home Assistant transport, and speaker playback. After
each processed audio block, LVA forwards the same PCM to registered external
wake providers. The SaySo overlay registers an external wake hook for LiveKit
wake-word detection on that feed. It does not wrap LVA's `record()` path or
create a second capture path.

The overlay owns one deliberate resample and the capture timeline:

- Upstream LVA hardcodes `samplerate=16000`, which makes the audio server
  resample the device implicitly. The overlay wraps `process_audio`, opens the
  recorder at `audio.capture_rate`, and resamples **once** to 16 kHz through a
  continuous polyphase resampler (`satellite/sayso/wake/capture.py`) when
  `capture_rate` ≠ `sample_rate`. The living-room Pi uses an **EMEET OfficeCore
  M0 Plus** at **16 kHz native** (`capture_rate: 16000`, `mic_gain_db: 0.0`) so
  transport matches the device without a 16→48→16 chain. Snowball-class mics at
  44.1/48 kHz remain supported the same way. No per-block interpolation.
- One `WakeCaptureRing` is the single source of truth for sample order. Wake
  inference reads a window from it, the detection carries the window's end
  sample index, and the wake→STT handoff drains
  `[detection_index - wake_skip_ms, end)` atomically. Each sample reaches Home
  Assistant exactly once, in order, regardless of when the detection thread
  runs. See `satellite/sayso/wake/hook.py`. With `--disable-built-in-wake-word`,
  live STT is produced only by the overlay ring (`flush_preroll` and
  `_forward_live`); LVA does not send a second copy via `handle_audio`.
- `wake_word.wake_skip_ms` is a **lookback**, not a skip: it starts the handoff
  that far *before* the detection boundary so a command spoken straight through
  the wake word ("SaySo turn on the TV") keeps its onset. It is bounded on both
  sides: below by the detection lag (0-240 ms, median 80, measured by re-scoring
  `wake_word.mine_dir` clips under a growing tail cut), above by Home
  Assistant's VAD, which opens on any prepended wake-word audio and then times
  out during the speaker's pause, closing STT before the command arrives.
  **The two bounds do not overlap**, so the 120 ms default only picks the
  recoverable failure (a clipped onset) over the total one (a dropped command).
  Trimming to the real phrase end is the actual fix — see issue #49.
- The microphone does not open until any in-flight playback has genuinely
  finished, plus an optional `audio.aec_gate_ms` delay. That delay only
  postpones when the mic opens; it does not remove samples from the STT
  payload (`flush_preroll` still emits `[trim, now]`). Default is 0. There is
  no AEC on this path (`webrtc-noise-gain` exposes AGC/NS only).
- Gain is a single fixed `audio.mic_gain_db` multiply applied once, not a
  runtime lookup. AGC and NS are pinned from config; the default is NS off.
- The exact PCM sent to Home Assistant is retained. See
  `docs/STT_AUDIO_CAPTURE.md` for the milestone-0 listening check.

Upstream LVA lives in `linux-voice-assistant/` and must stay mergeable. Custom
code is only under `sayso/` plus patches:

- `patches/0001-sayso-stable-device-name.patch` (`--device-name` for stable HA device id)
- `patches/0002-lva-external-wake-provider.patch` (processed PCM external wake hook + `--disable-built-in-wake-word`)

CI on this repo applies the patches to the pinned LVA tree; a running satellite still needs its `/opt/sayso-satellite` venv rebuilt from that patched tree.

`satellite.name` in `config.yaml` is the friendly display name passed to LVA `--name`.
`satellite.device_name` is the stable Home Assistant device id passed to LVA `--device-name`
(for example `sayso-living-room`).

## Wake model

Copy `models/sayso.onnx` to `/opt/sayso-satellite/models/` before start.
Production on the Pi is **`livekit-corpus-hn-v1`** (`0a3260c8`) at threshold
**0.42**, single-stage LiveKit (no mel verifier). See `models/README.md`.
Historical living2 + `sayso-verifier.npz` notes remain in that file.

### Wake mining (opt-in)

Set `wake_word.mine_dir` to retain bounded scored 2 s windows for offline
labelling. Mining is off by default and must not affect wake detection or the
Home Assistant voice path. Each published record is an atomically renamed
directory under `records/<capture_id>/` containing `window.wav`, optional
`pre.wav` / `post.wav` ring context, and `record.json` with hashes, absolute
sample range, session id, provider, model SHA-256, score/thresholds, processing
settings, sampling reason, and quality flags. Labels stay null at capture;
`fired` records threshold crossing only, not accepted wake or user intent.

Wake acceptance/suppression and later HA/STT outcomes are separate files under
`outcomes/`, linked by `capture_id` (distinct from Home Assistant's pipeline
trace id and from STT `run_id`). Transfer is offline: run
`python scripts/wake_mine_report.py <mine_dir> --ingest` on the host to verify
hashes and write `acks/`; the satellite deletes only acknowledged records via
`drain_acks()` and resumes collection after drain without restart.

## Service

The satellite runs as a **system** service under a dedicated `sayso` account,
not as a `systemd --user` unit. A user unit needs a per-user service manager
kept alive, which on DietPi means enabling lingering and pulling in
systemd-logind and the D-Bus session machinery. A system service with `User=`
avoids all of that.

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin \
    --groups audio sayso
sudo install -m 0644 satellite/systemd/sayso-satellite.service \
    /etc/systemd/system/sayso-satellite.service
sudo systemctl daemon-reload
sudo systemctl enable --now sayso-satellite
```

`/etc/sayso-satellite/config.yaml` and `secrets.yaml` must be readable by
`sayso`; keep `secrets.yaml` at `0640` owned by `root:sayso`.

State lives in `/var/lib/sayso-satellite`, created by systemd through
`StateDirectory=` and owned by the service user. The unit does not use `%h`:
in a system unit that specifier resolves to root's home regardless of `User=`.

### Audio without a user session

A system service has no `/run/user/<uid>`, so the audio server must be reachable
system-wide. The unit defaults to the conventional PulseAudio system-mode
socket:

```ini
Environment=PULSE_SERVER=unix:/run/pulse/native
```

Override it without editing the unit:

```bash
echo 'PULSE_SERVER=unix:/run/pipewire-0' | sudo tee /etc/default/sayso-satellite
```

If PulseAudio runs in system mode with access control, add the service user to
its group as well: `sudo usermod -aG pulse-access sayso`.

`ExecStartPre` runs `sayso-satellite wait-audio`, which polls `pactl` until the
configured source and sink appear, so startup fails fast and loudly when the
audio server is not reachable.

## Commands

Operational commands (also `/usr/local/bin/sayso-satellite`):

```text
sayso-satellite start|stop|restart|status
sayso-satellite logs
sayso-satellite validate
sayso-satellite devices
sayso-satellite test-mic
sayso-satellite test-speaker
sayso-satellite test-wake-word   # recorded-audio eval (see satellite/eval/README.md)
python3 satellite/eval/run.py --model /path/to/sayso.onnx
python3 satellite/benchmarks/run.py --model large-v3  # 20-command STT benchmark
```
