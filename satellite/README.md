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
wake-word detection on that feed. It does not wrap LVA’s `record()` path or
create a second capture path.

Upstream LVA lives in `linux-voice-assistant/` and must stay mergeable. Custom
code is only under `sayso/` plus patches:

- `patches/0001-sayso-stable-device-name.patch` (`--device-name` for stable HA device id)
- `patches/0002-lva-external-wake-provider.patch` (processed PCM external wake hook + `--disable-built-in-wake-word`)

`satellite.name` in `config.yaml` is the friendly display name passed to LVA `--name`.
`satellite.device_name` is the stable Home Assistant device id passed to LVA `--device-name`
(for example `sayso-living-room`).

## Wake model

Copy `models/sayso.onnx` to `/opt/sayso-satellite/models/sayso.onnx` before start.
That classifier detects the spoken phrase **Sayso** only. See `models/README.md`
and `models/sayso_eval.json` for the operating point (threshold 0.19).

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
```
