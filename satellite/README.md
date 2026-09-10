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
and `models/sayso_eval.json` for the operating point (threshold 0.5).

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
```
