from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from .config import CONFIG_PATH, load_config, validate_config

USER_UNIT = "sayso-satellite.service"
PULSE_SOURCE = "alsa_input.usb-BLUE_MICROPHONE_Blue_Snowball_201301-00.mono-fallback"
PULSE_SINK = "alsa_output.platform-fe00b840.mailbox.stereo-fallback"
CHECK_DIR = Path("/var/tmp/sayso-satellite")


def _systemctl(*args: str) -> int:
    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return subprocess.call(["systemctl", "--user", *args], env=env)


def cmd_start(_: argparse.Namespace) -> int:
    return _systemctl("start", USER_UNIT)


def cmd_stop(_: argparse.Namespace) -> int:
    return _systemctl("stop", USER_UNIT)


def cmd_restart(_: argparse.Namespace) -> int:
    return _systemctl("restart", USER_UNIT)


def cmd_status(_: argparse.Namespace) -> int:
    return _systemctl("status", USER_UNIT)


def cmd_logs(_: argparse.Namespace) -> int:
    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return subprocess.call(["journalctl", "--user", "-u", USER_UNIT, "-f"], env=env)


def cmd_validate(_: argparse.Namespace) -> int:
    cfg = load_config()
    validate_config(cfg, check_port_bind=True)
    print(f"OK {CONFIG_PATH}")
    if not cfg.wake_word.model.is_file():
        print(
            f"WARNING: wake model missing at {cfg.wake_word.model}. "
            "Service may start but wake detection is disabled. "
            "sayso-satellite test-wake-word"
        )
    return 0


def cmd_devices(_: argparse.Namespace) -> int:
    print("PipeWire/Pulse sources:")
    subprocess.call(["pactl", "list", "short", "sources"])
    print("\nPipeWire/Pulse sinks:")
    subprocess.call(["pactl", "list", "short", "sinks"])
    print("\nALSA capture:")
    subprocess.call(["arecord", "-l"])
    print("\nALSA playback:")
    subprocess.call(["aplay", "-l"])
    venv_python = Path("/opt/sayso-satellite/.venv/bin/python")
    if venv_python.is_file():
        print("\nLinux Voice Assistant input devices:")
        subprocess.call(
            [str(venv_python), "-m", "linux_voice_assistant", "--list-input-devices"],
            cwd="/opt/sayso-satellite/linux-voice-assistant",
        )
        print("\nLinux Voice Assistant output devices:")
        subprocess.call(
            [str(venv_python), "-m", "linux_voice_assistant", "--list-output-devices"],
            cwd="/opt/sayso-satellite/linux-voice-assistant",
        )
    return 0


def _level_report(wav_path: Path) -> None:
    import wave

    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sw != 2:
        print(f"sample width {sw} bytes (expected 2)")
        return
    data = np.frombuffer(frames, dtype="<i2")
    if nch > 1:
        data = data.reshape(-1, nch)[:, 0]
    peak = int(np.max(np.abs(data))) if data.size else 0
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) if data.size else 0.0
    clip_count = int(np.sum(np.abs(data) >= 32767))
    clip_pct = (clip_count / max(data.size, 1)) * 100
    db = 20 * np.log10(rms / 32768.0) if rms > 0 else -120.0
    print(f"file={wav_path} rate={rate} samples={data.size}")
    print(f"peak={peak}/32767  rms={rms:.1f}  rms_dbfs={db:.1f}  clipped={clip_count} ({clip_pct:.2f}%)")
    if clip_pct > 1:
        print("WARNING: clipping detected — move the Snowball back or lower analog gain")
    elif peak < 1000:
        print("WARNING: input is very quiet — speak closer to the Snowball")


def cmd_test_mic(_: argparse.Namespace) -> int:
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    wav = CHECK_DIR / "mic-check.wav"
    print("Recording 5 seconds from the Snowball…")
    rc = subprocess.call(
        [
            "timeout",
            "5",
            "parecord",
            "--device",
            PULSE_SOURCE,
            "--file-format=wav",
            "--rate=16000",
            "--channels=1",
            "--format=s16le",
            str(wav),
        ]
    )
    if rc not in (0, 124):
        print("parecord failed", rc)
        return rc or 1
    _level_report(wav)
    print("Playing recording on the wired speaker…")
    return subprocess.call(["paplay", "--device", PULSE_SINK, str(wav)])


def cmd_test_speaker(_: argparse.Namespace) -> int:
    tone = Path("/opt/sayso-satellite/sounds/wake.wav")
    if not tone.is_file():
        print(f"missing {tone}")
        return 1
    print("Playing local notification tone…")
    return subprocess.call(["paplay", "--device", PULSE_SINK, str(tone)])


def cmd_test_wake(_: argparse.Namespace) -> int:
    cfg = load_config()
    path = cfg.wake_word.model
    print(f"model={path}")
    if not path.is_file():
        print(
            "FAIL: model missing. Copy a LiveKit-exported SaySo ONNX classifier to:\n"
            f"  {path}\n"
            "Then run: sayso-satellite test-wake-word\n"
            "Do not substitute hey_livekit, hey_jarvis, or another phrase."
        )
        return 2
    import numpy as np
    from livekit.wakeword import WakeWordModel

    model = WakeWordModel(models=[str(path)])
    silence = np.zeros(16000 * 2, dtype=np.int16)
    scores = model.predict(silence)
    print(f"silence_scores={scores}")
    print(
        "Model loaded. Positive/negative sample detection is NOT claimed "
        "without recorded SaySo utterances. Wake is only operational after "
        "those samples pass."
    )
    return 0


def wait_for_audio(timeout: float = 60.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        src = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True)
        snk = subprocess.run(["pactl", "list", "short", "sinks"], capture_output=True, text=True)
        if PULSE_SOURCE in src.stdout and PULSE_SINK in snk.stdout:
            subprocess.call(["pactl", "set-default-source", PULSE_SOURCE])
            subprocess.call(["pactl", "set-default-sink", PULSE_SINK])
            return 0
        time.sleep(0.5)
    print("Audio devices not ready:", PULSE_SOURCE, PULSE_SINK, file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sayso-satellite")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start").set_defaults(func=cmd_start)
    sub.add_parser("stop").set_defaults(func=cmd_stop)
    sub.add_parser("restart").set_defaults(func=cmd_restart)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("logs").set_defaults(func=cmd_logs)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("devices").set_defaults(func=cmd_devices)
    sub.add_parser("test-mic").set_defaults(func=cmd_test_mic)
    sub.add_parser("test-speaker").set_defaults(func=cmd_test_speaker)
    sub.add_parser("test-wake-word").set_defaults(func=cmd_test_wake)
    sub.add_parser("wait-audio").set_defaults(func=lambda _: wait_for_audio())
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    try:
        raise SystemExit(args.func(args))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
