from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import CONFIG_PATH, load_config, validate_config

USER_UNIT = "sayso-satellite.service"
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
    print("\nLinux Voice Assistant input devices:")
    subprocess.call([sys.executable, "-m", "linux_voice_assistant", "--list-input-devices"])
    print("\nLinux Voice Assistant output devices:")
    subprocess.call([sys.executable, "-m", "linux_voice_assistant", "--list-output-devices"])
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
        print("WARNING: clipping detected — move the configured microphone back or lower analog gain")
    elif peak < 1000:
        print("WARNING: input is very quiet — speak closer to the configured microphone")


def _pulse_device(device: str) -> str:
    return device.removeprefix("pulse/")


def _play_sound(path: Path | str, device: str, timeout: float = 15.0) -> int:
    """Play through the same repaired mpv path used by the satellite."""
    from .launcher import _configure_mpv

    try:
        _configure_mpv()
        from linux_voice_assistant.mpv_player import MpvMediaPlayer
        from linux_voice_assistant.player.libmpv import LibMpvPlayer

        done = threading.Event()
        failed = threading.Event()
        original_end_file = LibMpvPlayer._on_end_file

        def observe_end_file(player, event) -> None:
            if getattr(getattr(event, "data", None), "reason", -1) == 4:
                failed.set()
            original_end_file(player, event)

        LibMpvPlayer._on_end_file = observe_end_file
        try:
            player = MpvMediaPlayer(device=device)
        finally:
            LibMpvPlayer._on_end_file = original_end_file
        player.play(str(path), done_callback=done.set)
    except Exception:
        logging.exception("Audio playback failed")
        return 1

    if not done.wait(timeout):
        logging.error("Audio playback timed out after %.1f seconds", timeout)
        try:
            player.stop()
        except Exception:
            logging.exception("Could not stop timed-out audio playback")
        return 1
    return int(failed.is_set())


def cmd_test_mic(_: argparse.Namespace) -> int:
    cfg = load_config()
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    wav = CHECK_DIR / "mic-check.wav"
    print("Recording 5 seconds from the configured microphone…")
    rc = subprocess.call(
        [
            "timeout",
            "5",
            "parecord",
            "--device",
            _pulse_device(cfg.audio.input_device),
            "--file-format=wav",
            "--rate",
            str(cfg.audio.sample_rate),
            "--channels",
            str(cfg.audio.channels),
            "--format=s16le",
            str(wav),
        ]
    )
    if rc not in (0, 124):
        print("parecord failed", rc)
        return rc or 1
    _level_report(wav)
    print("Playing recording on the configured speaker…")
    return _play_sound(wav, cfg.audio.output_device)


def cmd_test_speaker(_: argparse.Namespace) -> int:
    cfg = load_config()
    tone = cfg.sounds.wake
    if not tone.is_file():
        print(f"missing {tone}")
        return 1
    print("Playing local notification tone…")
    return _play_sound(tone, cfg.audio.output_device)


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
    cfg = load_config()
    source = _pulse_device(cfg.audio.input_device)
    sink = _pulse_device(cfg.audio.output_device)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        src = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True)
        snk = subprocess.run(["pactl", "list", "short", "sinks"], capture_output=True, text=True)
        if source in src.stdout and sink in snk.stdout:
            return 0
        time.sleep(0.5)
    print("Audio devices not ready:", source, sink, file=sys.stderr)
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
