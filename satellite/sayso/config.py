from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

CONFIG_PATH = Path("/etc/sayso-satellite/config.yaml")
SECRETS_PATH = Path("/etc/sayso-satellite/secrets.yaml")


@dataclass
class SatelliteCfg:
    name: str
    area: str


@dataclass
class HomeAssistantCfg:
    port: int


@dataclass
class AudioCfg:
    input_device: str
    output_device: str
    sample_rate: int
    channels: int
    noise_suppression: int
    auto_gain: int


@dataclass
class WakeWordCfg:
    provider: str
    phrase: str
    model: Path
    threshold: float
    refractory_seconds: float
    preroll_ms: int
    post_tts_cooldown_ms: int
    wake_skip_ms: int = 500


@dataclass
class SoundsCfg:
    wake: Path
    failure: Path
    unavailable: Path


@dataclass
class AppConfig:
    satellite: SatelliteCfg
    home_assistant: HomeAssistantCfg
    audio: AudioCfg
    wake_word: WakeWordCfg
    sounds: SoundsCfg
    secrets: dict[str, Any]


def _req(d: dict, *keys: str) -> Any:
    cur: Any = d
    path = []
    for k in keys:
        path.append(k)
        if not isinstance(cur, dict) or k not in cur:
            raise ValueError(f"Missing config key: {'.'.join(path)}")
        cur = cur[k]
    return cur


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    if not path.is_file():
        raise ValueError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    secrets: dict[str, Any] = {}
    if SECRETS_PATH.is_file():
        with SECRETS_PATH.open("r", encoding="utf-8") as fh:
            secrets = yaml.safe_load(fh) or {}

    audio = AudioCfg(
        input_device=str(_req(raw, "audio", "input_device")),
        output_device=str(_req(raw, "audio", "output_device")),
        sample_rate=int(_req(raw, "audio", "sample_rate")),
        channels=int(_req(raw, "audio", "channels")),
        noise_suppression=int(_req(raw, "audio", "noise_suppression")),
        auto_gain=int(_req(raw, "audio", "auto_gain")),
    )
    ww = WakeWordCfg(
        provider=str(_req(raw, "wake_word", "provider")),
        phrase=str(_req(raw, "wake_word", "phrase")),
        model=Path(_req(raw, "wake_word", "model")),
        threshold=float(_req(raw, "wake_word", "threshold")),
        refractory_seconds=float(_req(raw, "wake_word", "refractory_seconds")),
        preroll_ms=int(_req(raw, "wake_word", "preroll_ms")),
        post_tts_cooldown_ms=int(_req(raw, "wake_word", "post_tts_cooldown_ms")),
        wake_skip_ms=int(raw.get("wake_word", {}).get("wake_skip_ms", 500)),
    )
    sounds = SoundsCfg(
        wake=Path(_req(raw, "sounds", "wake")),
        failure=Path(_req(raw, "sounds", "failure")),
        unavailable=Path(_req(raw, "sounds", "unavailable")),
    )
    cfg = AppConfig(
        satellite=SatelliteCfg(
            name=str(_req(raw, "satellite", "name")),
            area=str(_req(raw, "satellite", "area")),
        ),
        home_assistant=HomeAssistantCfg(port=int(_req(raw, "home_assistant", "port"))),
        audio=audio,
        wake_word=ww,
        sounds=sounds,
        secrets=secrets,
    )
    validate_config(cfg, check_port_bind=False)
    return cfg


def validate_config(cfg: AppConfig, check_port_bind: bool = True) -> None:
    errors: list[str] = []
    if not cfg.satellite.name.strip():
        errors.append("satellite.name is empty")
    if cfg.home_assistant.port < 1 or cfg.home_assistant.port > 65535:
        errors.append(f"home_assistant.port {cfg.home_assistant.port} is not a valid TCP port")
    if cfg.audio.sample_rate != 16000:
        errors.append("audio.sample_rate must be 16000")
    if cfg.audio.channels != 1:
        errors.append("audio.channels must be 1")
    if cfg.audio.noise_suppression not in (0, 1, 2, 3, 4):
        errors.append("audio.noise_suppression must be 0-4")
    if not (0 <= cfg.audio.auto_gain <= 31):
        errors.append("audio.auto_gain must be 0-31")
    if not cfg.audio.input_device:
        errors.append("audio.input_device is empty; run sayso-satellite devices")
    if not cfg.audio.output_device:
        errors.append("audio.output_device is empty; run sayso-satellite devices")
    if cfg.wake_word.provider != "livekit":
        errors.append("wake_word.provider must be 'livekit' for this satellite")
    if cfg.wake_word.phrase != "SaySo":
        errors.append("wake_word.phrase must be exactly 'SaySo'")
    if not (0.0 < cfg.wake_word.threshold < 1.0):
        errors.append("wake_word.threshold must be between 0 and 1 exclusive")
    if cfg.wake_word.refractory_seconds < 0:
        errors.append("wake_word.refractory_seconds must be >= 0")
    if cfg.wake_word.preroll_ms < 0:
        errors.append("wake_word.preroll_ms must be >= 0")
    for label, path in (
        ("sounds.wake", cfg.sounds.wake),
        ("sounds.failure", cfg.sounds.failure),
        ("sounds.unavailable", cfg.sounds.unavailable),
    ):
        if not path.is_file():
            errors.append(f"{label} file missing: {path}")
    if check_port_bind:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", cfg.home_assistant.port))
        except OSError as exc:
            pid = os.environ.get("SAYSO_OWN_PORT", "")
            errors.append(
                f"Port {cfg.home_assistant.port} is unavailable ({exc}). "
                "Stop the other process or change home_assistant.port."
            )
        finally:
            sock.close()
    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))
