from pathlib import Path

import pytest
import yaml

from .config import (
    AppConfig,
    AudioCfg,
    HomeAssistantCfg,
    SatelliteCfg,
    SoundsCfg,
    WakeWordCfg,
    load_config,
    validate_config,
)


def test_load_config_reads_device_name_separate_from_friendly_name(tmp_path: Path) -> None:
    sound = Path(__file__).parents[1] / "sounds" / "wake.wav"
    model = tmp_path / "wake.onnx"
    model.write_bytes(b"onnx")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "satellite": {
                    "name": "Living Room",
                    "device_name": "sayso-living-room",
                    "area": "Living Room",
                },
                "home_assistant": {"port": 6053},
                "audio": {
                    "input_device": "mic",
                    "output_device": "pulse/speaker",
                    "sample_rate": 16000,
                    "channels": 1,
                    "noise_suppression": 0,
                    "auto_gain": 0,
                },
                "wake_word": {
                    "provider": "livekit",
                    "phrase": "SaySo",
                    "model": str(model),
                    "threshold": 0.5,
                    "refractory_seconds": 2.0,
                    "preroll_ms": 500,
                    "post_tts_cooldown_ms": 500,
                },
                "sounds": {
                    "wake": str(sound),
                    "failure": str(sound),
                    "unavailable": str(sound),
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.satellite.name == "Living Room"
    assert cfg.satellite.device_name == "sayso-living-room"


def test_load_config_reads_native_capture_profile(tmp_path: Path) -> None:
    sound = Path(__file__).parents[1] / "sounds" / "wake.wav"
    model = tmp_path / "wake.onnx"
    model.write_bytes(b"onnx")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "satellite": {
                    "name": "Living Room",
                    "device_name": "sayso-living-room",
                    "area": "Living Room",
                },
                "home_assistant": {"port": 6053},
                "audio": {
                    "input_device": "mic",
                    "output_device": "pulse/speaker",
                    "sample_rate": 16000,
                    "channels": 1,
                    "noise_suppression": 0,
                    "auto_gain": 0,
                    "capture_rate": 44100,
                    "mic_gain_db": 6.0,
                    "aec_gate_ms": 150,
                    "stt_capture_dir": str(tmp_path / "stt"),
                    "stt_capture_enabled": True,
                },
                "wake_word": {
                    "provider": "livekit",
                    "phrase": "SaySo",
                    "model": str(model),
                    "threshold": 0.5,
                    "refractory_seconds": 2.0,
                    "preroll_ms": 500,
                    "post_tts_cooldown_ms": 500,
                },
                "sounds": {
                    "wake": str(sound),
                    "failure": str(sound),
                    "unavailable": str(sound),
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.audio.capture_rate == 44100
    assert cfg.audio.mic_gain_db == 6.0
    assert cfg.audio.aec_gate_ms == 150
    assert cfg.audio.stt_capture_dir == tmp_path / "stt"
    assert cfg.audio.stt_capture_enabled is True
    # Noise suppression stays off by default: measure gain before suppressing.
    assert cfg.audio.noise_suppression == 0


def test_audio_defaults_preserve_legacy_configs(tmp_path: Path) -> None:
    """A config predating the audio profile must still load with safe defaults."""
    sound = Path(__file__).parents[1] / "sounds" / "wake.wav"
    model = tmp_path / "wake.onnx"
    model.write_bytes(b"onnx")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "satellite": {
                    "name": "Living Room",
                    "device_name": "sayso-living-room",
                    "area": "Living Room",
                },
                "home_assistant": {"port": 6053},
                "audio": {
                    "input_device": "mic",
                    "output_device": "pulse/speaker",
                    "sample_rate": 16000,
                    "channels": 1,
                    "noise_suppression": 0,
                    "auto_gain": 0,
                },
                "wake_word": {
                    "provider": "livekit",
                    "phrase": "SaySo",
                    "model": str(model),
                    "threshold": 0.5,
                    "refractory_seconds": 2.0,
                    "preroll_ms": 500,
                    "post_tts_cooldown_ms": 500,
                },
                "sounds": {
                    "wake": str(sound),
                    "failure": str(sound),
                    "unavailable": str(sound),
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.audio.capture_rate == 44100
    assert cfg.audio.mic_gain_db == 0.0
    assert cfg.audio.aec_gate_ms == 0


def test_validate_config_rejects_bad_capture_rate(tmp_path: Path) -> None:
    sound = Path(__file__).parents[1] / "sounds" / "wake.wav"
    cfg = AppConfig(
        satellite=SatelliteCfg(
            name="Living Room", device_name="sayso-living-room", area="Living Room"
        ),
        home_assistant=HomeAssistantCfg(port=6053),
        audio=AudioCfg(
            input_device="mic",
            output_device="pulse/speaker",
            sample_rate=16000,
            channels=1,
            noise_suppression=0,
            auto_gain=0,
            capture_rate=96000,
        ),
        wake_word=WakeWordCfg(
            provider="livekit",
            phrase="SaySo",
            model=tmp_path / "wake.onnx",
            threshold=0.5,
            refractory_seconds=2.0,
            preroll_ms=500,
            post_tts_cooldown_ms=500,
        ),
        sounds=SoundsCfg(wake=sound, failure=sound, unavailable=sound),
        secrets={},
    )

    with pytest.raises(ValueError, match="audio.capture_rate"):
        validate_config(cfg, check_port_bind=False)


def test_validate_config_rejects_empty_device_name(tmp_path: Path) -> None:
    sound = Path(__file__).parents[1] / "sounds" / "wake.wav"
    cfg = AppConfig(
        satellite=SatelliteCfg(name="Living Room", device_name="  ", area="Living Room"),
        home_assistant=HomeAssistantCfg(port=6053),
        audio=AudioCfg(
            input_device="mic",
            output_device="pulse/speaker",
            sample_rate=16000,
            channels=1,
            noise_suppression=0,
            auto_gain=0,
        ),
        wake_word=WakeWordCfg(
            provider="livekit",
            phrase="SaySo",
            model=tmp_path / "wake.onnx",
            threshold=0.5,
            refractory_seconds=2.0,
            preroll_ms=500,
            post_tts_cooldown_ms=500,
        ),
        sounds=SoundsCfg(wake=sound, failure=sound, unavailable=sound),
        secrets={},
    )

    with pytest.raises(ValueError, match="satellite.device_name is empty"):
        validate_config(cfg, check_port_bind=False)


def test_validate_config_rejects_missing_wake_model(tmp_path: Path) -> None:
    sound = Path(__file__).parents[1] / "sounds" / "wake.wav"
    cfg = AppConfig(
        satellite=SatelliteCfg(
            name="Living Room",
            device_name="sayso-living-room",
            area="Living Room",
        ),
        home_assistant=HomeAssistantCfg(port=6053),
        audio=AudioCfg(
            input_device="mic",
            output_device="pulse/speaker",
            sample_rate=16000,
            channels=1,
            noise_suppression=0,
            auto_gain=0,
        ),
        wake_word=WakeWordCfg(
            provider="livekit",
            phrase="SaySo",
            model=tmp_path / "missing.onnx",
            threshold=0.5,
            refractory_seconds=2.0,
            preroll_ms=500,
            post_tts_cooldown_ms=500,
        ),
        sounds=SoundsCfg(wake=sound, failure=sound, unavailable=sound),
        secrets={},
    )

    with pytest.raises(ValueError, match="wake_word.model file missing"):
        validate_config(cfg, check_port_bind=False)
