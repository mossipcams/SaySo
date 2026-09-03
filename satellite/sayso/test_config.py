from pathlib import Path

import pytest

from .config import (
    AppConfig,
    AudioCfg,
    HomeAssistantCfg,
    SatelliteCfg,
    SoundsCfg,
    WakeWordCfg,
    validate_config,
)


def test_validate_config_rejects_missing_wake_model(tmp_path: Path) -> None:
    sound = Path(__file__).parents[1] / "sounds" / "wake.wav"
    cfg = AppConfig(
        satellite=SatelliteCfg(name="sayso-living-room", area="Living Room"),
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
