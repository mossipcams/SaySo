from pathlib import Path


SERVICE = Path(__file__).parents[1] / "systemd" / "sayso-satellite.service"


def test_user_service_uses_user_runtime_and_state_paths() -> None:
    unit = SERVICE.read_text(encoding="utf-8")

    assert "WorkingDirectory=" not in unit
    assert "Environment=XDG_RUNTIME_DIR=" not in unit
    assert "Environment=PULSE_SERVER=" not in unit
    assert "Environment=PREFERENCES_FILE=%h/.local/state/sayso-satellite/preferences.json" in unit
