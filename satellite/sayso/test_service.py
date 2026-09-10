from __future__ import annotations

import configparser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from satellite.sayso import cli

UNIT_NAME = "sayso-satellite.service"
SERVICE = Path(__file__).parents[1] / "systemd" / UNIT_NAME


def _unit() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False)
    # systemd unit keys are case-sensitive and repeat (Environment=).
    parser.optionxform = str  # type: ignore[assignment]
    parser.read_string(SERVICE.read_text(encoding="utf-8"))
    return parser


def _directives() -> list[str]:
    """Return the unit's directive lines, without comments."""
    return [
        line.strip()
        for line in SERVICE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _environment() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in SERVICE.read_text(encoding="utf-8").splitlines():
        if line.startswith("Environment="):
            key, _, value = line.removeprefix("Environment=").partition("=")
            values[key] = value
    return values


def test_runs_as_a_system_service_under_a_dedicated_user() -> None:
    """A system service running as User= avoids needing a systemd --user manager.

    A --user unit would require lingering plus systemd-logind and D-Bus, which
    DietPi does not run by default.
    """
    unit = _unit()

    assert unit["Service"]["User"] == "sayso"
    assert unit["Service"]["Group"] == "audio"
    assert unit["Service"]["ExecStart"] == (
        "/opt/sayso-satellite/.venv/bin/python -m sayso.launcher"
    )
    assert unit["Service"]["Restart"] == "on-failure"
    assert unit["Install"]["WantedBy"] == "multi-user.target"


def test_state_does_not_use_the_home_specifier() -> None:
    """%h is not influenced by User= in a system unit; it resolves to root's home.

    Using it here would write preferences under /root while the process runs as
    sayso. StateDirectory is created by systemd owned by User=/Group=.
    """
    unit = _unit()

    assert not [line for line in _directives() if "%h" in line]
    assert unit["Service"]["StateDirectory"] == "sayso-satellite"
    assert unit["Service"]["StateDirectoryMode"] == "0700"
    assert (
        _environment()["PREFERENCES_FILE"]
        == "%S/sayso-satellite/preferences.json"
    )


def test_audio_is_reachable_without_a_user_session() -> None:
    """No user session means no /run/user/<uid>, so both paths are explicit."""
    unit = _unit()
    environment = _environment()

    assert unit["Service"]["RuntimeDirectory"] == "sayso-satellite"
    assert environment["XDG_RUNTIME_DIR"] == "%t/sayso-satellite"
    assert environment["PULSE_SERVER"] == "unix:/run/pulse/native"
    # Operators override the socket without editing the shipped unit.
    assert "EnvironmentFile=-/etc/default/sayso-satellite" in _directives()


def test_audio_readiness_still_gates_startup() -> None:
    """wait-audio remains the readiness gate now that ordering is advisory."""
    unit = _unit()

    assert unit["Service"]["ExecStartPre"] == (
        "/opt/sayso-satellite/.venv/bin/python -m sayso.cli wait-audio"
    )


def test_unit_is_not_a_user_unit() -> None:
    """Nothing may reintroduce the --user dependency the change removed."""
    directives = _directives()

    assert "WantedBy=default.target" not in directives
    assert not [line for line in directives if "/run/user/" in line]


@pytest.mark.parametrize(
    ("command", "action"),
    [(cli.cmd_start, "start"), (cli.cmd_stop, "stop"), (cli.cmd_restart, "restart")],
)
def test_state_changes_target_the_system_unit(
    monkeypatch: pytest.MonkeyPatch, command, action: str
) -> None:
    """State changes drive the system manager, never `systemctl --user`."""
    call_process = Mock(return_value=0)
    monkeypatch.setattr(cli.subprocess, "call", call_process)
    monkeypatch.setattr(cli.os, "geteuid", lambda: 0)

    assert command(SimpleNamespace()) == 0
    assert call_process.call_args.args[0] == ["systemctl", action, UNIT_NAME]


def test_state_changes_escalate_when_unprivileged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unprivileged operator gets sudo rather than an auth-required error."""
    call_process = Mock(return_value=0)
    monkeypatch.setattr(cli.subprocess, "call", call_process)
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/sudo")

    assert cli.cmd_restart(SimpleNamespace()) == 0
    assert call_process.call_args.args[0] == [
        "/usr/bin/sudo",
        "systemctl",
        "restart",
        UNIT_NAME,
    ]


def test_state_changes_run_directly_without_sudo_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing sudo must not turn into a TypeError before systemctl runs."""
    call_process = Mock(return_value=1)
    monkeypatch.setattr(cli.subprocess, "call", call_process)
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)

    assert cli.cmd_stop(SimpleNamespace()) == 1
    assert call_process.call_args.args[0] == ["systemctl", "stop", UNIT_NAME]


def test_reads_do_not_require_privileges(monkeypatch: pytest.MonkeyPatch) -> None:
    """status and logs work as any user, so they never reach for sudo."""
    call_process = Mock(return_value=0)
    monkeypatch.setattr(cli.subprocess, "call", call_process)
    monkeypatch.setattr(cli.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        cli.shutil, "which", Mock(side_effect=AssertionError("must not use sudo"))
    )

    assert cli.cmd_status(SimpleNamespace()) == 0
    assert call_process.call_args.args[0] == ["systemctl", "status", UNIT_NAME]

    assert cli.cmd_logs(SimpleNamespace()) == 0
    assert call_process.call_args.args[0] == ["journalctl", "-u", UNIT_NAME, "-f"]


def test_cli_unit_matches_the_shipped_unit_file() -> None:
    """The CLI and the installed unit must not drift apart."""
    assert cli.UNIT == UNIT_NAME == SERVICE.name
