"""Tests for the satellite CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

def test_main_prints_usage_when_no_args(capsys: pytest.CaptureFixture[str]) -> None:
    from sayso_satellite.__main__ import main

    with patch("sys.argv", ["sayso_satellite"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_main_sends_audio_from_file(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from sayso_satellite.__main__ import main

    pcm = b"\x00\x01" * 100
    audio_path = Path(tmp_path) / "sample.bin"
    audio_path.write_bytes(pcm)
    result = {"status": "completed", "text": "turn on the lamp", "intent": {}}

    with patch(
        "sayso_satellite.__main__.run_assist",
        return_value=result,
    ) as mock_run:
        main(["sayso_satellite", "--audio-file", str(audio_path), "--ha-token", "secret-token"])

    mock_run.assert_called_once_with(
        pcm,
        token="secret-token",
        websocket_url="ws://127.0.0.1:8123/api/websocket",
        timeout=None,
    )
    assert capsys.readouterr().out == "turn on the lamp\n"


def test_main_audio_exits_non_zero_on_assist_error(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from sayso_satellite.__main__ import main

    pcm = b"\x00\x00" * 50
    audio_path = tmp_path / "sample.bin"
    audio_path.write_bytes(pcm)
    from sayso_satellite.assist import AssistError

    with patch(
        "sayso_satellite.__main__.run_assist",
        side_effect=AssistError("authentication failed"),
    ):
        with pytest.raises(SystemExit) as exc:
            main(["sayso_satellite", "--audio-file", str(audio_path), "--ha-token", "secret-token"])

    assert exc.value.code == 1
    assert "authentication failed" in capsys.readouterr().err


def test_main_passes_timeout_to_assist(
    tmp_path: Path,
) -> None:
    from sayso_satellite.__main__ import main

    pcm = b"\x00\x01" * 100
    audio_path = tmp_path / "sample.bin"
    audio_path.write_bytes(pcm)

    with patch(
        "sayso_satellite.__main__.run_assist",
        return_value={"status": "completed", "text": "done", "intent": {}},
    ) as mock_run:
        main(
            [
                "sayso_satellite",
                "--audio-file",
                str(audio_path),
                "--timeout",
                "240",
                "--ha-token",
                "secret-token",
            ]
        )

    mock_run.assert_called_once_with(
        pcm,
        token="secret-token",
        websocket_url="ws://127.0.0.1:8123/api/websocket",
        timeout=240.0,
    )


def test_main_sends_corner_lamp_fixture_via_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sayso_satellite.__main__ import main

    fixtures = Path(__file__).resolve().parents[3] / "evals" / "fixtures"
    pcm_path = fixtures / "turn_off_the_corner_lamp.pcm"
    pcm = pcm_path.read_bytes()
    result = {"status": "completed", "text": "turn off the lamp", "intent": {}}

    with patch(
        "sayso_satellite.__main__.run_assist",
        return_value=result,
    ) as mock_run:
        main(["sayso_satellite", "--audio-file", str(pcm_path), "--ha-token", "secret-token"])

    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == pcm
    assert capsys.readouterr().out == "turn off the lamp\n"


def test_main_sends_audio_to_home_assistant_assist(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from sayso_satellite.__main__ import main

    pcm = b"\x00\x01" * 100
    audio_path = tmp_path / "sample.bin"
    audio_path.write_bytes(pcm)
    result = {
        "status": "completed",
        "text": "turn on the lamp",
        "intent": {
            "response": {
                "speech": {"plain": {"speech": "Done"}},
            },
        },
    }

    with patch(
        "sayso_satellite.__main__.run_assist",
        return_value=result,
    ) as mock_run:
        main(
            [
                "sayso_satellite",
                "--audio-file",
                str(audio_path),
                "--ha-websocket-url",
                "ws://ha.example/api/websocket",
                "--ha-token",
                "secret-token",
            ]
        )

    mock_run.assert_called_once_with(
        pcm,
        token="secret-token",
        websocket_url="ws://ha.example/api/websocket",
        timeout=None,
    )
    assert capsys.readouterr().out == "Done\n"


def test_main_uses_assist_environment_configuration(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sayso_satellite.__main__ import main

    pcm = b"\x00\x01" * 100
    audio_path = tmp_path / "sample.bin"
    audio_path.write_bytes(pcm)
    monkeypatch.setenv("SAYSO_HA_TOKEN", "environment-secret")
    monkeypatch.setenv("SAYSO_HA_WEBSOCKET_URL", "ws://ha.example/api/websocket")

    with patch(
        "sayso_satellite.__main__.run_assist",
        return_value={"status": "completed", "text": "Done", "intent": {}},
    ) as mock_run:
        main(["sayso_satellite", "--audio-file", str(audio_path)])

    mock_run.assert_called_once_with(
        pcm,
        token="environment-secret",
        websocket_url="ws://ha.example/api/websocket",
        timeout=None,
    )
    captured = capsys.readouterr()
    assert captured.out == "Done\n"
    assert "environment-secret" not in captured.out + captured.err
