"""Tests for the satellite CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sayso_satellite.__main__ import HA_DEVICE_ID_ENV_VAR, resolve_device_id


def test_resolve_device_id_prefers_cli_over_environment() -> None:
    environ = {HA_DEVICE_ID_ENV_VAR: "env-device"}

    assert resolve_device_id("cli-device", environ) == "cli-device"


def test_resolve_device_id_uses_environment_when_cli_absent() -> None:
    environ = {HA_DEVICE_ID_ENV_VAR: "env-device"}

    assert resolve_device_id(None, environ) == "env-device"


def test_resolve_device_id_unset_when_cli_and_environment_absent() -> None:
    assert resolve_device_id(None, {}) is None


def test_resolve_device_id_ignores_blank_environment() -> None:
    environ = {HA_DEVICE_ID_ENV_VAR: "   "}

    assert resolve_device_id(None, environ) is None


def test_parser_exposes_device_id_flag() -> None:
    from sayso_satellite.__main__ import _build_parser

    args = _build_parser().parse_args(["--device-id", "cli-device"])

    assert args.device_id == "cli-device"


def test_parser_exposes_live_mode_flags() -> None:
    from sayso_satellite.__main__ import _build_parser

    args = _build_parser().parse_args(["--live", "--capture-ms", "2500"])

    assert args.live is True
    assert args.capture_ms == 2500


def test_parser_exposes_wake_mode_flags() -> None:
    from sayso_satellite.__main__ import _build_parser

    args = _build_parser().parse_args(
        ["--live", "--wake", "--wake-threshold", "4000", "--wake-hits", "2", "--listen-ms", "5000"]
    )

    assert args.live is True
    assert args.wake is True
    assert args.wake_threshold == 4000
    assert args.wake_hits == 2
    assert args.listen_ms == 5000


def test_main_rejects_wake_without_live() -> None:
    from sayso_satellite.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(["sayso_satellite", "--wake", "--ha-token", "secret-token"])

    assert exc.value.code == 1


def test_resolve_wake_threshold_prefers_cli_over_environment() -> None:
    from sayso_satellite.__main__ import WAKE_THRESHOLD_ENV_VAR, resolve_wake_threshold

    environ = {WAKE_THRESHOLD_ENV_VAR: "1234.5"}

    assert resolve_wake_threshold(999.0, environ) == 999.0


def test_main_wake_mode_skips_assist_when_no_detection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sayso_satellite.__main__ import main

    with (
        patch("sayso_satellite.__main__.open_mac_microphone") as mock_open,
        patch("sayso_satellite.__main__.capture_wake_pcm", return_value=None) as mock_capture,
        patch("sayso_satellite.__main__.run_assist") as mock_run,
    ):
        main(
            [
                "sayso_satellite",
                "--live",
                "--wake",
                "--capture-ms",
                "1500",
                "--listen-ms",
                "500",
                "--ha-token",
                "secret-token",
            ]
        )

    mock_open.assert_called_once()
    mock_capture.assert_called_once()
    mock_run.assert_not_called()
    assert capsys.readouterr().out == ""


def test_main_wake_mode_runs_assist_after_detection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sayso_satellite.__main__ import main

    pcm = b"\x00\x01" * 100
    result = {"status": "completed", "text": "turn on the lamp", "intent": {}}

    with (
        patch("sayso_satellite.__main__.open_mac_microphone") as mock_open,
        patch("sayso_satellite.__main__.capture_wake_pcm", return_value=pcm) as mock_capture,
        patch("sayso_satellite.__main__.run_assist", return_value=result) as mock_run,
    ):
        main(
            [
                "sayso_satellite",
                "--live",
                "--wake",
                "--capture-ms",
                "1500",
                "--ha-token",
                "secret-token",
            ]
        )

    mock_open.assert_called_once()
    mock_capture.assert_called_once()
    assert mock_capture.call_args.kwargs["capture_ms"] == 1500
    mock_run.assert_called_once_with(
        pcm,
        token="secret-token",
        websocket_url="ws://127.0.0.1:8123/api/websocket",
        device_id=None,
        timeout=None,
    )
    assert capsys.readouterr().out == "turn on the lamp\n"


def test_parser_rejects_audio_file_and_live_together() -> None:
    from sayso_satellite.__main__ import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--live", "--audio-file", "sample.bin"])

def test_main_prints_usage_when_no_args(capsys: pytest.CaptureFixture[str]) -> None:
    from sayso_satellite.__main__ import main

    with patch("sys.argv", ["sayso_satellite"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


def test_main_live_mode_captures_from_microphone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sayso_satellite.__main__ import main

    pcm = b"\x00\x01" * 100
    result = {"status": "completed", "text": "turn on the lamp", "intent": {}}

    with (
        patch("sayso_satellite.__main__.open_mac_microphone") as mock_open,
        patch("sayso_satellite.__main__.capture_live_pcm", return_value=pcm) as mock_capture,
        patch("sayso_satellite.__main__.run_assist", return_value=result) as mock_run,
    ):
        main(
            [
                "sayso_satellite",
                "--live",
                "--capture-ms",
                "1500",
                "--ha-token",
                "secret-token",
            ]
        )

    mock_open.assert_called_once()
    mock_capture.assert_called_once()
    assert mock_capture.call_args.kwargs["duration_ms"] == 1500
    mock_run.assert_called_once_with(
        pcm,
        token="secret-token",
        websocket_url="ws://127.0.0.1:8123/api/websocket",
        device_id=None,
        timeout=None,
    )
    assert capsys.readouterr().out == "turn on the lamp\n"


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
        device_id=None,
        timeout=None,
    )
    assert capsys.readouterr().out == "turn on the lamp\n"


def test_main_passes_cli_device_id_to_assist(tmp_path: Path) -> None:
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
                "--ha-token",
                "secret-token",
                "--device-id",
                "cli-device",
            ]
        )

    assert mock_run.call_args.kwargs["device_id"] == "cli-device"


def test_main_passes_environment_device_id_to_assist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sayso_satellite.__main__ import main

    pcm = b"\x00\x01" * 100
    audio_path = tmp_path / "sample.bin"
    audio_path.write_bytes(pcm)
    monkeypatch.setenv(HA_DEVICE_ID_ENV_VAR, "env-device")

    with patch(
        "sayso_satellite.__main__.run_assist",
        return_value={"status": "completed", "text": "done", "intent": {}},
    ) as mock_run:
        main(["sayso_satellite", "--audio-file", str(audio_path), "--ha-token", "secret-token"])

    assert mock_run.call_args.kwargs["device_id"] == "env-device"


def test_main_cli_device_id_wins_over_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sayso_satellite.__main__ import main

    pcm = b"\x00\x01" * 100
    audio_path = tmp_path / "sample.bin"
    audio_path.write_bytes(pcm)
    monkeypatch.setenv(HA_DEVICE_ID_ENV_VAR, "env-device")

    with patch(
        "sayso_satellite.__main__.run_assist",
        return_value={"status": "completed", "text": "done", "intent": {}},
    ) as mock_run:
        main(
            [
                "sayso_satellite",
                "--audio-file",
                str(audio_path),
                "--ha-token",
                "secret-token",
                "--device-id",
                "cli-device",
            ]
        )

    assert mock_run.call_args.kwargs["device_id"] == "cli-device"


def test_main_leaves_device_id_unset_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sayso_satellite.__main__ import main

    pcm = b"\x00\x01" * 100
    audio_path = tmp_path / "sample.bin"
    audio_path.write_bytes(pcm)
    monkeypatch.delenv(HA_DEVICE_ID_ENV_VAR, raising=False)

    with patch(
        "sayso_satellite.__main__.run_assist",
        return_value={"status": "completed", "text": "done", "intent": {}},
    ) as mock_run:
        main(["sayso_satellite", "--audio-file", str(audio_path), "--ha-token", "secret-token"])

    assert mock_run.call_args.kwargs["device_id"] is None


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
        device_id=None,
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
                "speech": {"plain": {"speech": "turn on the lamp"}},
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
        device_id=None,
        timeout=None,
    )
    assert capsys.readouterr().out == "turn on the lamp\n"


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
        return_value={"status": "completed", "text": "turn on the lamp", "intent": {}},
    ) as mock_run:
        main(["sayso_satellite", "--audio-file", str(audio_path)])

    mock_run.assert_called_once_with(
        pcm,
        token="environment-secret",
        websocket_url="ws://ha.example/api/websocket",
        device_id=None,
        timeout=None,
    )
    captured = capsys.readouterr()
    assert captured.out == "turn on the lamp\n"
    assert "environment-secret" not in captured.out + captured.err


def test_main_plays_tts_when_present(
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
        "intent": {},
        "tts": {
            "media_id": "media-source://tts/-stream-/abc.mp3",
            "token": "abc.mp3",
            "url": "/api/tts_proxy/abc.mp3",
            "mime_type": "audio/mpeg",
        },
    }

    with (
        patch("sayso_satellite.__main__.run_assist", return_value=result),
        patch("sayso_satellite.__main__.play_tts_response") as mock_play,
    ):
        main(["sayso_satellite", "--audio-file", str(audio_path), "--ha-token", "secret-token"])

    mock_play.assert_called_once()
    assert mock_play.call_args.kwargs["token"] == "secret-token"
    assert mock_play.call_args.kwargs["base_url"] == "http://127.0.0.1:8123"
    assert capsys.readouterr().out == "turn on the lamp\n"


def test_main_exits_non_zero_on_playback_failure(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from sayso_satellite.__main__ import main
    from sayso_satellite.playback import PlaybackError

    pcm = b"\x00\x01" * 100
    audio_path = tmp_path / "sample.bin"
    audio_path.write_bytes(pcm)
    result = {
        "status": "completed",
        "text": "turn on the lamp",
        "intent": {},
        "tts": {
            "media_id": "media-source://tts/-stream-/abc.mp3",
            "token": "abc.mp3",
            "url": "/api/tts_proxy/abc.mp3",
            "mime_type": "audio/mpeg",
        },
    }

    with (
        patch("sayso_satellite.__main__.run_assist", return_value=result),
        patch(
            "sayso_satellite.__main__.play_tts_response",
            side_effect=PlaybackError("playback failed"),
        ),
    ):
        with pytest.raises(SystemExit) as exc:
            main(["sayso_satellite", "--audio-file", str(audio_path), "--ha-token", "secret-token"])

    assert exc.value.code == 1
    assert "playback failed" in capsys.readouterr().err


def test_handle_assist_result_print_only_does_not_construct_default_audio_player(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sayso_satellite.__main__ import handle_assist_result

    result = {
        "status": "completed",
        "text": "turn on the lamp",
        "intent": {},
    }

    with patch("sayso_satellite.__main__.default_audio_player") as mock_default_player:
        handle_assist_result(
            result,
            ha_token="secret-token",
            websocket_url="ws://127.0.0.1:8123/api/websocket",
        )

    mock_default_player.assert_not_called()
    assert capsys.readouterr().out == "turn on the lamp\n"


def test_handle_assist_result_plays_earcon_for_completed_action(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sayso_satellite.__main__ import handle_assist_result

    player = RecordingPlayer()
    result = {
        "status": "completed",
        "text": "turn off the lamp",
        "intent": {"response": "Done."},
        "tts": {
            "media_id": "media-source://tts/-stream-/abc.mp3",
            "token": "abc.mp3",
            "url": "/api/tts_proxy/abc.mp3",
            "mime_type": "audio/mpeg",
        },
    }

    with patch("sayso_satellite.__main__.play_tts_response") as mock_play_tts:
        handle_assist_result(
            result,
            ha_token="secret-token",
            websocket_url="ws://127.0.0.1:8123/api/websocket",
            player=player,
        )

    mock_play_tts.assert_not_called()
    assert len(player.calls) == 1
    assert player.calls[0][1] == "audio/wav"
    assert capsys.readouterr().out == ""


def test_handle_assist_result_plays_tts_for_clarification(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sayso_satellite.__main__ import handle_assist_result

    player = RecordingPlayer()
    result = {
        "status": "completed",
        "text": "turn off the lights",
        "intent": {
            "response": {
                "speech": {"plain": {"speech": "which lights?"}},
            },
        },
        "tts": {
            "media_id": "media-source://tts/-stream-/abc.mp3",
            "token": "abc.mp3",
            "url": "/api/tts_proxy/abc.mp3",
            "mime_type": "audio/mpeg",
        },
    }

    with patch("sayso_satellite.__main__.play_tts_response") as mock_play_tts:
        handle_assist_result(
            result,
            ha_token="secret-token",
            websocket_url="ws://127.0.0.1:8123/api/websocket",
            player=player,
        )

    mock_play_tts.assert_called_once()
    assert player.calls == []
    assert capsys.readouterr().out == "which lights?\n"


def test_handle_assist_result_plays_tts_for_error_response(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sayso_satellite.__main__ import handle_assist_result

    player = RecordingPlayer()
    result = {
        "status": "completed",
        "text": "turn off the lamp",
        "intent": {"response": "entity unavailable"},
        "tts": {
            "media_id": "media-source://tts/-stream-/abc.mp3",
            "token": "abc.mp3",
            "url": "/api/tts_proxy/abc.mp3",
            "mime_type": "audio/mpeg",
        },
    }

    with patch("sayso_satellite.__main__.play_tts_response") as mock_play_tts:
        handle_assist_result(
            result,
            ha_token="secret-token",
            websocket_url="ws://127.0.0.1:8123/api/websocket",
            player=player,
        )

    mock_play_tts.assert_called_once()
    assert player.calls == []
    assert capsys.readouterr().out == "entity unavailable\n"


def test_handle_assist_result_plays_tts_for_query_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sayso_satellite.__main__ import handle_assist_result

    player = RecordingPlayer()
    result = {
        "status": "completed",
        "text": "is the lamp on",
        "intent": {"response": "on"},
        "tts": {
            "media_id": "media-source://tts/-stream-/abc.mp3",
            "token": "abc.mp3",
            "url": "/api/tts_proxy/abc.mp3",
            "mime_type": "audio/mpeg",
        },
    }

    with patch("sayso_satellite.__main__.play_tts_response") as mock_play_tts:
        handle_assist_result(
            result,
            ha_token="secret-token",
            websocket_url="ws://127.0.0.1:8123/api/websocket",
            player=player,
        )

    mock_play_tts.assert_called_once()
    assert player.calls == []
    assert capsys.readouterr().out == "on\n"


def test_main_rejects_loop_without_live_wake() -> None:
    from sayso_satellite.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(["sayso_satellite", "--loop", "--ha-token", "secret-token"])

    assert exc.value.code == 1


def test_main_loop_mode_runs_continuous_loop() -> None:
    from sayso_satellite.__main__ import main

    with (
        patch("sayso_satellite.__main__.open_mac_microphone") as mock_open,
        patch("sayso_satellite.__main__.run_continuous_loop") as mock_loop,
    ):
        main(
            [
                "sayso_satellite",
                "--live",
                "--wake",
                "--loop",
                "--capture-ms",
                "1500",
                "--ha-token",
                "secret-token",
            ]
        )

    mock_open.assert_called_once()
    mock_loop.assert_called_once()
    assert mock_loop.call_args.kwargs["capture_ms"] == 1500
    assert mock_loop.call_args.kwargs["on_turn"] is not None


def test_main_loop_does_not_playback_when_assist_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sayso_satellite.__main__ import main
    from sayso_satellite.assist import AssistError

    pcm = b"\x00\x01" * 100

    def fake_loop(_mic, _engine, *, on_turn, **kwargs):
        on_turn(pcm)

    with (
        patch("sayso_satellite.__main__.open_mac_microphone"),
        patch("sayso_satellite.__main__.run_continuous_loop", side_effect=fake_loop),
        patch(
            "sayso_satellite.__main__.run_assist",
            side_effect=AssistError("assist failed"),
        ) as mock_run,
        patch("sayso_satellite.__main__.handle_assist_result") as mock_handle,
        patch("sayso_satellite.__main__.play_tts_response") as mock_play,
    ):
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "sayso_satellite",
                    "--live",
                    "--wake",
                    "--loop",
                    "--ha-token",
                    "secret-token",
                ]
            )

    assert exc.value.code == 1
    mock_run.assert_called_once()
    mock_handle.assert_not_called()
    mock_play.assert_not_called()
    assert "assist failed" in capsys.readouterr().err


def test_main_loop_exits_cleanly_on_keyboard_interrupt() -> None:
    from sayso_satellite.__main__ import main

    with (
        patch("sayso_satellite.__main__.open_mac_microphone"),
        patch(
            "sayso_satellite.__main__.run_continuous_loop",
            side_effect=KeyboardInterrupt,
        ),
    ):
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "sayso_satellite",
                    "--live",
                    "--wake",
                    "--loop",
                    "--ha-token",
                    "secret-token",
                ]
            )

    assert exc.value.code == 0


class RecordingPlayer:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    def play(self, audio: bytes, *, mime_type: str) -> None:
        self.calls.append((audio, mime_type))

