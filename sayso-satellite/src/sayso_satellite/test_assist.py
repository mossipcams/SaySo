"""Tests for the Home Assistant Assist pipeline WebSocket client."""

from __future__ import annotations

import json
import pytest

from sayso_satellite.assist import AssistError, run_assist


class FakeWebSocket:
    def __init__(self, incoming: list[str | bytes]) -> None:
        self.incoming = iter(incoming)
        self.sent: list[str | bytes] = []

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        return next(self.incoming)


class FakeConnection:
    def __init__(self, websocket: FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> FakeWebSocket:
        return self.websocket

    async def __aexit__(self, *_args: object) -> None:
        return None


def _message(message_type: str, **payload: object) -> str:
    return json.dumps({"type": message_type, **payload})


def _event(event_type: str, data: dict[str, object]) -> str:
    return _message(
        "event",
        id=1,
        event={"type": event_type, "data": data},
    )


def _tts_output() -> dict[str, str]:
    return {
        "media_id": "media-source://tts/-stream-/abc.mp3",
        "token": "abc.mp3",
        "url": "/api/tts_proxy/abc.mp3",
        "mime_type": "audio/mpeg",
    }


def _successful_messages() -> list[str]:
    return [
        _message("auth_required"),
        _message("auth_ok"),
        _message("result", id=1, success=True, result=None),
        _event("run-start", {"runner_data": {"stt_binary_handler_id": 7}}),
        _event("stt-start", {"engine": "test"}),
        _event("stt-end", {"stt_output": {"text": "turn on the lamp"}}),
        _event("intent-end", {"intent_output": {"response": "Done"}}),
        _event(
            "tts-start",
            {
                "engine": "test-tts",
                "language": "en",
                "voice": "default",
                "tts_input": "Done",
            },
        ),
        _event("tts-end", {"tts_output": _tts_output()}),
        _event("run-end", {}),
    ]


def test_run_assist_authenticates_streams_pcm_and_returns_tts() -> None:
    websocket = FakeWebSocket(_successful_messages())
    calls: list[str] = []

    def connect(url: str) -> FakeConnection:
        calls.append(url)
        return FakeConnection(websocket)

    result = run_assist(
        b"abcd1234",
        token="secret-token",
        websocket_url="ws://ha.example/api/websocket",
        connect=connect,
        chunk_size=4,
    )

    assert calls == ["ws://ha.example/api/websocket"]
    assert json.loads(websocket.sent[0]) == {
        "type": "auth",
        "access_token": "secret-token",
    }
    assert json.loads(websocket.sent[1]) == {
        "type": "assist_pipeline/run",
        "id": 1,
        "start_stage": "stt",
        "end_stage": "tts",
        "input": {"sample_rate": 16_000},
    }
    assert websocket.sent[2:] == [b"\x07abcd", b"\x071234", b"\x07"]
    assert result == {
        "status": "completed",
        "text": "turn on the lamp",
        "intent": {"response": "Done"},
        "tts": _tts_output(),
    }


def test_run_assist_rejects_run_end_before_tts_end() -> None:
    messages = _successful_messages()[:-2] + [_event("run-end", {})]
    websocket = FakeWebSocket(messages)

    with pytest.raises(AssistError, match="TTS"):
        run_assist(
            b"abcd",
            token="secret-token",
            connect=lambda _url: FakeConnection(websocket),
        )


@pytest.mark.parametrize(
    ("tts_output", "error"),
    [
        ({}, "malformed TTS output"),
        ({"token": "abc.mp3"}, "malformed TTS output"),
        ({"url": "/api/tts_proxy/abc.mp3"}, "malformed TTS output"),
        ({"url": "", "token": "abc.mp3", "mime_type": "audio/mpeg"}, "malformed TTS output"),
    ],
)
def test_run_assist_rejects_malformed_tts_output(
    tts_output: dict[str, str],
    error: str,
) -> None:
    messages = _successful_messages()[:-2] + [
        _event("tts-end", {"tts_output": tts_output}),
        _event("run-end", {}),
    ]
    websocket = FakeWebSocket(messages)

    with pytest.raises(AssistError, match=error):
        run_assist(
            b"abcd",
            token="secret-token",
            connect=lambda _url: FakeConnection(websocket),
        )


def test_run_assist_surfaces_tts_pipeline_errors() -> None:
    messages = _successful_messages()[:-3] + [
        _event("error", {"code": "tts-failed", "message": "synthesis broke"}),
    ]
    websocket = FakeWebSocket(messages)

    with pytest.raises(AssistError, match="tts-failed"):
        run_assist(
            b"abcd",
            token="secret-token",
            connect=lambda _url: FakeConnection(websocket),
        )


def test_run_assist_omits_device_id_when_unset() -> None:
    websocket = FakeWebSocket(_successful_messages())

    run_assist(
        b"abcd1234",
        token="secret-token",
        connect=lambda _url: FakeConnection(websocket),
        chunk_size=4,
    )

    run_message = json.loads(websocket.sent[1])
    assert run_message["type"] == "assist_pipeline/run"
    assert "device_id" not in run_message


def test_run_assist_includes_device_id_when_set() -> None:
    websocket = FakeWebSocket(_successful_messages())

    run_assist(
        b"abcd1234",
        token="secret-token",
        device_id="living-room-mac",
        connect=lambda _url: FakeConnection(websocket),
        chunk_size=4,
    )

    run_message = json.loads(websocket.sent[1])
    assert run_message["device_id"] == "living-room-mac"


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        ([_message("auth_required"), _message("auth_invalid")], "authentication"),
        (
            [
                _message("auth_required"),
                _message("auth_ok"),
                _message("result", id=1, success=True, result=None),
                _event("run-start", {"runner_data": {}}),
                _event("stt-start", {"engine": "test"}),
            ],
            "handler",
        ),
        (
            [
                _message("auth_required"),
                _message("auth_ok"),
                _message("result", id=1, success=True, result=None),
                _event("run-start", {"runner_data": {"stt_binary_handler_id": 1}}),
                _event("stt-start", {"engine": "test"}),
                _event("error", {"code": "stt-stream-failed", "message": "broken"}),
            ],
            "stt-stream-failed",
        ),
        ([b"not-json"], "JSON"),
    ],
)
def test_run_assist_fails_closed(messages: list[str | bytes], error: str) -> None:
    websocket = FakeWebSocket(messages)

    with pytest.raises(AssistError, match=error):
        run_assist(
            b"abcd",
            token="secret-token",
            connect=lambda _url: FakeConnection(websocket),
        )
