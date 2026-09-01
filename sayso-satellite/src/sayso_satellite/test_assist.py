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


def _successful_messages() -> list[str]:
    return [
        _message("auth_required"),
        _message("auth_ok"),
        _message("result", id=1, success=True, result=None),
        _event("run-start", {"runner_data": {"stt_binary_handler_id": 7}}),
        _event("stt-start", {"engine": "test"}),
        _event("stt-end", {"stt_output": {"text": "turn on the lamp"}}),
        _event("intent-end", {"intent_output": {"response": "Done"}}),
        _event("run-end", {}),
    ]


def test_run_assist_authenticates_streams_pcm_and_returns_intent() -> None:
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
        "end_stage": "intent",
        "input": {"sample_rate": 16_000},
    }
    assert websocket.sent[2:] == [b"\x07abcd", b"\x071234", b"\x07"]
    assert result == {
        "status": "completed",
        "text": "turn on the lamp",
        "intent": {"response": "Done"},
    }


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
