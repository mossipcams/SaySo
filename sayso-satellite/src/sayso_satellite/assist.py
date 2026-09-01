"""Synchronous client for Home Assistant's Assist pipeline WebSocket API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, AsyncContextManager, Literal, Protocol, TypedDict

from sayso_satellite.capture import BYTES_PER_SAMPLE, SAMPLE_RATE_HZ

DEFAULT_WEBSOCKET_URL = "ws://127.0.0.1:8123/api/websocket"
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_CHUNK_SIZE = 4096


class AssistError(RuntimeError):
    """Raised when the Assist WebSocket conversation cannot be trusted."""


class AssistResult(TypedDict):
    status: Literal["completed"]
    text: str
    intent: dict[str, Any]


class AssistWebSocket(Protocol):
    async def send(self, message: str | bytes) -> None: ...

    async def recv(self) -> str | bytes: ...


Connect = Callable[[str], AsyncContextManager[AssistWebSocket]]


def run_assist(
    pcm: bytes,
    *,
    token: str,
    websocket_url: str = DEFAULT_WEBSOCKET_URL,
    pipeline: str | None = None,
    conversation_id: str | None = None,
    device_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    connect: Connect | None = None,
) -> AssistResult:
    """Run one recorded 16 kHz mono PCM turn through Home Assistant Assist."""

    if not token.strip():
        raise ValueError("token must not be empty")
    if not pcm:
        raise ValueError("PCM is empty")
    if len(pcm) % BYTES_PER_SAMPLE:
        raise ValueError("PCM byte length must be even")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    try:
        return asyncio.run(
            asyncio.wait_for(
                _run_assist(
                    pcm,
                    token=token,
                    websocket_url=websocket_url,
                    pipeline=pipeline,
                    conversation_id=conversation_id,
                    device_id=device_id,
                    chunk_size=chunk_size,
                    connect=connect,
                ),
                timeout=timeout,
            )
        )
    except TimeoutError as exc:
        raise AssistError("Assist pipeline timed out") from exc


async def _run_assist(
    pcm: bytes,
    *,
    token: str,
    websocket_url: str,
    pipeline: str | None,
    conversation_id: str | None,
    device_id: str | None,
    chunk_size: int,
    connect: Connect | None,
) -> AssistResult:
    websocket_connect = connect or _default_connect()
    async with websocket_connect(websocket_url) as websocket:
        await _authenticate(websocket, token)

        run_id = 1
        run_message: dict[str, Any] = {
            "type": "assist_pipeline/run",
            "id": run_id,
            "start_stage": "stt",
            "end_stage": "intent",
            "input": {"sample_rate": SAMPLE_RATE_HZ},
        }
        for key, value in (
            ("pipeline", pipeline),
            ("conversation_id", conversation_id),
            ("device_id", device_id),
        ):
            if value is not None:
                run_message[key] = value
        await _send_json(websocket, run_message)

        started = await _receive_json(websocket)
        _require_result(started, run_id)

        handler_id: int | None = None
        run_started = False
        stt_started = False
        transcript: str | None = None
        intent: dict[str, Any] | None = None

        while True:
            message = await _receive_json(websocket)
            if message.get("type") != "event" or message.get("id") != run_id:
                raise AssistError("malformed Assist pipeline event")
            event = message.get("event")
            if not isinstance(event, dict):
                raise AssistError("malformed Assist pipeline event")
            event_type = event.get("type")
            data = event.get("data", {})
            if not isinstance(event_type, str) or not isinstance(data, dict):
                raise AssistError("malformed Assist pipeline event")

            if event_type == "error":
                _raise_pipeline_error(data)
            if event_type == "run-start":
                if run_started:
                    raise AssistError("duplicate run-start event")
                run_started = True
                parsed_handler_id = _handler_id(data)
                if parsed_handler_id is not None:
                    handler_id = parsed_handler_id
            elif event_type == "stt-start":
                if not run_started or stt_started:
                    raise AssistError("malformed stt-start event")
                stt_started = True
                if handler_id is None:
                    handler_id = _handler_id(data)
                if handler_id is None:
                    raise AssistError("missing stt binary handler ID")
                for offset in range(0, len(pcm), chunk_size):
                    await websocket.send(bytes((handler_id,)) + pcm[offset : offset + chunk_size])
                await websocket.send(bytes((handler_id,)))
            elif event_type == "stt-end":
                output = data.get("stt_output")
                if not stt_started or not isinstance(output, dict):
                    raise AssistError("malformed stt-end event")
                text = output.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise AssistError("malformed STT output")
                transcript = text
            elif event_type == "intent-end":
                output = data.get("intent_output")
                if transcript is None or not isinstance(output, dict):
                    raise AssistError("malformed intent-end event")
                intent = output
            elif event_type == "run-end":
                if not run_started or not stt_started or transcript is None or intent is None:
                    raise AssistError("Assist pipeline ended before completion")
                return {"status": "completed", "text": transcript, "intent": intent}
            elif event_type not in {
                "intent-start",
                "intent-progress",
                "stt-vad-start",
                "stt-vad-end",
            }:
                raise AssistError(f"unexpected Assist pipeline event: {event_type}")


async def _authenticate(websocket: AssistWebSocket, token: str) -> None:
    message = await _receive_json(websocket)
    if message.get("type") != "auth_required":
        raise AssistError("malformed Home Assistant auth challenge")
    await _send_json(websocket, {"type": "auth", "access_token": token})
    response = await _receive_json(websocket)
    if response.get("type") != "auth_ok":
        raise AssistError("Home Assistant authentication failed")


async def _send_json(websocket: AssistWebSocket, message: dict[str, Any]) -> None:
    await websocket.send(json.dumps(message, separators=(",", ":")))


async def _receive_json(websocket: AssistWebSocket) -> dict[str, Any]:
    message = await websocket.recv()
    if not isinstance(message, str):
        raise AssistError("malformed Assist message: expected JSON text")
    try:
        parsed = json.loads(message)
    except json.JSONDecodeError as exc:
        raise AssistError("malformed Assist message: invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise AssistError("malformed Assist message: expected an object")
    return parsed


def _require_result(message: dict[str, Any], run_id: int) -> None:
    if (
        message.get("type") != "result"
        or message.get("id") != run_id
        or "result" not in message
    ):
        raise AssistError("malformed Assist pipeline start result")
    if message.get("success") is not True:
        error = message.get("error")
        raise AssistError(f"Assist pipeline start failed: {error!r}")


def _handler_id(data: dict[str, Any]) -> int | None:
    runner_data = data.get("runner_data")
    if not isinstance(runner_data, dict):
        return None
    value = runner_data.get("stt_binary_handler_id")
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        if value is not None:
            raise AssistError("malformed stt binary handler ID")
        return None
    return value


def _raise_pipeline_error(data: dict[str, Any]) -> None:
    code = data.get("code")
    message = data.get("message")
    if not isinstance(code, str) or not code:
        raise AssistError("malformed Assist pipeline error")
    detail = message if isinstance(message, str) and message else ""
    raise AssistError(f"Assist pipeline error: {code}{': ' + detail if detail else ''}")


def _default_connect() -> Connect:
    from websockets.asyncio.client import connect

    return connect
