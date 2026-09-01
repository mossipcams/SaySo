"""HTTP contract tests for POST /api/v1/audio."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from sayso_server.api import API_VERSION
from sayso_server.audio_api import (
    CHANNELS,
    PCM_ENCODING,
    SAMPLE_RATE_HZ,
    create_audio_handler,
    expected_pcm_byte_length,
)
from sayso_server.const import AUDIO_PATH
from sayso_server.graph_store import HomeGraphStore
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.satellites import SatelliteRegistry

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"
RECORDED_PCM = FIXTURES / "audio_pcm16_mono_16k.bin"
RECORDED_DURATION_MS = 160


def _load_graph() -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    return HomeGraphSnapshot.model_validate(data)


def _audio_request(
    *,
    correlation_id: str = "corr-audio-1",
    satellite_id: str = "macbook",
    sequence: int = 0,
    duration_ms: int = RECORDED_DURATION_MS,
    pcm: bytes | None = None,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    channels: int = CHANNELS,
    encoding: str = PCM_ENCODING,
) -> dict[str, object]:
    payload_pcm = pcm if pcm is not None else RECORDED_PCM.read_bytes()
    return {
        "version": API_VERSION,
        "type": "audio",
        "correlation_id": correlation_id,
        "payload": {
            "satellite_id": satellite_id,
            "sequence": sequence,
            "duration_ms": duration_ms,
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
            "encoding": encoding,
            "pcm_base64": base64.b64encode(payload_pcm).decode("ascii"),
        },
    }


class RecordingAudioController:
    """Spy stand-in for the audio transport controller."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def handle(
        self,
        *,
        satellite_id: str,
        area_id: str,
        sequence: int,
        duration_ms: int,
        pcm: bytes,
        correlation_id: str,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "satellite_id": satellite_id,
                "area_id": area_id,
                "sequence": sequence,
                "duration_ms": duration_ms,
                "pcm": pcm,
                "correlation_id": correlation_id,
            },
        )
        return {
            "sequence": sequence,
            "duration_ms": duration_ms,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": CHANNELS,
            "encoding": PCM_ENCODING,
            "byte_length": len(pcm),
            "pcm_base64": base64.b64encode(pcm).decode("ascii"),
        }


def _build_handler(
    controller: RecordingAudioController,
    *,
    satellite_id: str = "macbook",
    area_id: str = "area_living_room",
    with_graph: bool = True,
):
    registry = SatelliteRegistry()
    registry.register(satellite_id, area_id)
    graph_store = HomeGraphStore()
    if with_graph:
        graph_store.replace_snapshot(_load_graph())
    return create_audio_handler(
        token="secret-token",
        satellite_registry=registry,
        graph_store=graph_store,
        audio_controller=controller,
    )


async def _post_audio(
    handler,
    body: dict[str, object] | str,
    *,
    token: str | None = "secret-token",
) -> tuple[int, dict[str, object] | None]:
    request = MagicMock()
    header_values: dict[str, str] = {"Content-Type": "application/json"}
    if token is not None:
        header_values["Authorization"] = f"Bearer {token}"
    request.headers = MagicMock()
    request.headers.get = lambda key, default=None: header_values.get(key, default)
    request.text = AsyncMock(return_value=body if isinstance(body, str) else json.dumps(body))
    response = await handler(request)
    if response.body is None:
        return response.status, None
    payload = json.loads(response.text)
    return response.status, payload


def test_audio_path_constant() -> None:
    assert AUDIO_PATH == "/api/v1/audio"


def test_recorded_fixture_matches_pcm16_mono_16k_framing() -> None:
    pcm = RECORDED_PCM.read_bytes()
    assert len(pcm) == expected_pcm_byte_length(
        duration_ms=RECORDED_DURATION_MS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        channels=CHANNELS,
    )
    assert len(pcm) % 2 == 0


@pytest.mark.asyncio
async def test_missing_auth_returns_401() -> None:
    controller = RecordingAudioController()
    handler = _build_handler(controller)
    status, body = await _post_audio(handler, _audio_request(), token=None)
    assert status == 401
    assert body is None
    assert controller.calls == []


@pytest.mark.asyncio
async def test_invalid_json_returns_400() -> None:
    controller = RecordingAudioController()
    handler = _build_handler(controller)
    status, body = await _post_audio(handler, "{not-json")
    assert status == 400
    assert body is not None
    assert body["type"] == "error"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_invalid_envelope_returns_400() -> None:
    controller = RecordingAudioController()
    handler = _build_handler(controller)
    status, body = await _post_audio(
        handler,
        {"version": 2, "type": "audio", "correlation_id": "x", "payload": {}},
    )
    assert status == 400
    assert body is not None
    assert body["type"] == "error"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_unknown_satellite_never_reaches_controller() -> None:
    controller = RecordingAudioController()
    handler = _build_handler(controller)
    status, body = await _post_audio(handler, _audio_request(satellite_id="unknown-sat"))
    assert status == 400
    assert body is not None
    assert body["payload"]["code"] == "unknown_satellite"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_missing_graph_never_reaches_controller() -> None:
    controller = RecordingAudioController()
    handler = _build_handler(controller, with_graph=False)
    status, body = await _post_audio(handler, _audio_request())
    assert status == 400
    assert body is not None
    assert body["payload"]["code"] == "no_graph"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_wrong_sample_rate_is_rejected() -> None:
    controller = RecordingAudioController()
    handler = _build_handler(controller)
    status, body = await _post_audio(
        handler,
        _audio_request(sample_rate_hz=48_000),
    )
    assert status == 400
    assert body is not None
    assert body["payload"]["code"] == "invalid_sample_rate"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_wrong_channels_is_rejected() -> None:
    controller = RecordingAudioController()
    handler = _build_handler(controller)
    status, body = await _post_audio(
        handler,
        _audio_request(channels=2),
    )
    assert status == 400
    assert body is not None
    assert body["payload"]["code"] == "invalid_channels"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_corrupt_base64_is_rejected() -> None:
    controller = RecordingAudioController()
    handler = _build_handler(controller)
    body = _audio_request()
    payload = body["payload"]
    assert isinstance(payload, dict)
    payload["pcm_base64"] = "%%%not-base64%%%"
    status, response = await _post_audio(handler, body)
    assert status == 400
    assert response is not None
    assert response["payload"]["code"] == "invalid_pcm"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_duration_mismatch_is_rejected() -> None:
    controller = RecordingAudioController()
    handler = _build_handler(controller)
    status, body = await _post_audio(
        handler,
        _audio_request(duration_ms=RECORDED_DURATION_MS + 10),
    )
    assert status == 400
    assert body is not None
    assert body["payload"]["code"] == "duration_mismatch"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_odd_pcm_byte_length_is_rejected() -> None:
    controller = RecordingAudioController()
    handler = _build_handler(controller)
    pcm = RECORDED_PCM.read_bytes()[:-1]
    duration_ms = len(pcm) * 1000 // (SAMPLE_RATE_HZ * CHANNELS * 2)
    status, body = await _post_audio(
        handler,
        _audio_request(duration_ms=duration_ms, pcm=pcm),
    )
    assert status == 400
    assert body is not None
    assert body["payload"]["code"] == "invalid_pcm"
    assert controller.calls == []


@pytest.mark.asyncio
async def test_recorded_fixture_round_trips_with_sequence_metadata() -> None:
    controller = RecordingAudioController()
    handler = _build_handler(controller)
    original_pcm = RECORDED_PCM.read_bytes()
    status, body = await _post_audio(
        handler,
        _audio_request(correlation_id="round-trip-1", sequence=3),
    )
    assert status == 200
    assert body is not None
    assert body["version"] == API_VERSION
    assert body["type"] == "audio_response"
    assert body["correlation_id"] == "round-trip-1"
    payload = body["payload"]
    assert payload["sequence"] == 3
    assert payload["duration_ms"] == RECORDED_DURATION_MS
    assert payload["sample_rate_hz"] == SAMPLE_RATE_HZ
    assert payload["channels"] == CHANNELS
    assert payload["encoding"] == PCM_ENCODING
    assert payload["byte_length"] == len(original_pcm)
    assert base64.b64decode(payload["pcm_base64"]) == original_pcm
    assert controller.calls == [
        {
            "satellite_id": "macbook",
            "area_id": "area_living_room",
            "sequence": 3,
            "duration_ms": RECORDED_DURATION_MS,
            "pcm": original_pcm,
            "correlation_id": "round-trip-1",
        },
    ]


def test_create_aiohttp_app_registers_audio_route() -> None:
    from sayso_server.app import create_aiohttp_app

    app = create_aiohttp_app("secret-token")
    paths = {route.resource.canonical for route in app.router.routes()}
    assert AUDIO_PATH in paths
