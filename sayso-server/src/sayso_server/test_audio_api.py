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
from sayso_server.stt import FakeSpeechToTextRuntime
from sayso_server.text_api import TextController, create_text_handler

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
    controller: RecordingAudioController | None = None,
    *,
    satellite_id: str = "macbook",
    area_id: str = "area_living_room",
    with_graph: bool = True,
    stt_runtime: FakeSpeechToTextRuntime | None = None,
    text_controller: TextController | None = None,
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
        stt_runtime=stt_runtime,
        text_controller=text_controller,
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
async def test_voice_pipeline_emits_audio_input_type_telemetry() -> None:
    from io import StringIO

    from sayso_server.conversation import ConversationStore
    from sayso_server.ha_client import FakeHaClient
    from sayso_server.runtime import FakeModelRuntime
    from sayso_server.telemetry import InteractionTelemetryRecord, JsonlTelemetrySink
    from sayso_server.text_api import OrchestratorTextController

    graph_store = HomeGraphStore()
    graph_store.replace_snapshot(_load_graph())
    runtime = FakeModelRuntime()
    runtime.load()
    sink_buffer = StringIO()
    sink = JsonlTelemetrySink(sink_buffer)
    text_controller = OrchestratorTextController(
        runtime=runtime,
        ha_client=FakeHaClient(),
        graph_store=graph_store,
        conversation_store=ConversationStore(ttl_seconds=300.0),
        telemetry_sink=sink,
    )
    transcript = "turn off the floor lamp"
    stt = FakeSpeechToTextRuntime(transcript=transcript)
    stt.load()
    handler = _build_handler(stt_runtime=stt, text_controller=text_controller)

    status, body = await _post_audio(
        handler,
        _audio_request(correlation_id="corr-audio-telemetry"),
    )
    assert status == 200
    assert body is not None

    lines = [line for line in sink_buffer.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    record = InteractionTelemetryRecord.model_validate(parsed)
    assert record.correlation_id == "corr-audio-telemetry"
    assert record.input_type == "audio"
    assert record.stages.stt_ms >= 0.0


@pytest.mark.asyncio
async def test_voice_pipeline_records_stt_stage_timing() -> None:
    from io import StringIO
    from unittest.mock import patch

    from sayso_server import audio_api
    from sayso_server.conversation import ConversationStore
    from sayso_server.ha_client import FakeHaClient
    from sayso_server.runtime import FakeModelRuntime
    from sayso_server.telemetry import InteractionTelemetryRecord, JsonlTelemetrySink
    from sayso_server.text_api import OrchestratorTextController

    class SlowLoadFakeSpeechToTextRuntime(FakeSpeechToTextRuntime):
        def load(self) -> None:
            audio_api.time.monotonic()
            super().load()

    graph_store = HomeGraphStore()
    graph_store.replace_snapshot(_load_graph())
    runtime = FakeModelRuntime()
    runtime.load()
    sink_buffer = StringIO()
    sink = JsonlTelemetrySink(sink_buffer)
    text_controller = OrchestratorTextController(
        runtime=runtime,
        ha_client=FakeHaClient(),
        graph_store=graph_store,
        conversation_store=ConversationStore(ttl_seconds=300.0),
        telemetry_sink=sink,
    )
    transcript = "turn off the floor lamp"
    stt = SlowLoadFakeSpeechToTextRuntime(transcript=transcript)
    stt.load()
    handler = _build_handler(stt_runtime=stt, text_controller=text_controller)
    monotonic_values = iter([100.0, 100.05, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6])

    with patch(
        "sayso_server.audio_api.time.monotonic",
        side_effect=lambda: next(monotonic_values),
    ):
        status, body = await _post_audio(
            handler,
            _audio_request(correlation_id="corr-audio-stt-timing"),
        )

    assert status == 200
    assert body is not None
    parsed = json.loads(sink_buffer.getvalue().strip())
    record = InteractionTelemetryRecord.model_validate(parsed)
    assert record.stages.stt_ms == pytest.approx(100.0)


class UnavailableSpeechToTextRuntime(FakeSpeechToTextRuntime):
    """STT stand-in that simulates missing mlx-whisper at load time."""

    def load(self) -> None:
        msg = "mlx-whisper is required for MLX STT but is not installed"
        raise RuntimeError(msg)


class UnavailableTranscribeSpeechToTextRuntime(FakeSpeechToTextRuntime):
    """STT stand-in that simulates mlx-whisper failure during transcribe."""

    def transcribe(
        self,
        pcm: bytes,
        *,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
    ):
        del pcm, sample_rate_hz
        msg = "mlx-whisper is required for MLX STT but is not installed"
        raise RuntimeError(msg)


class DtypeMismatchTranscribeSpeechToTextRuntime(FakeSpeechToTextRuntime):
    """STT stand-in that simulates mlx-whisper dtype failure during transcribe."""

    def transcribe(
        self,
        pcm: bytes,
        *,
        sample_rate_hz: int = SAMPLE_RATE_HZ,
    ):
        del pcm, sample_rate_hz
        msg = "audio_features has an incorrect dtype: mlx.core.float32"
        raise TypeError(msg)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stt_factory",
    [
        UnavailableSpeechToTextRuntime,
        UnavailableTranscribeSpeechToTextRuntime,
        DtypeMismatchTranscribeSpeechToTextRuntime,
    ],
)
async def test_stt_unavailable_returns_classified_response_not_500(
    stt_factory: type[FakeSpeechToTextRuntime],
) -> None:
    from sayso_server.test_text_api import RecordingTextController

    stt = stt_factory()
    text_controller = RecordingTextController()
    handler = _build_handler(stt_runtime=stt, text_controller=text_controller)

    status, body = await _post_audio(
        handler,
        _audio_request(correlation_id="corr-stt-unavailable"),
    )

    assert status != 500
    assert body is not None
    if body["type"] == "text_response":
        assert body["payload"]["category"] == "no_action"
        reason = body["payload"].get("reason") or ""
        assert (
            "unavailable" in reason.lower()
            or "mlx-whisper" in reason.lower()
            or "dtype" in reason.lower()
        )
    else:
        assert body["type"] == "error"
        assert body["payload"]["code"] == "stt_unavailable"
    assert text_controller.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("transcript", ["", "   ", "\n\t"])
async def test_blank_transcript_returns_classified_no_action_not_500(
    transcript: str,
) -> None:
    from sayso_server.test_text_api import RecordingTextController

    stt = FakeSpeechToTextRuntime(transcript=transcript)
    stt.load()
    text_controller = RecordingTextController()
    handler = _build_handler(stt_runtime=stt, text_controller=text_controller)

    status, body = await _post_audio(
        handler,
        _audio_request(correlation_id="corr-empty-transcript"),
    )

    assert status != 500
    assert status == 200
    assert body is not None
    assert body["type"] == "text_response"
    assert body["payload"]["category"] == "no_action"
    assert "empty transcript" in (body["payload"].get("reason") or "").lower()
    assert text_controller.calls == []


@pytest.mark.asyncio
async def test_recorded_fixture_transcribes_then_runs_text_controller() -> None:
    from sayso_server.test_text_api import RecordingTextController

    transcript = "turn off the floor lamp"
    stt = FakeSpeechToTextRuntime(transcript=transcript)
    stt.load()
    text_controller = RecordingTextController()
    handler = _build_handler(stt_runtime=stt, text_controller=text_controller)
    original_pcm = RECORDED_PCM.read_bytes()
    status, body = await _post_audio(
        handler,
        _audio_request(correlation_id="voice-pipeline-1", sequence=3),
    )
    assert status == 200
    assert body is not None
    assert body["version"] == API_VERSION
    assert body["type"] == "text_response"
    assert body["correlation_id"] == "voice-pipeline-1"
    assert body["payload"]["category"] == "completed"
    assert stt.transcribe_calls == [original_pcm]
    assert len(text_controller.calls) == 1
    call = text_controller.calls[0]
    assert call == {
        "satellite_id": "macbook",
        "area_id": "area_living_room",
        "text": transcript,
        "correlation_id": "voice-pipeline-1",
        "input_type": "audio",
        "stt_ms": call["stt_ms"],
    }
    assert call["stt_ms"] >= 0.0


@pytest.mark.asyncio
async def test_identical_transcript_via_text_and_audio_produces_same_outcome() -> None:
    from sayso_server.ha_client import FakeHaClient
    from sayso_server.runtime import FakeModelRuntime
    from sayso_server.text_api import OrchestratorTextController

    graph_store = HomeGraphStore()
    graph_store.replace_snapshot(_load_graph())
    runtime = FakeModelRuntime()
    runtime.load()
    text_controller = OrchestratorTextController(
        runtime=runtime,
        ha_client=FakeHaClient(),
        graph_store=graph_store,
    )
    transcript = "turn off the floor lamp"
    stt = FakeSpeechToTextRuntime(transcript=transcript)
    stt.load()

    registry = SatelliteRegistry()
    registry.register("macbook", "area_living_room")
    text_handler = create_text_handler(
        token="secret-token",
        satellite_registry=registry,
        graph_store=graph_store,
        text_controller=text_controller,
    )
    audio_handler = create_audio_handler(
        token="secret-token",
        satellite_registry=registry,
        graph_store=graph_store,
        stt_runtime=stt,
        text_controller=text_controller,
    )

    text_status, text_body = await _post_text_like(text_handler, transcript, "parity-text")
    audio_status, audio_body = await _post_audio(
        audio_handler,
        _audio_request(correlation_id="parity-audio"),
    )

    assert text_status == 200
    assert audio_status == 200
    assert text_body is not None
    assert audio_body is not None
    assert text_body["type"] == "text_response"
    assert audio_body["type"] == "text_response"
    assert text_body["payload"]["category"] == audio_body["payload"]["category"]
    assert text_body["payload"]["plan"] == audio_body["payload"]["plan"]


async def _post_text_like(handler, text: str, correlation_id: str) -> tuple[int, dict[str, object] | None]:
    body = {
        "version": API_VERSION,
        "type": "text",
        "correlation_id": correlation_id,
        "payload": {
            "satellite_id": "macbook",
            "text": text,
        },
    }
    request = MagicMock()
    header_values = {"Content-Type": "application/json", "Authorization": "Bearer secret-token"}
    request.headers = MagicMock()
    request.headers.get = lambda key, default=None: header_values.get(key, default)
    request.text = AsyncMock(return_value=json.dumps(body))
    response = await handler(request)
    if response.body is None:
        return response.status, None
    payload = json.loads(response.text)
    return response.status, payload


def test_create_aiohttp_app_registers_audio_route() -> None:
    from sayso_server.app import create_aiohttp_app

    app = create_aiohttp_app("secret-token")
    paths = {route.resource.canonical for route in app.router.routes()}
    assert AUDIO_PATH in paths
