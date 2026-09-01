"""Validated POST /api/v1/audio endpoint for 16 kHz mono PCM transport."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Literal, Protocol

from aiohttp import web
from pydantic import BaseModel, Field, ValidationError

from sayso_server.api import API_VERSION
from sayso_server.auth import bearer_token_valid
from sayso_server.const import AUDIO_PATH
from sayso_server.graph_store import HomeGraphStore
from sayso_server.satellites import SatelliteRegistry
from sayso_server.text_api import ErrorResponseEnvelope

SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
BYTES_PER_SAMPLE = 2
PCM_ENCODING = "pcm_s16le"


class AudioRequestPayload(BaseModel):
    satellite_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    duration_ms: int = Field(gt=0)
    sample_rate_hz: int
    channels: int
    encoding: str = Field(min_length=1)
    pcm_base64: str = Field(min_length=1)


class AudioRequestEnvelope(BaseModel):
    version: Literal[API_VERSION]
    type: Literal["audio"]
    correlation_id: str = Field(min_length=1)
    payload: AudioRequestPayload


class AudioResponsePayload(BaseModel):
    sequence: int
    duration_ms: int
    sample_rate_hz: int
    channels: int
    encoding: str
    byte_length: int
    pcm_base64: str


class AudioResponseEnvelope(BaseModel):
    version: Literal[API_VERSION]
    type: Literal["audio_response"]
    correlation_id: str = Field(min_length=1)
    payload: AudioResponsePayload


class AudioController(Protocol):
    def handle(
        self,
        *,
        satellite_id: str,
        area_id: str,
        sequence: int,
        duration_ms: int,
        pcm: bytes,
        correlation_id: str,
    ) -> dict[str, Any]: ...


class EchoAudioController:
    """Validate PCM framing and echo accepted audio back to the caller."""

    def handle(
        self,
        *,
        satellite_id: str,
        area_id: str,
        sequence: int,
        duration_ms: int,
        pcm: bytes,
        correlation_id: str,
    ) -> dict[str, Any]:
        del satellite_id, area_id, correlation_id
        return {
            "sequence": sequence,
            "duration_ms": duration_ms,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": CHANNELS,
            "encoding": PCM_ENCODING,
            "byte_length": len(pcm),
            "pcm_base64": base64.b64encode(pcm).decode("ascii"),
        }


def expected_pcm_byte_length(*, duration_ms: int, sample_rate_hz: int, channels: int) -> int:
    """Return the exact byte length for a PCM16 chunk."""

    return duration_ms * sample_rate_hz * channels * BYTES_PER_SAMPLE // 1000


def validate_pcm_payload(payload: AudioRequestPayload) -> tuple[bytes | None, str | None]:
    """Validate declared format and decode PCM bytes."""

    if payload.sample_rate_hz != SAMPLE_RATE_HZ:
        return None, "invalid_sample_rate"
    if payload.channels != CHANNELS:
        return None, "invalid_channels"
    if payload.encoding != PCM_ENCODING:
        return None, "invalid_encoding"
    try:
        pcm = base64.b64decode(payload.pcm_base64, validate=True)
    except (binascii.Error, ValueError):
        return None, "invalid_pcm"
    if len(pcm) == 0 or len(pcm) % BYTES_PER_SAMPLE != 0:
        return None, "invalid_pcm"
    expected = expected_pcm_byte_length(
        duration_ms=payload.duration_ms,
        sample_rate_hz=payload.sample_rate_hz,
        channels=payload.channels,
    )
    if len(pcm) != expected:
        return None, "duration_mismatch"
    return pcm, None


def create_audio_handler(
    *,
    token: str,
    satellite_registry: SatelliteRegistry,
    graph_store: HomeGraphStore,
    audio_controller: AudioController | None = None,
) -> web.RequestHandler:
    """Create the aiohttp handler for POST /api/v1/audio."""

    controller = audio_controller or EchoAudioController()

    async def audio(request: web.Request) -> web.Response:
        if not bearer_token_valid(
            authorization=request.headers.get("Authorization"),
            expected_token=token,
        ):
            return web.Response(status=401)

        raw = await request.text()
        try:
            data = json.loads(raw)
            envelope = AudioRequestEnvelope.model_validate(data)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError):
            correlation_id = _extract_correlation_id(raw)
            error = ErrorResponseEnvelope(
                version=API_VERSION,
                type="error",
                correlation_id=correlation_id,
                payload={"code": "invalid_request", "message": "invalid audio request envelope"},
            )
            return web.json_response(error.model_dump(mode="json"), status=400)

        area_id, error_code = satellite_registry.resolve_area_id(
            envelope.payload.satellite_id,
            snapshot=graph_store.snapshot,
        )
        if error_code is not None or area_id is None:
            error = ErrorResponseEnvelope(
                version=API_VERSION,
                type="error",
                correlation_id=envelope.correlation_id,
                payload={
                    "code": error_code or "invalid_satellite",
                    "message": "satellite context is invalid",
                },
            )
            return web.json_response(error.model_dump(mode="json"), status=400)

        pcm, pcm_error = validate_pcm_payload(envelope.payload)
        if pcm is None:
            error = ErrorResponseEnvelope(
                version=API_VERSION,
                type="error",
                correlation_id=envelope.correlation_id,
                payload={
                    "code": pcm_error or "invalid_pcm",
                    "message": "audio payload is invalid",
                },
            )
            return web.json_response(error.model_dump(mode="json"), status=400)

        payload = controller.handle(
            satellite_id=envelope.payload.satellite_id,
            area_id=area_id,
            sequence=envelope.payload.sequence,
            duration_ms=envelope.payload.duration_ms,
            pcm=pcm,
            correlation_id=envelope.correlation_id,
        )
        response = AudioResponseEnvelope(
            version=API_VERSION,
            type="audio_response",
            correlation_id=envelope.correlation_id,
            payload=AudioResponsePayload.model_validate(payload),
        )
        return web.json_response(response.model_dump(mode="json"))

    return audio


def _extract_correlation_id(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "invalid"
    correlation_id = parsed.get("correlation_id")
    if isinstance(correlation_id, str) and correlation_id:
        return correlation_id
    return "invalid"


__all__ = [
    "AudioController",
    "AudioRequestEnvelope",
    "AudioResponseEnvelope",
    "EchoAudioController",
    "PCM_ENCODING",
    "SAMPLE_RATE_HZ",
    "CHANNELS",
    "create_audio_handler",
    "expected_pcm_byte_length",
    "validate_pcm_payload",
]
