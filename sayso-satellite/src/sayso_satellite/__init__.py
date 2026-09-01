"""SaySo satellite package."""

from sayso_satellite.capture import (
    BYTES_PER_SAMPLE,
    CHANNELS,
    DEFAULT_PRE_ROLL_MS,
    FixtureMicSource,
    PushToTalkCapture,
    SAMPLE_RATE_HZ,
    expected_pcm_byte_length,
    pcm_duration_ms,
)
from sayso_satellite.client import DEFAULT_SATELLITE_ID, PCM_ENCODING, send_audio, send_text
from sayso_satellite.response import (
    EARCON_TOKEN,
    ResponseMode,
    render_response,
    render_text_response_payload,
)

__all__ = [
    "BYTES_PER_SAMPLE",
    "CHANNELS",
    "DEFAULT_PRE_ROLL_MS",
    "DEFAULT_SATELLITE_ID",
    "EARCON_TOKEN",
    "PCM_ENCODING",
    "FixtureMicSource",
    "PushToTalkCapture",
    "ResponseMode",
    "SAMPLE_RATE_HZ",
    "__version__",
    "expected_pcm_byte_length",
    "pcm_duration_ms",
    "render_response",
    "render_text_response_payload",
    "send_audio",
    "send_text",
]
__version__ = "0.1.0"
