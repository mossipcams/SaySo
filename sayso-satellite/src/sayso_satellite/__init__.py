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
from sayso_satellite.client import DEFAULT_SATELLITE_ID, send_text

__all__ = [
    "BYTES_PER_SAMPLE",
    "CHANNELS",
    "DEFAULT_PRE_ROLL_MS",
    "DEFAULT_SATELLITE_ID",
    "FixtureMicSource",
    "PushToTalkCapture",
    "SAMPLE_RATE_HZ",
    "__version__",
    "expected_pcm_byte_length",
    "pcm_duration_ms",
    "send_text",
]
__version__ = "0.1.0"
