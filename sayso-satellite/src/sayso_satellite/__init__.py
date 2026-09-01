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

__all__ = [
    "BYTES_PER_SAMPLE",
    "CHANNELS",
    "DEFAULT_PRE_ROLL_MS",
    "FixtureMicSource",
    "PushToTalkCapture",
    "SAMPLE_RATE_HZ",
    "__version__",
    "expected_pcm_byte_length",
    "pcm_duration_ms",
]
__version__ = "0.1.0"
