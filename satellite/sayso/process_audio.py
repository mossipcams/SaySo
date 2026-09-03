"""Inject LiveKit wake detection without replacing upstream audio processing."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import numpy as np

from .wake.livekit import LiveKitWakeWordProvider

_LOGGER = logging.getLogger(__name__)


class _WakePhrase:
    def __init__(self, phrase: str) -> None:
        self.wake_word = phrase
        self.id = "sayso"


class _WakeStream:
    def __init__(self, stream: Any, state: Any, provider: LiveKitWakeWordProvider) -> None:
        self._stream = stream
        self._state = state
        self._provider = provider

    def record(self, numframes: int) -> Any:
        raw = self._stream.record(numframes)
        satellite = self._state.satellite
        if satellite is None or self._state.muted:
            return raw
        if getattr(satellite, "_pipeline_active", False):
            self._provider.reset()
            return raw

        try:
            primary = raw[:, 0] if raw.ndim > 1 else raw
            pcm = (np.clip(primary, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            detection = self._provider.process_pcm(pcm)
            if detection is not None:
                satellite.wakeup(_WakePhrase(detection.phrase))
        except Exception:  # Keep upstream audio alive if wake inference fails.
            _LOGGER.exception("Unexpected error detecting the SaySo wake word")

        return raw


class _WakeMicrophone:
    def __init__(self, mic: Any, state: Any, provider: LiveKitWakeWordProvider) -> None:
        self._mic = mic
        self._state = state
        self._provider = provider
        self.name = mic.name

    @contextmanager
    def recorder(self, *args: Any, **kwargs: Any) -> Iterator[_WakeStream]:
        with self._mic.recorder(*args, **kwargs) as stream:
            yield _WakeStream(stream, self._state, self._provider)


def make_process_audio(
    upstream_process_audio: Callable[[Any, Any, int], None],
    provider: LiveKitWakeWordProvider,
) -> Callable[[Any, Any, int], None]:
    """Wrap the upstream microphone while delegating its complete audio loop."""

    def process_audio(state: Any, mic: Any, block_size: int) -> None:
        provider.start()
        try:
            upstream_process_audio(
                state,
                _WakeMicrophone(mic, state, provider),
                block_size,
            )
        except SystemExit as err:
            code = err.code if isinstance(err.code, int) and err.code > 0 else 1
            os._exit(code)
        finally:
            provider.shutdown()

    return process_audio
