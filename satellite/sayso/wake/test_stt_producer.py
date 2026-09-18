"""Regression: overlay ring is the sole STT producer when builtin wake is disabled."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from satellite.sayso.test_launcher import _run_patched_wake_stop_control_flow
from satellite.sayso.wake.hook import SaySoExternalWakeHook
from satellite.sayso.wake.test_handoff import _RecordingSatellite


def test_one_capture_block_yields_one_handle_audio_send() -> None:
    """Overlay feed_pcm plus exec'd patch-0002 hunk must not double-send a block."""
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    satellite = _RecordingSatellite()
    state = SimpleNamespace(satellite=satellite, disable_builtin_wake_word=True)
    hook.bind_satellite(lambda: satellite)
    hook.start()
    try:
        block = np.arange(1024, dtype="<i2")
        pcm = block.tobytes()
        hook.feed_pcm(state, pcm)
        _run_patched_wake_stop_control_flow(
            disable_builtin_wake_word=True,
            wake_activated=False,
            stop_detected=False,
            satellite=satellite,
            audio_chunk=pcm,
        )
    finally:
        hook.shutdown()

    assert len(satellite.chunks) == 1
    assert satellite.chunks[0] == pcm
    assert np.frombuffer(satellite.chunks[0], dtype="<i2").size == 1024
