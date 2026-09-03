"""The SaySo wake overlay must leave upstream audio processing intact."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

import sayso.process_audio as process_audio_module
from sayso.process_audio import make_process_audio
from sayso.wake.detection import Detection


def test_livekit_wake_wraps_upstream_microphone_without_replacing_audio_loop() -> None:
    raw = np.array([[0.25], [-0.25]], dtype=np.float32)
    mic_input = MagicMock()
    mic_input.record.return_value = raw

    @contextmanager
    def recorder(**_kwargs):
        yield mic_input

    mic = SimpleNamespace(name="default", recorder=recorder)
    satellite = SimpleNamespace(_pipeline_active=False, wakeup=MagicMock())
    state = SimpleNamespace(satellite=satellite, muted=False)
    provider = MagicMock()
    provider.process_pcm.return_value = Detection("hey sayso", 0.9, 1.0)

    def upstream_process_audio(upstream_state, upstream_mic, block_size):
        assert upstream_state is state
        assert upstream_mic.name == mic.name
        with upstream_mic.recorder(samplerate=16000, channels=1, blocksize=block_size) as stream:
            assert stream.record(block_size) is raw

    process_audio = make_process_audio(upstream_process_audio, provider)
    process_audio(state, mic, 2)

    provider.start.assert_called_once_with()
    provider.process_pcm.assert_called_once_with(np.array([8191, -8191], dtype="<i2").tobytes())
    satellite.wakeup.assert_called_once()
    assert satellite.wakeup.call_args.args[0].wake_word == "hey sayso"
    provider.shutdown.assert_called_once_with()


def test_upstream_system_exit_terminates_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminate = Mock()
    monkeypatch.setattr(process_audio_module.os, "_exit", terminate)
    provider = MagicMock()
    upstream_process_audio = Mock(side_effect=SystemExit(1))
    process_audio = make_process_audio(upstream_process_audio, provider)

    process_audio(SimpleNamespace(), SimpleNamespace(name="default"), 512)

    terminate.assert_called_once_with(1)
    provider.shutdown.assert_called_once_with()
