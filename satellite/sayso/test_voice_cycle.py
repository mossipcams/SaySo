"""Regression coverage for wake recovery after a terminal TTS error."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from satellite.sayso.launcher import _configure_mpv
from satellite.sayso.process_audio import make_process_audio
from satellite.sayso.wake.detection import Detection


def test_tts_error_does_not_make_the_next_wake_inaudible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLibMpvPlayer:
        def __init__(self) -> None:
            self._done_callback = None

        def _on_end_file(self, event) -> None:
            if event.data.reason == 0 and self._done_callback:
                callback = self._done_callback
                self._done_callback = None
                callback()

    package = ModuleType("linux_voice_assistant")
    package.__path__ = []  # type: ignore[attr-defined]
    player_package = ModuleType("linux_voice_assistant.player")
    player_package.__path__ = []  # type: ignore[attr-defined]
    libmpv = ModuleType("linux_voice_assistant.player.libmpv")
    libmpv.mpv = SimpleNamespace(MPV=Mock())  # type: ignore[attr-defined]
    libmpv.LibMpvPlayer = FakeLibMpvPlayer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "linux_voice_assistant", package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.player", player_package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.player.libmpv", libmpv)
    _configure_mpv()

    player = FakeLibMpvPlayer()
    satellite = SimpleNamespace(_pipeline_active=False)
    satellite.wakeup = MagicMock(
        side_effect=lambda _phrase: setattr(satellite, "_pipeline_active", True)
    )
    state = SimpleNamespace(satellite=satellite, muted=False)
    provider = MagicMock()
    provider.process_pcm.side_effect = [
        Detection("hey sayso", 0.9, 1.0),
        Detection("hey sayso", 0.9, 2.0),
    ]
    raw = np.zeros((2, 1), dtype=np.float32)
    mic_input = MagicMock()
    mic_input.record.return_value = raw

    @contextmanager
    def recorder(**_kwargs):
        yield mic_input

    mic = SimpleNamespace(name="default", recorder=recorder)
    tts_finished = Mock(
        side_effect=lambda: setattr(satellite, "_pipeline_active", False)
    )

    def upstream_process_audio(_state, wrapped_mic, block_size):
        with wrapped_mic.recorder(blocksize=block_size) as stream:
            stream.record(block_size)
            assert satellite._pipeline_active

            player._done_callback = tts_finished
            player._on_end_file(SimpleNamespace(data=SimpleNamespace(reason=4)))
            assert not satellite._pipeline_active

            stream.record(block_size)

    make_process_audio(upstream_process_audio, provider)(state, mic, 2)

    assert satellite._pipeline_active
    assert satellite.wakeup.call_count == 2
    tts_finished.assert_called_once_with()
