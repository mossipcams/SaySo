"""Native-rate capture and the external wake hook must coexist.

Upstream LVA hardcodes a 16 kHz recorder; the overlay forces the native device
rate and resamples once. The wake overlay must still use LVA external wake hooks
and never wrap record() for detection.
"""

from __future__ import annotations

import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

import sayso.process_audio as process_audio_module
from sayso.process_audio import (
    TARGET_RATE,
    _ResamplingRecorder,
    install_native_rate_capture,
    install_wake_audio_path,
)
from sayso.wake.detection import Detection
from sayso.wake.hook import SaySoExternalWakeHook
from sayso.wake.livekit import HOP_SAMPLES, WINDOW_SAMPLES

def _install_external_wake_module(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    external_wake = ModuleType("linux_voice_assistant.external_wake")
    external_wake.set_provider = MagicMock()  # type: ignore[attr-defined]
    package = ModuleType("linux_voice_assistant")
    package.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "linux_voice_assistant", package)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.external_wake", external_wake)
    return external_wake.set_provider  # type: ignore[attr-defined]


def test_install_wake_audio_path_registers_external_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_provider = _install_external_wake_module(monkeypatch)
    hook = SaySoExternalWakeHook(MagicMock(available=True, predict_window=MagicMock(return_value=None)))
    lva_main = SimpleNamespace(run=Mock())
    install_wake_audio_path(lva_main, hook)
    set_provider.assert_called_once_with(hook.feed_pcm)


def test_external_wake_hook_forwards_detection_to_satellite_wakeup() -> None:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = Detection("hey sayso", 0.9, 1.0)
    hook = SaySoExternalWakeHook(provider)
    satellite = SimpleNamespace(_pipeline_active=False, wakeup=MagicMock())
    state = SimpleNamespace(satellite=satellite)
    hook.start()
    try:
        chunk = np.zeros(512, dtype="<i2").tobytes()
        samples_needed = WINDOW_SAMPLES + HOP_SAMPLES
        fed = 0
        while fed < samples_needed:
            hook.feed_pcm(state, chunk)
            fed += 512
            time.sleep(0.01)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not satellite.wakeup.called:
            time.sleep(0.01)
    finally:
        hook.shutdown()

    satellite.wakeup.assert_called()
    assert satellite.wakeup.call_args.args[0].wake_word == "hey sayso"


def test_external_wake_hook_does_not_reset_while_suspended() -> None:
    provider = MagicMock(available=True)
    provider.predict_window.return_value = None
    hook = SaySoExternalWakeHook(provider)
    hook.suspend()
    pcm = np.zeros(512, dtype="<i2").tobytes()
    hook.feed_pcm(SimpleNamespace(satellite=None), pcm)
    provider.predict_window.assert_not_called()


def test_install_wake_audio_path_exits_on_upstream_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_external_wake_module(monkeypatch)
    terminate = Mock()
    monkeypatch.setattr(process_audio_module.os, "_exit", terminate)
    hook = SaySoExternalWakeHook(MagicMock(available=True, predict_window=MagicMock(return_value=None)))
    lva_main = SimpleNamespace(run=Mock(side_effect=SystemExit(1)))
    install_wake_audio_path(lva_main, hook)

    lva_main.run()

    terminate.assert_called_once_with(1)


class _FakeRecorder:
    def __init__(self, rate: int, channels: int, block: int) -> None:
        self.rate = rate
        self.channels = channels
        self.block = block
        self.requested: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def record(self, numframes: int) -> Any:
        self.requested.append(numframes)
        return np.zeros((numframes, self.channels), dtype=np.float32)


def test_recorder_opens_at_native_rate_and_outputs_16k() -> None:
    inner = _FakeRecorder(44100, 1, 1024)
    rec = _ResamplingRecorder(inner, native_rate=44100, channels=1, gain=1.0)
    # Upstream calls mic_in.record(block_size) expecting block_size frames of
    # 16 kHz audio, so 160 frames out means reading 441 native frames in.
    out = rec.record(160)
    assert out.shape == (160, 1)
    assert inner.requested and inner.requested[0] >= 441


def test_recorder_returns_exactly_the_frames_requested_every_call() -> None:
    """A short block would desynchronise anything upstream that reshapes it."""
    inner = _FakeRecorder(44100, 1, 1024)
    rec = _ResamplingRecorder(inner, native_rate=44100, channels=1, gain=1.0)
    for _ in range(50):
        assert rec.record(160).shape == (160, 1)


def test_recorder_reads_native_frames_not_output_frames() -> None:
    """Requesting output frames 1:1 from a 44.1 kHz device starves the pipeline.

    160 output frames is 10 ms of audio; 160 *native* frames is 3.6 ms. Reading
    the latter would hand upstream roughly a third of the audio it asked for.
    """
    inner = _FakeRecorder(44100, 1, 1024)
    rec = _ResamplingRecorder(inner, native_rate=44100, channels=1, gain=1.0)
    rec.record(160)
    # 441 native frames per 160 output frames, so never fewer than 441.
    assert sum(inner.requested) >= 441


def test_recorder_passthrough_when_already_16k() -> None:
    inner = _FakeRecorder(16000, 1, 1024)
    rec = _ResamplingRecorder(inner, native_rate=16000, channels=1, gain=1.0)
    out = rec.record(320)
    assert out.shape == (320, 1)


def test_fixed_gain_is_applied_and_clipping_counted() -> None:
    class _LoudRecorder(_FakeRecorder):
        def record(self, numframes: int) -> Any:
            return np.full((numframes, 1), 0.9, dtype=np.float32)

    rec = _ResamplingRecorder(
        _LoudRecorder(16000, 1, 320), native_rate=16000, channels=1, gain=4.0
    )
    out = rec.record(320)
    assert float(np.max(out)) <= 1.0
    assert rec.clip_events == 1


def test_gain_is_not_applied_by_the_audio_server() -> None:
    """Gain is one deterministic multiply here, not a runtime state lookup."""
    rec = _ResamplingRecorder(
        _FakeRecorder(16000, 1, 320), native_rate=16000, channels=1, gain=2.0
    )
    # Replace the inner recorder's output with a known constant.
    rec._recorder.record = lambda n: np.full((n, 1), 0.25, dtype=np.float32)
    out = rec.record(320)
    assert np.allclose(out, 0.5)


def test_install_wraps_process_audio_and_forces_native_recorder_rate() -> None:
    recorded_rates: list[int] = []
    channels_seen: list[int] = []

    class _Mic:
        name = "snowball"

        def recorder(self, samplerate, channels, blocksize):
            recorded_rates.append(samplerate)
            channels_seen.append(channels)
            return _FakeRecorder(samplerate, channels, blocksize)

    def fake_process_audio(state, mic, blocksize):
        with mic.recorder(samplerate=16000, channels=state.channels, blocksize=blocksize) as mic_in:
            mic_in.record(blocksize)
        state.calls.append(blocksize)

    lva_main = ModuleType("linux_voice_assistant.__main__")
    lva_main.process_audio = fake_process_audio  # type: ignore[attr-defined]

    install_native_rate_capture(
        lva_main, capture_rate=44100, gain_db=0.0, channels=1
    )

    state = SimpleNamespace(channels=1, calls=[])
    lva_main.process_audio(state, _Mic(), 1024)  # type: ignore[attr-defined]

    assert recorded_rates == [44100]
    assert channels_seen == [1]
    assert state.calls == [1024]


def test_install_restores_original_recorder_after_run() -> None:
    class _Mic:
        name = "snowball"

        def __init__(self) -> None:
            self._orig = Mock(return_value=_FakeRecorder(16000, 1, 1024))

        def recorder(self, *args, **kwargs):
            return self._orig(*args, **kwargs)

    def fake_process_audio(state, mic, blocksize):
        return None

    lva_main = ModuleType("linux_voice_assistant.__main__")
    lva_main.process_audio = fake_process_audio  # type: ignore[attr-defined]
    install_native_rate_capture(lva_main, capture_rate=44100, gain_db=0.0, channels=1)

    mic = _Mic()
    original = mic.recorder
    lva_main.process_audio(SimpleNamespace(), mic, 1024)  # type: ignore[attr-defined]
    assert mic.recorder == original
