"""The SaySo wake overlay must use LVA external wake hooks, not record() wrapping."""

from __future__ import annotations

import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

import sayso.process_audio as process_audio_module
from sayso.process_audio import install_wake_audio_path
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
