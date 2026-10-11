
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .wake.capture import CaptureResampler, gain_scalar_from_db
from .wake.hook import SaySoExternalWakeHook, install_external_wake_hook

_LOGGER = logging.getLogger(__name__)

TARGET_RATE = 16000


@dataclass
class NativeClipTally:

    count: int = 0


class _ResamplingRecorder:

    def __init__(
        self,
        recorder: Any,
        *,
        native_rate: int,
        channels: int,
        gain: float,
        native_clip_tally: NativeClipTally | None = None,
        raw_sink: Callable[[bytes], None] | None = None,
    ) -> None:
        self._recorder = recorder
        self._native_rate = native_rate
        self._channels = channels
        self._gain = gain
        self._native_clip_tally = native_clip_tally
        self._raw_sink = raw_sink
        self._raw_resampler = (
            CaptureResampler(native_rate, TARGET_RATE)
            if raw_sink is not None and native_rate != TARGET_RATE
            else None
        )
        self._resamplers = (
            None
            if native_rate == TARGET_RATE
            else [CaptureResampler(native_rate, TARGET_RATE) for _ in range(max(1, channels))]
        )
        self._clip_events = 0
        self._pending: Any = None

    def __enter__(self) -> "_ResamplingRecorder":
        self._recorder.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._recorder.__exit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._recorder, name)

    @property
    def clip_events(self) -> int:
        return self._clip_events

    def record(self, numframes: int) -> Any:
        import numpy as np

        if self._resamplers is None:
            raw = self._recorder.record(numframes)
            if raw is None:
                return raw
            data = np.asarray(raw, dtype=np.float32)
            if self._raw_sink is not None:
                self._capture_raw(data)
            return self._apply_gain(data)

        native_frames = max(1, -(-numframes * self._native_rate // TARGET_RATE))
        while self._pending is None or self._pending.shape[0] < numframes:
            raw = self._recorder.record(native_frames)
            if raw is None:
                pending, self._pending = self._pending, None
                return pending
            data = np.asarray(raw, dtype=np.float32)
            if self._raw_sink is not None:
                self._capture_raw(data)
            block = self._resample(self._apply_gain(data))
            if block.shape[0] == 0:
                continue
            self._pending = (
                block if self._pending is None else np.concatenate((self._pending, block))
            )

        out = self._pending[:numframes]
        self._pending = self._pending[numframes:]
        return out

    def _capture_raw(self, data: Any) -> None:
        mono = data.mean(axis=1) if data.ndim > 1 else data
        pcm = np.clip(np.rint(mono * 32767.0), -32768, 32767).astype("<i2").tobytes()
        if self._raw_resampler is not None:
            pcm = self._raw_resampler.process(pcm)
        if pcm:
            self._raw_sink(pcm)

    def _apply_gain(self, data: Any) -> Any:
        if self._gain == 1.0:
            return data
        data = data * self._gain
        if float(np.max(np.abs(data))) > 1.0:
            self._clip_events += 1
            if self._native_clip_tally is not None:
                self._native_clip_tally.count += 1
        return np.clip(data, -1.0, 1.0)

    def _resample(self, data: Any) -> Any:
        import numpy as np

        n_channels = data.shape[1] if data.ndim > 1 else 1
        resampled = []
        for ch in range(n_channels):
            column = data[:, ch] if n_channels > 1 else data.reshape(-1)
            resampler = self._resamplers[ch]
            scaled = np.rint(column * 32767.0)
            pcm = np.clip(scaled, -32768, 32767).astype("<i2").tobytes()
            out_i16 = np.frombuffer(resampler.process(pcm), dtype="<i2")
            resampled.append(out_i16.astype(np.float32) / 32767.0)
        return np.stack(resampled, axis=1)


_PINNED_SETTERS = ("persist_mic_volume", "persist_mic_gain", "persist_mic_noise")


def _pin_audio_settings(state: Any, *, auto_gain: int, noise_suppression: int) -> None:
    pinned = {
        "mic_volume": 100,
        "mic_auto_gain": int(auto_gain),
        "mic_noise_suppression": int(noise_suppression),
    }
    targets = [state, getattr(state, "preferences", None)]
    for target in targets:
        if target is None:
            continue
        for name, value in pinned.items():
            if hasattr(target, name):
                setattr(target, name, value)
    for name in _PINNED_SETTERS:
        if hasattr(state, name):
            setattr(state, name, lambda *args, **kwargs: None)


def install_native_rate_capture(
    lva_main: Any,
    *,
    capture_rate: int,
    gain_db: float,
    channels: int,
    auto_gain: int = 0,
    noise_suppression: int = 0,
    native_clip_tally: NativeClipTally | None = None,
    raw_sink: Callable[[bytes], None] | None = None,
) -> Any:
    original_process_audio = lva_main.process_audio
    gain = gain_scalar_from_db(gain_db)

    def process_audio(state: Any, mic: Any, blocksize: int) -> None:
        _pin_audio_settings(
            state, auto_gain=auto_gain, noise_suppression=noise_suppression
        )
        original_recorder = mic.recorder

        def recorder(*args: Any, **kwargs: Any) -> _ResamplingRecorder:
            channels_arg = int(kwargs.get("channels", channels))
            inner = original_recorder(
                samplerate=capture_rate,
                channels=channels_arg,
                blocksize=blocksize,
            )
            return _ResamplingRecorder(
                inner,
                native_rate=capture_rate,
                channels=channels_arg,
                gain=gain,
                native_clip_tally=native_clip_tally,
                raw_sink=raw_sink,
            )

        mic.recorder = recorder
        try:
            original_process_audio(state, mic, blocksize)
        finally:
            mic.recorder = original_recorder

    lva_main.process_audio = process_audio
    _LOGGER.info(
        "Native-rate capture installed: device=%s Hz, transport=%s Hz, gain=%.2f dB, "
        "agc=%s, ns=%s (pinned from config; HA mic entities cannot override)",
        capture_rate,
        TARGET_RATE,
        gain_db,
        auto_gain,
        noise_suppression,
    )
    return process_audio


def install_wake_audio_path(lva_main: Any, hook: Any) -> None:
    install_external_wake_hook(lva_main, hook)
    original_run = lva_main.run

    def run() -> None:
        hook.start()
        try:
            original_run()
        except SystemExit as err:
            code = err.code if isinstance(err.code, int) and err.code > 0 else 1
            os._exit(code)
        finally:
            hook.shutdown()

    lva_main.run = run
