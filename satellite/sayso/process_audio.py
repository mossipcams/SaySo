"""Native-rate capture with one deliberate resample before LVA processes audio.

Upstream LVA opens the microphone with a hardcoded ``samplerate=16000``:

    with mic.recorder(samplerate=16000, channels=n_channels, blocksize=...) as mic_in:
        raw = mic_in.record(block_size)   # float32

For a 44.1 kHz device that makes the audio server resample implicitly, with no
anti-alias filter under our control, and nothing downstream can tell that it
happened. The overlay cannot change that call from the outside without patching
upstream, so instead we wrap ``process_audio`` and force the *recorder* to open
at the configured native rate, then resample each recorded block to 16 kHz
before LVA sees it.

LVA's own loop, WebRTC processing, satellite transport, wake feed, and the SaySo
external wake hook all keep operating on 16 kHz PCM exactly as before. The only
change is that the single 44.1 kHz -> 16 kHz conversion is ours, explicit, and
continuous across block boundaries.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .wake.capture import CaptureResampler, gain_scalar_from_db
from .wake.hook import SaySoExternalWakeHook, install_external_wake_hook

_LOGGER = logging.getLogger(__name__)

TARGET_RATE = 16000


class _ResamplingRecorder:
    """Wrap a soundcard recorder so ``record()`` returns 16 kHz float32.

    The underlying recorder is opened at the native rate; each block is scaled
    by the fixed gain and resampled once. Output keeps LVA's contract exactly:
    ``record(n)`` returns float32 in ``[-1, 1]`` shaped ``(n, channels)`` at
    16 kHz, because that is what upstream asks for when it calls
    ``mic_in.record(block_size)``.

    Honouring that needs two things the native rate makes awkward. Reading ``n``
    frames of *output* means reading ``n * native / 16000`` frames of input --
    at 44.1 kHz that is 2.76x as many, so requesting ``n`` directly would return
    barely a third of the audio asked for. And the ratio is not an integer, so a
    fixed read returns ``n`` or ``n +/- 1`` output frames depending on phase.
    Surplus frames are therefore held over to the next call rather than handed
    back as a short block.
    """

    def __init__(self, recorder: Any, *, native_rate: int, channels: int, gain: float) -> None:
        self._recorder = recorder
        self._native_rate = native_rate
        self._channels = channels
        self._gain = gain
        # One resampler per channel: each channel is an independent stream, and
        # sharing one filter state across channels would corrupt both.
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
            return raw if raw is None else self._apply_gain(np.asarray(raw, dtype=np.float32))

        # Read native audio until enough output frames exist to satisfy the
        # caller. The first call reads slightly extra to cover the resampler's
        # fixed start-up delay; after that the loop runs once.
        native_frames = max(1, -(-numframes * self._native_rate // TARGET_RATE))
        while self._pending is None or self._pending.shape[0] < numframes:
            raw = self._recorder.record(native_frames)
            if raw is None:
                # Device closed mid-stream: hand back whatever is buffered so
                # the caller sees the stream end rather than a silent stall.
                pending, self._pending = self._pending, None
                return pending
            block = self._resample(self._apply_gain(np.asarray(raw, dtype=np.float32)))
            if block.shape[0] == 0:
                continue
            self._pending = (
                block if self._pending is None else np.concatenate((self._pending, block))
            )

        out = self._pending[:numframes]
        self._pending = self._pending[numframes:]
        return out

    def _apply_gain(self, data: Any) -> Any:
        import numpy as np

        if self._gain == 1.0:
            return data
        data = data * self._gain
        if float(np.max(np.abs(data))) > 1.0:
            self._clip_events += 1
        return np.clip(data, -1.0, 1.0)

    def _resample(self, data: Any) -> Any:
        import numpy as np

        n_channels = data.shape[1] if data.ndim > 1 else 1
        resampled = []
        for ch in range(n_channels):
            column = data[:, ch] if n_channels > 1 else data.reshape(-1)
            resampler = self._resamplers[ch]
            # Round rather than truncate: astype() alone truncates toward zero,
            # which biases every sample by up to 1 LSB towards silence.
            scaled = np.rint(column * 32767.0)
            pcm = np.clip(scaled, -32768, 32767).astype("<i2").tobytes()
            out_i16 = np.frombuffer(resampler.process(pcm), dtype="<i2")
            resampled.append(out_i16.astype(np.float32) / 32767.0)
        return np.stack(resampled, axis=1)


# Home Assistant exposes mic volume, auto gain, and noise suppression as
# writable entities, and upstream re-reads all three inside the capture loop --
# `state.mic_volume` every block, `state.preferences.mic_*` every block. Left
# alone, a slider drag in HA silently re-scales the audio Whisper sees and
# re-instantiates the WebRTC processor mid-stream. The command path has to be
# reproducible, so config wins and HA's copies are frozen.
_PINNED_SETTERS = ("persist_mic_volume", "persist_mic_gain", "persist_mic_noise")


def _pin_audio_settings(state: Any, *, auto_gain: int, noise_suppression: int) -> None:
    """Freeze mic volume, AGC, and NS at their configured values.

    ``persist_mic_volume`` / ``persist_mic_gain`` / ``persist_mic_noise`` are the
    only runtime writers of these fields, so shadowing those three bound methods
    on the instance is enough to hold the values for the process lifetime. HA's
    writes become no-ops instead of errors; its UI keeps reading back the pinned
    value, which is the truth about what the audio path is using.
    """
    # 100 makes upstream's `max(0.1, min(1.0, mic_volume / 100))` exactly 1.0, so
    # the multiply becomes a no-op and audio.mic_gain_db is the only gain applied.
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
) -> Any:
    """Wrap ``lva_main.process_audio`` to capture natively and resample once."""
    original_process_audio = lva_main.process_audio
    gain = gain_scalar_from_db(gain_db)

    def process_audio(state: Any, mic: Any, blocksize: int) -> None:
        # Pinned here, not at install time: `state` only exists once upstream
        # starts the capture thread, and this runs before its first block.
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
    """Install the external wake hook and preserve upstream SystemExit handling."""
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
