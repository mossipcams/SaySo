"""Replacement audio loop for LVA. Isolated overlay; upstream process_audio is not edited."""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from typing import Optional

import numpy as np

from linux_voice_assistant.models import ServerState
from linux_voice_assistant.webrtc import WebRTCProcessor

from .config import AppConfig
from .wake.detection import Detection
from .wake.livekit import LiveKitWakeWordProvider

_LOGGER = logging.getLogger(__name__)


class _WakePhrase:
    def __init__(self, phrase: str) -> None:
        self.wake_word = phrase
        self.id = "sayso"


def _preroll_after_wake(buffer: deque[int], preroll_ms: int, skip_ms: int) -> bytes:
    need = int(16000 * preroll_ms / 1000)
    skip = int(16000 * skip_ms / 1000)
    samples = list(buffer)
    if len(samples) > need:
        samples = samples[-need:]
    if skip < len(samples):
        samples = samples[skip:]
    if not samples:
        return b""
    return np.asarray(samples, dtype="<i2").tobytes()


def make_process_audio(cfg: AppConfig, provider: LiveKitWakeWordProvider):
    preroll_samples = max(1, int(16000 * cfg.wake_word.preroll_ms / 1000))
    cooldown = cfg.wake_word.post_tts_cooldown_ms / 1000.0

    def process_audio(state: ServerState, mic, block_size: int) -> None:
        n_channels = state.audio_input_channels
        webrtc: Optional[WebRTCProcessor] = None
        rolling: deque[int] = deque(maxlen=preroll_samples)
        last_pipeline = False
        cooldown_until = 0.0
        ha_was_connected = False
        saw_ha = False
        provider.start()

        try:
            _LOGGER.debug("Opening audio input device: %s", mic.name)
            with mic.recorder(samplerate=16000, channels=n_channels, blocksize=block_size) as mic_in:
                while True:
                    raw = mic_in.record(block_size)
                    mic_vol_scalar = max(0.1, min(1.0, state.mic_volume / 100.0))
                    channel_chunks: list[bytes] = []
                    for ch in range(n_channels):
                        col = raw[:, ch] if n_channels > 1 else raw.reshape(-1)
                        chunk = (np.clip(col * mic_vol_scalar, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                        channel_chunks.append(chunk)

                    audio_chunk = channel_chunks[0]
                    agc = state.preferences.mic_auto_gain or 0
                    ns = state.preferences.mic_noise_suppression or 0
                    if agc > 0 or ns > 0:
                        if webrtc is None:
                            webrtc = WebRTCProcessor(agc_level=agc, ns_level=ns)
                        else:
                            webrtc.update_settings(agc, ns)
                        audio_chunk = webrtc.process(audio_chunk)
                        if not audio_chunk:
                            continue

                    if state.satellite is None or not hasattr(state.satellite, "_is_streaming_audio"):
                        continue

                    sat = state.satellite
                    ha_connected = bool(getattr(state, "connections", []))
                    if ha_connected:
                        saw_ha = True
                    if saw_ha and ha_was_connected and not ha_connected:
                        try:
                            state.tts_player.play(str(cfg.sounds.unavailable))
                        except Exception:
                            _LOGGER.debug("Could not play unavailable sound", exc_info=True)
                    ha_was_connected = ha_connected

                    pipeline_active = bool(getattr(sat, "_pipeline_active", False))
                    streaming = bool(getattr(sat, "_is_streaming_audio", False))

                    if pipeline_active and not streaming:
                        # Half-duplex: TTS / processing. Drain mic, suspend wake.
                        provider.suspend()
                        rolling.clear()
                        last_pipeline = True
                        continue

                    if last_pipeline and not pipeline_active:
                        provider.reset()
                        cooldown_until = time.monotonic() + cooldown
                        last_pipeline = False

                    if time.monotonic() < cooldown_until:
                        rolling.clear()
                        continue

                    if not pipeline_active:
                        provider.resume()

                    samples = np.frombuffer(audio_chunk, dtype="<i2")
                    rolling.extend(int(x) for x in samples.tolist())

                    if streaming and not state.muted:
                        audio_chunk_2 = channel_chunks[1] if n_channels >= 2 else None
                        sat.handle_audio(audio_chunk, audio_chunk_2)
                        continue

                    det: Optional[Detection] = provider.process_pcm(audio_chunk)
                    if det and not state.muted:
                        provider.suspend()
                        provider.stop()
                        preroll = _preroll_after_wake(
                            rolling,
                            cfg.wake_word.preroll_ms,
                            cfg.wake_word.wake_skip_ms,
                        )
                        rolling.clear()
                        try:
                            sat.wakeup(_WakePhrase(det.phrase))
                            if preroll and getattr(sat, "_is_streaming_audio", False):
                                sat.handle_audio(preroll)
                        except Exception:
                            _LOGGER.exception("Wake handling failed")
                            try:
                                state.tts_player.play(str(cfg.sounds.failure))
                            except Exception:
                                pass
                        provider.start()
        except Exception:
            _LOGGER.exception("Unexpected error processing audio")
            provider.shutdown()
            sys.exit(1)

    return process_audio
