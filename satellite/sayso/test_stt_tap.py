"""The capture tap must equal the bytes actually sent to Home Assistant."""

from __future__ import annotations

import sys
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sayso.events import install_voice_handlers
from sayso.wake.stt_capture import SttAudioRecorder


class _EventType:
    VOICE_ASSISTANT_STT_END = 1
    VOICE_ASSISTANT_ERROR = 2


class _LVAEvent:
    WAKE_WORD_DETECTED = "wake_word_detected"
    LISTENING = "listening"


class _Satellite:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.state = SimpleNamespace(
            muted=False,
            tts_player=SimpleNamespace(play=Mock()),
        )
        self._chime_rearm_pending = False
        self._pipeline_active = False
        self._tts_played = False

    def handle_audio(self, audio_chunk: bytes, audio_chunk_2=None) -> None:
        self.sent.append(audio_chunk)

    def duck(self) -> None:
        pass

    def _emit(self, event) -> None:
        pass

    def _start_audio_streaming(self, phrase) -> None:
        pass


def _install(monkeypatch: pytest.MonkeyPatch, capture: SttAudioRecorder):
    model = ModuleType("aioesphomeapi.model")
    model.VoiceAssistantEventType = _EventType  # type: ignore[attr-defined]
    events = ModuleType("linux_voice_assistant.events")
    events.LVAEvent = _LVAEvent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aioesphomeapi.model", model)
    monkeypatch.setitem(sys.modules, "linux_voice_assistant.events", events)

    protocol = type(
        "VoiceSatelliteProtocol",
        (),
        {
            "handle_voice_event": Mock(),
            "_tts_finished": Mock(),
            "stop": Mock(),
            "handle_audio": _Satellite.handle_audio,
        },
    )
    install_voice_handlers(
        protocol,
        SimpleNamespace(wake="a.wav", failure="b.wav", unavailable="c.wav"),
        None,
        stt_capture=capture,
    )
    return protocol


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")


def test_wav_is_byte_exact_to_what_handle_audio_sent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capture = SttAudioRecorder(tmp_path, sample_rate=16000)
    capture.start()
    protocol = _install(monkeypatch, capture)
    try:
        satellite = _Satellite()
        protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]

        pcm = np.arange(1600, dtype="<i2")
        protocol.handle_audio(satellite, pcm.tobytes())  # type: ignore[attr-defined]
        protocol.handle_voice_event(  # type: ignore[attr-defined]
            satellite, _EventType.VOICE_ASSISTANT_STT_END, {"text": "turn on the light"}
        )
        capture.flush(timeout=2.0)
    finally:
        capture.stop()

    wavs = list((tmp_path / "commands").glob("*.wav"))
    assert len(wavs) == 1
    # Byte-exact: the WAV is the stream the satellite would have sent.
    sent = np.frombuffer(b"".join(satellite.sent), dtype="<i2")
    assert np.array_equal(_read_wav(wavs[0]), sent)
    assert np.array_equal(_read_wav(wavs[0]), pcm)


def test_empty_transcript_produces_a_failure_wav(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capture = SttAudioRecorder(tmp_path, sample_rate=16000)
    capture.start()
    protocol = _install(monkeypatch, capture)
    try:
        satellite = _Satellite()
        protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]
        protocol.handle_audio(  # type: ignore[attr-defined]
            satellite, np.full(500, 7, dtype="<i2").tobytes()
        )
        protocol.handle_voice_event(  # type: ignore[attr-defined]
            satellite, _EventType.VOICE_ASSISTANT_STT_END, {"text": "  "}
        )
        capture.flush(timeout=2.0)
    finally:
        capture.stop()

    assert len(list((tmp_path / "commands").glob("*.wav"))) == 1
    failures = list((tmp_path / "failures").glob("*.wav"))
    assert len(failures) == 1
    assert np.all(_read_wav(failures[0]) == 7)


def test_tap_stops_when_capture_not_started(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capture = SttAudioRecorder(tmp_path, sample_rate=16000)
    capture.start()
    protocol = _install(monkeypatch, capture)
    try:
        satellite = _Satellite()
        # No wakeup: the tap must remain inactive.
        protocol.handle_audio(satellite, np.zeros(100, dtype="<i2").tobytes())  # type: ignore[attr-defined]
        capture.flush(timeout=1.0)
    finally:
        capture.stop()
    assert not list((tmp_path / "commands").glob("*.wav"))


def test_aborted_turn_still_writes_a_failure_wav(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A turn with no stt_end must not leave the capture open forever."""
    capture = SttAudioRecorder(tmp_path, sample_rate=16000)
    capture.start()
    protocol = _install(monkeypatch, capture)
    try:
        satellite = _Satellite()
        protocol.wakeup(satellite, SimpleNamespace(wake_word="SaySo"))  # type: ignore[attr-defined]
        protocol.handle_audio(satellite, np.full(300, 4, dtype="<i2").tobytes())  # type: ignore[attr-defined]
        # No stt_end: the turn is torn down anyway.
        protocol._tts_finished(satellite)  # type: ignore[attr-defined]
        capture.flush(timeout=2.0)
    finally:
        capture.stop()

    failures = list((tmp_path / "failures").glob("*.wav"))
    assert len(failures) == 1
    assert np.all(_read_wav(failures[0]) == 4)
