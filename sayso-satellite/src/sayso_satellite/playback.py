"""Fetch and play Home Assistant TTS response audio on the Mac."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen as _default_urlopen

from sayso_satellite.assist import TtsOutput

UrlOpen = Callable[[Request], object]


class PlaybackError(RuntimeError):
    """Raised when TTS audio cannot be fetched or played."""


class AudioPlayer(Protocol):
    def play(self, audio: bytes, *, mime_type: str) -> None: ...


def ha_base_url_from_websocket(websocket_url: str) -> str:
    """Derive the Home Assistant HTTP base URL from its WebSocket URL."""

    parsed = urlparse(websocket_url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise PlaybackError("malformed Home Assistant WebSocket URL")
    http_scheme = "https" if parsed.scheme == "wss" else "http"
    return f"{http_scheme}://{parsed.netloc}"


def fetch_tts_audio(
    tts: TtsOutput,
    *,
    token: str,
    base_url: str,
    urlopen: UrlOpen = _default_urlopen,
) -> bytes:
    """Download one HA TTS proxy response with bearer authentication."""

    if not token.strip():
        raise PlaybackError("Home Assistant access token is required for TTS fetch")
    media_url = tts["url"]
    if not media_url.startswith("/"):
        media_url = f"/{media_url.lstrip('/')}"
    request = Request(
        urljoin(f"{base_url.rstrip('/')}/", media_url.lstrip("/")),
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urlopen(request) as response:
            payload = response.read()
    except HTTPError as exc:
        raise PlaybackError(f"TTS fetch failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise PlaybackError(f"TTS fetch failed: {exc.reason}") from exc
    if not payload:
        raise PlaybackError("TTS fetch returned empty audio")
    return payload


def play_tts_response(
    tts: TtsOutput,
    *,
    token: str,
    base_url: str,
    player: AudioPlayer,
    fetch: Callable[..., bytes] | None = None,
) -> None:
    """Fetch HA TTS audio once and play it through the injected player."""

    fetch_tts = fetch or fetch_tts_audio
    audio = fetch_tts(tts, token=token, base_url=base_url)
    try:
        player.play(audio, mime_type=tts["mime_type"])
    except PlaybackError:
        raise
    except OSError as exc:
        raise PlaybackError(f"TTS playback failed: {exc}") from exc


def _suffix_for_mime_type(mime_type: str) -> str:
    lowered = mime_type.lower()
    if "wav" in lowered:
        return ".wav"
    if "ogg" in lowered:
        return ".ogg"
    return ".mp3"


class AfplayAudioPlayer:
    """Play response audio with macOS afplay."""

    def play(self, audio: bytes, *, mime_type: str) -> None:
        suffix = _suffix_for_mime_type(mime_type)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(audio)
                temp_path = handle.name
            subprocess.run(
                ["afplay", temp_path],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise PlaybackError("TTS playback failed: afplay returned non-zero") from exc
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


def default_audio_player() -> AudioPlayer:
    """Return the native Mac player when available."""

    if sys.platform == "darwin":
        return AfplayAudioPlayer()
    raise PlaybackError("no supported audio player on this platform")


def generate_earcon_wav(
    *,
    sample_rate: int = 16_000,
    duration_s: float = 0.12,
    frequency_hz: float = 880.0,
) -> bytes:
    """Return a short mono PCM16 WAV suitable for a completion earcon."""

    import math
    import struct

    sample_count = max(1, int(sample_rate * duration_s))
    frames = bytearray()
    for index in range(sample_count):
        envelope = 1.0 - (index / sample_count)
        sample = int(
            0.35 * envelope * 32_767 * math.sin(2 * math.pi * frequency_hz * index / sample_rate)
        )
        frames.extend(struct.pack("<h", sample))

    data_size = len(frames)
    byte_rate = sample_rate * 2
    block_align = 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )
    return header + bytes(frames)


def play_earcon(player: AudioPlayer) -> None:
    """Play the local completion earcon without HA TTS."""

    try:
        player.play(generate_earcon_wav(), mime_type="audio/wav")
    except PlaybackError:
        raise
    except OSError as exc:
        raise PlaybackError(f"earcon playback failed: {exc}") from exc


def afplay_audio_player() -> AudioPlayer:
    """Construct the macOS afplay-backed player."""

    return AfplayAudioPlayer()
