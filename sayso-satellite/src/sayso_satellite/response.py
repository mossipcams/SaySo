"""Render Assist speech and map response policy output to playback."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

# Mirrors sayso_server.response_policy.EARCON_TOKEN and text_api earcon placeholder.
EARCON_TOKEN = "\a"
EARCON_SPEECH_PLACEHOLDER = "Done."


class ResponsePlaybackMode(StrEnum):
    EARCON = "earcon"
    SPEECH = "speech"


def extract_assist_response_speech(result: dict[str, Any]) -> str | None:
    """Return assistant response speech from the Assist intent payload."""

    intent = result.get("intent")
    if not isinstance(intent, dict):
        return None
    response = intent.get("response")
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        speech = response.get("speech")
        plain = speech.get("plain") if isinstance(speech, dict) else None
        if isinstance(plain, dict) and isinstance(plain.get("speech"), str):
            return plain["speech"]
    return None


def extract_assist_speech(result: dict[str, Any]) -> str | None:
    """Return Assist speech for display, falling back to the transcript."""

    content = extract_assist_response_speech(result)
    if content is not None:
        return content
    if isinstance(result.get("text"), str):
        return result["text"]
    return None


def resolve_playback_mode(speech: str | None) -> ResponsePlaybackMode:
    """Map server response-policy speech to local earcon or HA TTS playback."""

    if speech is None:
        return ResponsePlaybackMode.SPEECH
    normalized = speech.strip()
    if speech == EARCON_TOKEN or normalized in {EARCON_SPEECH_PLACEHOLDER, "Done"}:
        return ResponsePlaybackMode.EARCON
    return ResponsePlaybackMode.SPEECH


def render_assist_response(
    content: str | None,
    sink: Callable[[str], None] | None = None,
) -> str | None:
    """Print Assist speech and return it when present."""

    if not content or resolve_playback_mode(content) is ResponsePlaybackMode.EARCON:
        return None
    writer = sink or print
    writer(content)
    return content


__all__ = [
    "EARCON_SPEECH_PLACEHOLDER",
    "EARCON_TOKEN",
    "ResponsePlaybackMode",
    "extract_assist_response_speech",
    "extract_assist_speech",
    "render_assist_response",
    "resolve_playback_mode",
]
