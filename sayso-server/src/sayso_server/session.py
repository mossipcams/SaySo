"""Established SaySo server sessions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HaSession:
    """An authenticated Home Assistant integration WebSocket session."""

    correlation_id: str
