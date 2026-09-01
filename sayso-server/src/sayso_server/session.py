"""Established SaySo server sessions."""

from __future__ import annotations

from dataclasses import dataclass, field

from sayso_server.graph_store import HomeGraphStore


@dataclass(slots=True)
class HaSession:
    """An authenticated Home Assistant integration WebSocket session."""

    correlation_id: str
    graph: HomeGraphStore = field(default_factory=HomeGraphStore)
