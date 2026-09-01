"""In-memory Home Graph state with sequenced delta application."""

from __future__ import annotations

from typing import Any

from sayso_server.deltas import RegistryDeltaPayload, StateDeltaPayload
from sayso_server.graph import apply_registry_delta, apply_state_delta
from sayso_server.home_graph import HomeGraphSnapshot


class HomeGraphStore:
    """Hold one Home Graph snapshot and apply sequenced updates."""

    __slots__ = ("_sequence", "_snapshot")

    def __init__(self) -> None:
        self._snapshot: HomeGraphSnapshot | None = None
        self._sequence = 0

    @property
    def snapshot(self) -> HomeGraphSnapshot | None:
        return self._snapshot

    @property
    def sequence(self) -> int:
        return self._sequence

    def clear(self) -> None:
        """Drop graph state until a fresh snapshot arrives."""

        self._snapshot = None
        self._sequence = 0

    def replace_snapshot(self, snapshot: HomeGraphSnapshot) -> None:
        """Replace the entire graph atomically."""

        self._snapshot = snapshot
        self._sequence = snapshot.sequence

    def apply_state_delta(self, payload: dict[str, Any] | StateDeltaPayload) -> bool:
        delta = (
            payload
            if isinstance(payload, StateDeltaPayload)
            else StateDeltaPayload.model_validate(payload)
        )
        if not self._sequence_matches(delta.home_id, delta.sequence):
            return False
        if self._snapshot is None:
            return False
        if not apply_state_delta(
            self._snapshot,
            entity_id=delta.entity_id,
            state=delta.state,
        ):
            return False
        self._sequence = delta.sequence
        return True

    def apply_registry_delta(self, payload: dict[str, Any] | RegistryDeltaPayload) -> bool:
        delta = (
            payload
            if isinstance(payload, RegistryDeltaPayload)
            else RegistryDeltaPayload.model_validate(payload)
        )
        if not self._sequence_matches(delta.home_id, delta.sequence):
            return False
        if self._snapshot is None:
            return False
        if not apply_registry_delta(
            self._snapshot,
            change=delta.change,
            entity_id=delta.entity_id,
            entity=delta.parsed_entity(),
        ):
            return False
        self._sequence = delta.sequence
        return True

    def _sequence_matches(self, home_id: str, sequence: int) -> bool:
        if self._snapshot is None:
            return False
        if self._snapshot.home_id != home_id:
            return False
        return sequence == self._sequence + 1
