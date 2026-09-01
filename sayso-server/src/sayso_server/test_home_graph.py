"""Home Graph snapshot validation and round-trip tests."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import pytest

from sayso_server.api import API_VERSION
from sayso_server.gateway import handle_ha_connection
from sayso_server.graph_store import HomeGraphStore
from sayso_server.home_graph import HomeGraphSnapshot, State
from sayso_server.messages import MessageType

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


def _load_fixture(name: str) -> HomeGraphSnapshot:
    data = json.loads((FIXTURES / name).read_text())
    return HomeGraphSnapshot.model_validate(data)


def test_home_graph_fixture_round_trips_without_field_loss() -> None:
    data = json.loads((FIXTURES / "home_graph.json").read_text())
    snapshot = HomeGraphSnapshot.model_validate(data)
    dumped = snapshot.model_dump(mode="json", exclude_unset=True)
    assert dumped == data
    assert HomeGraphSnapshot.model_validate(dumped) == snapshot


def test_registry_snapshot_fixture_round_trips_without_field_loss() -> None:
    data = json.loads((FIXTURES / "registry_snapshot.json").read_text())
    snapshot = HomeGraphSnapshot.model_validate(data)
    dumped = snapshot.model_dump(mode="json", exclude_unset=True)
    assert dumped == data
    assert HomeGraphSnapshot.model_validate(dumped) == snapshot


def test_snapshot_replaces_graph_state_atomically() -> None:
    baseline = _load_fixture("home_graph.json")
    replacement = baseline.model_copy(
        update={
            "sequence": 100,
            "entities": [
                entity.model_copy(
                    update={
                        "state": State(value="off", attributes={"brightness": 0}),
                    },
                )
                if entity.entity_id == "light.living_room_ceiling"
                else entity
                for entity in baseline.entities
            ],
        },
    )

    store = HomeGraphStore()
    store.replace_snapshot(baseline)
    assert store.sequence == 42
    assert store.snapshot is not None
    ceiling = next(
        entity for entity in store.snapshot.entities if entity.entity_id == "light.living_room_ceiling"
    )
    assert ceiling.state.value == "on"

    store.replace_snapshot(replacement)
    assert store.sequence == 100
    ceiling = next(
        entity for entity in store.snapshot.entities if entity.entity_id == "light.living_room_ceiling"
    )
    assert ceiling.state.value == "off"


def test_state_delta_applies_when_sequence_is_next() -> None:
    store = HomeGraphStore()
    store.replace_snapshot(_load_fixture("home_graph.json"))

    applied = store.apply_state_delta(
        {
            "version": 1,
            "home_id": "eval-home",
            "sequence": 43,
            "entity_id": "light.floor_lamp",
            "state": {"value": "on", "attributes": {"brightness": 128}},
        },
    )

    assert applied is True
    assert store.sequence == 43
    lamp = next(entity for entity in store.snapshot.entities if entity.entity_id == "light.floor_lamp")
    assert lamp.state.value == "on"
    assert lamp.state.attributes["brightness"] == 128


def test_stale_state_delta_does_not_mutate_graph() -> None:
    store = HomeGraphStore()
    store.replace_snapshot(_load_fixture("home_graph.json"))
    before = deepcopy(store.snapshot)

    assert store.apply_state_delta(
        {
            "version": 1,
            "home_id": "eval-home",
            "sequence": 42,
            "entity_id": "light.floor_lamp",
            "state": {"value": "on", "attributes": {}},
        },
    ) is False
    assert store.snapshot == before
    assert store.sequence == 42


def test_out_of_order_state_delta_does_not_mutate_graph() -> None:
    store = HomeGraphStore()
    store.replace_snapshot(_load_fixture("home_graph.json"))
    before = deepcopy(store.snapshot)

    assert store.apply_state_delta(
        {
            "version": 1,
            "home_id": "eval-home",
            "sequence": 45,
            "entity_id": "light.floor_lamp",
            "state": {"value": "on", "attributes": {}},
        },
    ) is False
    assert store.snapshot == before
    assert store.sequence == 42


def test_registry_delta_updates_entity_when_sequence_is_next() -> None:
    store = HomeGraphStore()
    store.replace_snapshot(_load_fixture("home_graph.json"))
    updated_entity = next(
        entity for entity in store.snapshot.entities if entity.entity_id == "light.floor_lamp"
    ).model_copy(update={"aliases": ["lamp", "reading lamp", "corner lamp"]})

    applied = store.apply_registry_delta(
        {
            "version": 1,
            "home_id": "eval-home",
            "sequence": 43,
            "change": "update",
            "entity_id": "light.floor_lamp",
            "entity": updated_entity.model_dump(mode="json"),
        },
    )

    assert applied is True
    assert store.sequence == 43
    lamp = next(entity for entity in store.snapshot.entities if entity.entity_id == "light.floor_lamp")
    assert "corner lamp" in lamp.aliases


def test_reconnect_snapshot_resyncs_expected_graph() -> None:
    store = HomeGraphStore()
    baseline = _load_fixture("home_graph.json")
    store.replace_snapshot(baseline)

    assert store.apply_state_delta(
        {
            "version": 1,
            "home_id": "eval-home",
            "sequence": 43,
            "entity_id": "light.floor_lamp",
            "state": {"value": "on", "attributes": {}},
        },
    )

    resynced = baseline.model_copy(
        update={
            "sequence": 200,
            "entities": [
                entity.model_copy(update={"state": State(value="off", attributes={})})
                if entity.entity_id == "light.floor_lamp"
                else entity
                for entity in baseline.entities
            ],
        },
    )
    store.replace_snapshot(resynced)

    assert store.sequence == 200
    lamp = next(entity for entity in store.snapshot.entities if entity.entity_id == "light.floor_lamp")
    assert lamp.state.value == "off"


class _FakeGatewayWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._recv_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def push(self, message: str) -> None:
        self._recv_queue.put_nowait(message)

    async def receive_str(self) -> str | None:
        return await self._recv_queue.get()


def _envelope(*, msg_type: str, payload: dict, correlation_id: str = "corr-1") -> str:
    return json.dumps(
        {
            "version": API_VERSION,
            "type": msg_type,
            "correlation_id": correlation_id,
            "payload": payload,
        },
    )


@pytest.mark.asyncio
async def test_gateway_applies_graph_snapshot_and_sequenced_deltas() -> None:
    ws = _FakeGatewayWebSocket()
    ws.push(
        json.dumps(
            {
                "version": API_VERSION,
                "type": MessageType.HELLO.value,
                "correlation_id": "session-1",
                "payload": {},
            },
        ),
    )
    snapshot = _load_fixture("home_graph.json")
    ws.push(_envelope(msg_type=MessageType.GRAPH_SNAPSHOT.value, payload=snapshot.model_dump(mode="json")))
    ws.push(
        _envelope(
            msg_type=MessageType.STATE_DELTA.value,
            payload={
                "version": 1,
                "home_id": "eval-home",
                "sequence": 43,
                "entity_id": "light.floor_lamp",
                "state": {"value": "on", "attributes": {"brightness": 90}},
            },
        ),
    )
    ws.push(None)

    store = HomeGraphStore()
    end_state: dict[str, object] = {}

    def capture_end_state(session: object) -> None:
        end_state["sequence"] = session.graph.sequence  # type: ignore[attr-defined]
        lamp = next(
            entity
            for entity in session.graph.snapshot.entities  # type: ignore[attr-defined]
            if entity.entity_id == "light.floor_lamp"
        )
        end_state["lamp_value"] = lamp.state.value

    session = await handle_ha_connection(
        ws,
        authorization="Bearer secret-token",
        server_token="secret-token",
        graph_store=store,
        on_session_ended=capture_end_state,
    )

    assert session is not None
    assert session.graph_ready is True
    assert end_state["sequence"] == 43
    assert end_state["lamp_value"] == "on"
    assert store.snapshot is None
