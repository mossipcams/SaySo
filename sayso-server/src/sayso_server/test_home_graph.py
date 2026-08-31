"""Home Graph snapshot validation and round-trip tests."""

import json
from pathlib import Path

from sayso_server.home_graph import HomeGraphSnapshot

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "fixtures"


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
