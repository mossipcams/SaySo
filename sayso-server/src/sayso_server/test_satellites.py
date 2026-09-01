"""Tests for satellite registration helpers."""

from __future__ import annotations

from sayso_server.const import DEFAULT_SATELLITE_AREA_ID, DEFAULT_SATELLITE_ID
from sayso_server.satellites import SatelliteRegistry, register_default_satellites


def test_register_default_satellites_maps_macbook_to_living_room() -> None:
    registry = SatelliteRegistry()
    register_default_satellites(registry)

    registration = registry.get(DEFAULT_SATELLITE_ID)
    assert registration is not None
    assert registration.satellite_id == DEFAULT_SATELLITE_ID
    assert registration.area_id == DEFAULT_SATELLITE_AREA_ID
