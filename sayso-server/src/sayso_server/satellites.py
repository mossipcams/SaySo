"""Registered satellite to Home Assistant area mappings."""

from __future__ import annotations

from dataclasses import dataclass

from sayso_server.const import DEFAULT_SATELLITE_AREA_ID, DEFAULT_SATELLITE_ID
from sayso_server.home_graph import HomeGraphSnapshot


@dataclass(frozen=True, slots=True)
class SatelliteRegistration:
    satellite_id: str
    area_id: str


class SatelliteRegistry:
    """In-memory satellite registration table for the MVP."""

    def __init__(self) -> None:
        self._by_id: dict[str, SatelliteRegistration] = {}

    def register(self, satellite_id: str, area_id: str) -> None:
        self._by_id[satellite_id] = SatelliteRegistration(
            satellite_id=satellite_id,
            area_id=area_id,
        )

    def get(self, satellite_id: str) -> SatelliteRegistration | None:
        return self._by_id.get(satellite_id)

    def resolve_area_id(
        self,
        satellite_id: str,
        *,
        snapshot: HomeGraphSnapshot | None,
    ) -> tuple[str | None, str | None]:
        """Return `(area_id, error_code)` where error_code is set on failure."""

        registration = self.get(satellite_id)
        if registration is None:
            return None, "unknown_satellite"
        if snapshot is None:
            return None, "no_graph"
        if not any(area.id == registration.area_id for area in snapshot.areas):
            return None, "unknown_area"
        return registration.area_id, None


def register_default_satellites(registry: SatelliteRegistry) -> None:
    """Register the MVP Mac living-room satellite."""

    registry.register(DEFAULT_SATELLITE_ID, DEFAULT_SATELLITE_AREA_ID)
