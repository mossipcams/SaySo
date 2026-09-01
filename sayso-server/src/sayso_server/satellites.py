"""Registered satellite to Home Assistant area mappings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from sayso_server.const import (
    DEFAULT_SATELLITE_AREA_ID,
    DEFAULT_SATELLITE_ID,
    SATELLITE_AREA_ID_ENV_VAR,
)
from sayso_server.home_graph import Area, HomeGraphSnapshot
from sayso_server.normalize import normalize_tokens


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
        matches = _areas_matching_configured(snapshot, registration.area_id)
        if len(matches) != 1:
            return None, "unknown_area"
        return matches[0].id, None


def _normalized_label(label: str) -> str:
    return " ".join(normalize_tokens(label))


def _id_variants(area_id: str) -> frozenset[str]:
    lowered = area_id.casefold()
    variants = {lowered}
    if lowered.startswith("area_"):
        variants.add(lowered.removeprefix("area_"))
    else:
        variants.add(f"area_{lowered}")
    return frozenset(variants)


def _areas_matching_configured(
    snapshot: HomeGraphSnapshot,
    configured: str,
) -> list[Area]:
    config_variants = _id_variants(configured)
    config_label = _normalized_label(configured)
    matches: list[Area] = []
    for area in snapshot.areas:
        if _id_variants(area.id) & config_variants:
            matches.append(area)
            continue
        if not config_label:
            continue
        for label in (area.name, *area.aliases):
            if _normalized_label(label) == config_label:
                matches.append(area)
                break
    return matches


def default_satellite_area_id(*, environ: Mapping[str, str] | None = None) -> str:
    """Return the configured default area id for the MVP Mac satellite."""

    source = os.environ if environ is None else environ
    configured = source.get(SATELLITE_AREA_ID_ENV_VAR, "").strip()
    return configured or DEFAULT_SATELLITE_AREA_ID


def register_default_satellites(
    registry: SatelliteRegistry,
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Register the MVP Mac living-room satellite."""

    registry.register(
        DEFAULT_SATELLITE_ID,
        default_satellite_area_id(environ=environ),
    )
