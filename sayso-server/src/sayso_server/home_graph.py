"""Typed Home Graph snapshot models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class CapabilityKind(StrEnum):
    POWER = "power"
    BRIGHTNESS = "brightness"
    TEMPERATURE = "temperature"
    QUERY = "query"
    SCENE = "scene"
    SCRIPT = "script"


class Capability(BaseModel):
    kind: CapabilityKind
    min_value: float | None = None
    max_value: float | None = None
    attributes: list[str] | None = None


class State(BaseModel):
    value: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    last_changed: str | None = None
    last_updated: str | None = None


class Floor(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)


class Area(BaseModel):
    id: str
    name: str
    floor_id: str | None = None
    aliases: list[str] = Field(default_factory=list)


class Device(BaseModel):
    id: str
    name: str
    manufacturer: str | None = None
    model: str | None = None
    area_id: str | None = None


class Entity(BaseModel):
    entity_id: str
    domain: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    area_id: str | None = None
    device_id: str | None = None
    capabilities: list[Capability] = Field(default_factory=list)
    state: State


class Scene(BaseModel):
    entity_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    area_id: str | None = None
    capabilities: list[Capability] = Field(default_factory=list)


class Script(BaseModel):
    entity_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    area_id: str | None = None
    capabilities: list[Capability] = Field(default_factory=list)


class HomeGraphSnapshot(BaseModel):
    version: int
    sequence: int
    home_id: str
    floors: list[Floor]
    areas: list[Area]
    devices: list[Device]
    entities: list[Entity]
    scenes: list[Scene]
    scripts: list[Script]
