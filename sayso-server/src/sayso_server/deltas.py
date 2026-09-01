"""Home Graph delta payload models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from sayso_server.home_graph import Entity, Scene, Script, State

RegistryChange = Literal["create", "update", "remove"]


class StateDeltaPayload(BaseModel):
    version: int
    home_id: str
    sequence: int
    entity_id: str
    state: State | None = None


class RegistryDeltaPayload(BaseModel):
    version: int
    home_id: str
    sequence: int
    change: RegistryChange
    entity_id: str
    entity: dict[str, Any] | None = None

    def parsed_entity(self) -> Entity | Scene | Script | None:
        if self.entity is None:
            return None
        if self.entity_id.startswith("scene."):
            return Scene.model_validate(self.entity)
        if self.entity_id.startswith("script."):
            return Script.model_validate(self.entity)
        return Entity.model_validate(self.entity)
