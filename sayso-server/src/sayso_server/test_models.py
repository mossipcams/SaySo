"""Tests for shared ControlPlan model types."""

import pytest
from pydantic import ValidationError

from sayso_server.models import Scope, ScopeKind, is_entity_id, validate_semantic_name


def test_scope_current_area_round_trips() -> None:
    scope = Scope.model_validate({"kind": "current_area"})
    dumped = scope.model_dump(mode="json")
    assert Scope.model_validate(dumped) == scope
    assert scope.kind == ScopeKind.CURRENT_AREA
    assert scope.name is None


def test_scope_named_area_requires_name() -> None:
    with pytest.raises(ValidationError):
        Scope.model_validate({"kind": "named_area"})


def test_scope_rejects_entity_id_as_name() -> None:
    with pytest.raises(ValidationError, match="entity"):
        Scope.model_validate({"kind": "named_area", "name": "area.living_room"})


def test_is_entity_id() -> None:
    assert is_entity_id("light.living_room") is True
    assert is_entity_id("climate.downstairs") is True
    assert is_entity_id("living room") is False
    assert is_entity_id("floor lamp") is False


def test_validate_semantic_name_rejects_entity_id() -> None:
    with pytest.raises(ValueError, match="entity"):
        validate_semantic_name("switch.kitchen")
