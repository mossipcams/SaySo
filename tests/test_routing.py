"""Tests for conservative command-domain routing hints."""

from __future__ import annotations

from typing import Any

import pytest
import voluptuous as vol
from homeassistant.components.fan import FanEntity
from homeassistant.components.light import ColorMode, LightEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import setup_test_component_platform

from custom_components.sayso.routing import (
    RoutingArea,
    RoutingCatalog,
    RoutingDevice,
    RoutingEntity,
    RoutingFloor,
    RoutingPreferences,
    RoutingRegistries,
    identify_command_domain,
    select_tools_for_domain,
)
from custom_components.sayso.schema import (
    compile_tools,
    emit_canonical_json,
    extract_tool_routing_metadata,
)


def _catalog(*entities: RoutingEntity) -> RoutingCatalog:
    return RoutingCatalog(entities=entities)


def _entity(
    entity_id: str,
    *,
    domain: str,
    name: str,
    aliases: tuple[str, ...] = (),
    area_id: str | None = None,
    device_id: str | None = None,
) -> RoutingEntity:
    return RoutingEntity(
        entity_id=entity_id,
        domain=domain,
        name=name,
        aliases=aliases,
        area_id=area_id,
        device_id=device_id,
    )


def _registries(
    *,
    areas: tuple[RoutingArea, ...] = (),
    floors: tuple[RoutingFloor, ...] = (),
    devices: tuple[RoutingDevice, ...] = (),
) -> RoutingRegistries:
    return RoutingRegistries(areas=areas, floors=floors, devices=devices)


class TestIdentifyCommandDomain:
    """Task 14: only exact, unambiguous matches produce a domain hint."""

    def test_exact_entity_name_returns_domain_hint(self) -> None:
        catalog = _catalog(
            _entity("light.living_room", domain="light", name="Living Room"),
            _entity("switch.porch", domain="switch", name="Porch"),
        )

        assert identify_command_domain("turn on the living room", catalog) == "light"

    def test_exact_domain_term_returns_domain_hint(self) -> None:
        catalog = _catalog(
            _entity("light.kitchen", domain="light", name="Kitchen"),
            _entity("switch.garage", domain="switch", name="Garage"),
        )

        assert identify_command_domain("turn off the lights", catalog) == "light"

    def test_alias_matches_like_entity_name(self) -> None:
        catalog = _catalog(
            _entity(
                "light.living_room",
                domain="light",
                name="Living Room Lamp",
                aliases=("LR light",),
            ),
        )

        assert identify_command_domain("turn on the lr light", catalog) == "light"

    def test_case_insensitive_match(self) -> None:
        catalog = _catalog(
            _entity("light.living_room", domain="light", name="Living Room"),
        )

        assert identify_command_domain("TURN ON THE LIVING ROOM", catalog) == "light"

    def test_punctuation_is_ignored(self) -> None:
        catalog = _catalog(
            _entity("light.living_room", domain="light", name="Living Room"),
        )

        assert identify_command_domain("turn on living room!", catalog) == "light"

    def test_plural_domain_wording(self) -> None:
        catalog = _catalog(
            _entity("switch.garage", domain="switch", name="Garage Door"),
        )

        assert identify_command_domain("flip the switches", catalog) == "switch"

    def test_conflicting_entity_domains_return_unknown(self) -> None:
        catalog = _catalog(
            _entity("light.kitchen", domain="light", name="Kitchen Light"),
            _entity("fan.kitchen", domain="fan", name="Kitchen Fan"),
        )

        assert identify_command_domain("turn on the kitchen", catalog) is None

    def test_unknown_term_returns_unknown(self) -> None:
        catalog = _catalog(
            _entity("light.living_room", domain="light", name="Living Room"),
        )

        assert identify_command_domain("turn on the flux capacitor", catalog) is None

    def test_ordinary_non_control_chat_returns_unknown(self) -> None:
        catalog = _catalog(
            _entity("light.living_room", domain="light", name="Living Room"),
            _entity("weather.home", domain="weather", name="Home"),
        )

        assert (
            identify_command_domain("what is the weather like today", catalog) is None
        )

    def test_empty_command_returns_unknown(self) -> None:
        catalog = _catalog(
            _entity("light.living_room", domain="light", name="Living Room"),
        )

        assert identify_command_domain("", catalog) is None

    def test_empty_catalog_returns_unknown(self) -> None:
        assert identify_command_domain("turn on the lights", _catalog()) is None


class TestAreaAndFloorEvidence:
    """Task 15: area and floor registry evidence narrows only when domains agree."""

    def test_exact_area_narrows_when_all_exposed_entities_share_domain(self) -> None:
        catalog = _catalog(
            _entity(
                "light.hall",
                domain="light",
                name="Hall Lamp",
                area_id="area_downstairs",
            ),
            _entity(
                "light.lounge",
                domain="light",
                name="Lounge Lamp",
                area_id="area_downstairs",
            ),
        )
        registries = _registries(
            areas=(RoutingArea("area_downstairs", "Downstairs"),),
        )

        assert (
            identify_command_domain(
                "turn on downstairs",
                catalog,
                registries=registries,
            )
            == "light"
        )

    def test_exact_floor_narrows_when_all_exposed_entities_share_domain(self) -> None:
        catalog = _catalog(
            _entity(
                "switch.office_a",
                domain="switch",
                name="Office A Switch",
                area_id="area_office_a",
            ),
            _entity(
                "switch.office_b",
                domain="switch",
                name="Office B Switch",
                area_id="area_office_b",
            ),
        )
        registries = _registries(
            areas=(
                RoutingArea("area_office_a", "Office A", floor_id="floor_one"),
                RoutingArea("area_office_b", "Office B", floor_id="floor_one"),
            ),
            floors=(RoutingFloor("floor_one", "First Floor"),),
        )

        assert (
            identify_command_domain(
                "turn off first floor",
                catalog,
                registries=registries,
            )
            == "switch"
        )

    def test_duplicate_area_names_return_unknown_without_preference(self) -> None:
        catalog = _catalog(
            _entity(
                "light.office_front",
                domain="light",
                name="Front Desk Lamp",
                area_id="area_office_front",
            ),
            _entity(
                "fan.office_back",
                domain="fan",
                name="Back Office Fan",
                area_id="area_office_back",
            ),
        )
        registries = _registries(
            areas=(
                RoutingArea("area_office_front", "Office"),
                RoutingArea("area_office_back", "Office"),
            ),
        )

        assert (
            identify_command_domain(
                "turn on office",
                catalog,
                registries=registries,
            )
            is None
        )

    def test_mixed_domain_area_contents_return_unknown(self) -> None:
        catalog = _catalog(
            _entity(
                "light.kitchen",
                domain="light",
                name="Kitchen Light",
                area_id="area_kitchen",
            ),
            _entity(
                "switch.kitchen",
                domain="switch",
                name="Kitchen Switch",
                area_id="area_kitchen",
            ),
        )
        registries = _registries(
            areas=(RoutingArea("area_kitchen", "Kitchen"),),
        )

        assert (
            identify_command_domain(
                "turn on kitchen",
                catalog,
                registries=registries,
            )
            is None
        )

    def test_preferred_satellite_area_resolves_duplicate_area_names(self) -> None:
        catalog = _catalog(
            _entity(
                "light.office_front",
                domain="light",
                name="Front Desk Lamp",
                area_id="area_office_front",
            ),
            _entity(
                "fan.office_back",
                domain="fan",
                name="Back Office Fan",
                area_id="area_office_back",
            ),
        )
        registries = _registries(
            areas=(
                RoutingArea("area_office_front", "Office"),
                RoutingArea("area_office_back", "Office"),
            ),
        )
        preferences = RoutingPreferences(area="area_office_front")

        assert (
            identify_command_domain(
                "turn on office",
                catalog,
                registries=registries,
                preferences=preferences,
            )
            == "light"
        )

    def test_preferred_area_is_not_sufficient_by_itself(self) -> None:
        catalog = _catalog(
            _entity(
                "light.living_room",
                domain="light",
                name="Living Room",
                area_id="area_living_room",
            ),
        )
        registries = _registries(
            areas=(RoutingArea("area_living_room", "Living Room"),),
        )
        preferences = RoutingPreferences(area="area_living_room")

        assert (
            identify_command_domain(
                "turn on",
                catalog,
                registries=registries,
                preferences=preferences,
            )
            is None
        )

    def test_no_registry_match_falls_back_to_task14_only(self) -> None:
        catalog = _catalog(
            _entity("light.living_room", domain="light", name="Living Room"),
        )
        registries = _registries(
            areas=(RoutingArea("area_living_room", "Living Room"),),
        )

        assert (
            identify_command_domain(
                "turn on the attic",
                catalog,
                registries=registries,
            )
            is None
        )

    def test_task14_confident_result_is_preserved(self) -> None:
        catalog = _catalog(
            _entity(
                "light.living_room",
                domain="light",
                name="Living Room",
                area_id="area_living_room",
            ),
            _entity(
                "switch.porch",
                domain="switch",
                name="Porch",
                area_id="area_porch",
            ),
        )
        registries = _registries(
            areas=(
                RoutingArea("area_living_room", "Living Room"),
                RoutingArea("area_porch", "Porch"),
            ),
        )

        assert (
            identify_command_domain(
                "turn on the living room",
                catalog,
                registries=registries,
            )
            == "light"
        )

    def test_area_evidence_uses_device_registry_for_entity_area(self) -> None:
        catalog = _catalog(
            _entity(
                "light.garage",
                domain="light",
                name="Garage Light",
                device_id="device_garage",
            ),
        )
        registries = _registries(
            areas=(RoutingArea("area_garage", "Garage"),),
            devices=(RoutingDevice("device_garage", area_id="area_garage"),),
        )

        assert (
            identify_command_domain(
                "turn on garage",
                catalog,
                registries=registries,
            )
            == "light"
        )

    def test_registries_do_not_mutate_catalog_entities(self) -> None:
        entities = (
            _entity(
                "light.hall",
                domain="light",
                name="Hall Lamp",
                area_id="area_downstairs",
            ),
        )
        catalog = RoutingCatalog(entities=entities)
        registries = _registries(
            areas=(RoutingArea("area_downstairs", "Downstairs"),),
        )

        identify_command_domain(
            "turn on downstairs",
            catalog,
            registries=registries,
        )

        assert catalog.entities is entities
        assert catalog.entities[0].area_id == "area_downstairs"


class _RoutingTestLight(LightEntity):
    """Light used to load representative HA LLM tools."""

    _attr_name = "Living Room"
    _attr_unique_id = "living_room"
    _attr_supported_color_modes = {ColorMode.ONOFF}
    _attr_color_mode = ColorMode.ONOFF

    def __init__(self) -> None:
        self._is_on = False

    @property
    def is_on(self) -> bool:
        return self._is_on

    async def async_turn_on(self, **kwargs: Any) -> None:
        self._is_on = True

    async def async_turn_off(self, **kwargs: Any) -> None:
        self._is_on = False


class _RoutingTestFan(FanEntity):
    """Fan used to load representative HA LLM tools."""

    _attr_name = "Bedroom Fan"
    _attr_unique_id = "bedroom_fan"

    def __init__(self) -> None:
        self._is_on = False
        self._percentage = 0

    @property
    def is_on(self) -> bool:
        return self._is_on

    @property
    def percentage(self) -> int:
        return self._percentage

    async def async_turn_on(self, **kwargs: Any) -> None:
        self._is_on = True

    async def async_turn_off(self, **kwargs: Any) -> None:
        self._is_on = False

    async def async_set_percentage(self, percentage: int) -> None:
        self._percentage = percentage


class _FakeDomainTool(llm.Tool):
    """Minimal HA tool for selector edge cases."""

    def __init__(
        self,
        *,
        name: str,
        domain_validator: Any | None = None,
    ) -> None:
        self.name = name
        self.description = f"Fake tool {name}."
        schema: dict[Any, Any] = {}
        if domain_validator is not None:
            schema["domain"] = domain_validator
        self.parameters = vol.Schema(schema)

    async def async_call(
        self,
        hass: Any,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> dict[str, Any]:
        return {"ok": True}


@pytest.fixture
async def representative_ha_tools(hass: HomeAssistant) -> list[llm.Tool]:
    """Load minimum/current HA tool objects with light and fan domain tools."""
    assert await async_setup_component(hass, "intent", {})
    setup_test_component_platform(hass, "light", [_RoutingTestLight()])
    setup_test_component_platform(hass, "fan", [_RoutingTestFan()])
    assert await async_setup_component(hass, "light", {"light": {"platform": "test"}})
    assert await async_setup_component(hass, "fan", {"fan": {"platform": "test"}})
    await hass.async_block_till_done()

    llm_context = llm.LLMContext(
        platform="conversation",
        context=None,
        language="en",
        assistant="conversation",
        device_id=None,
    )
    api = await llm.async_get_api(hass, llm.LLM_API_ASSIST, llm_context)
    return list(api.tools)


class TestSelectToolsForDomain:
    """Task 16: select a safe tool subset from compiled schemas."""

    def test_confident_light_command_filters_incompatible_domain_tools(
        self,
        representative_ha_tools: list[llm.Tool],
    ) -> None:
        """Domain-declared incompatible tools drop; generic/query tools remain."""
        source_tools = representative_ha_tools + [
            _FakeDomainTool(
                name="light_misleading",
                domain_validator=vol.All(cv.ensure_list, [cv.string]),
            ),
            _FakeDomainTool(
                name="switch_only",
                domain_validator=vol.All(cv.ensure_list, [vol.In(["switch"])]),
            ),
        ]
        compiled = compile_tools(source_tools)

        selected = select_tools_for_domain(compiled, source_tools, "light")
        selected_names = {tool["function"]["name"] for tool in selected}

        assert "HassFanSetSpeed" not in selected_names
        assert "switch_only" not in selected_names
        assert {
            "GetDateTime",
            "GetLiveContext",
            "HassTurnOn",
            "HassTurnOff",
            "HassLightSet",
            "HassCancelAllTimers",
            "light_misleading",
        } <= selected_names

    def test_uncertain_command_keeps_complete_schema_byte_for_byte(
        self,
        representative_ha_tools: list[llm.Tool],
    ) -> None:
        """Unknown routing must return the full compiled schema unchanged."""
        compiled = compile_tools(representative_ha_tools)

        selected = select_tools_for_domain(compiled, representative_ha_tools, None)

        assert selected is compiled
        assert emit_canonical_json(selected) == emit_canonical_json(compiled)

    def test_metadata_extraction_never_uses_tool_name_substrings(self) -> None:
        """Tools without explicit domain metadata are treated as unknown."""
        tool = _FakeDomainTool(name="light_by_name_only")

        metadata = extract_tool_routing_metadata(tool)

        assert metadata.declared_domains is None
        assert metadata.retain_always is False

    def test_selected_prompt_is_smaller_than_full_schema(
        self,
        representative_ha_tools: list[llm.Tool],
    ) -> None:
        """Filtered schema reduces serialized prompt size for confident routing."""
        source_tools = representative_ha_tools + [
            _FakeDomainTool(
                name="climate_only",
                domain_validator=vol.All(cv.ensure_list, [vol.In(["climate"])]),
            ),
        ]
        compiled = compile_tools(source_tools)
        full_bytes = len(emit_canonical_json(compiled))
        selected = select_tools_for_domain(compiled, source_tools, "light")
        selected_bytes = len(emit_canonical_json(selected))

        assert selected_bytes < full_bytes
        assert len(selected) < len(compiled)
