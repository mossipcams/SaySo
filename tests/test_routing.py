"""Tests for conservative command-domain routing hints."""

from __future__ import annotations

import pytest

from custom_components.sayso.routing import (
    RoutingCatalog,
    RoutingEntity,
    identify_command_domain,
)


def _catalog(*entities: RoutingEntity) -> RoutingCatalog:
    return RoutingCatalog(entities=entities)


def _entity(
    entity_id: str,
    *,
    domain: str,
    name: str,
    aliases: tuple[str, ...] = (),
) -> RoutingEntity:
    return RoutingEntity(
        entity_id=entity_id,
        domain=domain,
        name=name,
        aliases=aliases,
    )


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
