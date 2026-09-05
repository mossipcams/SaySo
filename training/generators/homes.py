"""Synthetic home generation with coherent entities and distractors."""

from __future__ import annotations

import random
from typing import Any

from generators.capability_registry import CAPABILITIES, CapabilitySpec

_AREAS = (
    "Kitchen",
    "Living Room",
    "Primary Bedroom",
    "Guest Room",
    "Office",
    "Garage",
    "Hallway",
    "Patio",
    "Laundry Room",
    "Workshop",
    "Nursery",
    "Basement",
    "Dining Room",
    "Foyer",
    "Sunroom",
    "Mudroom",
)
_FLOORS = ("Upstairs", "Downstairs", "Main Floor", "Basement")
_PREFIXES = ("North", "South", "East", "West", "Main", "Side", "Corner", "Ceiling", "Back", "Front")

_ENTITY_TEMPLATES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "lights": ("Light", ("on", "off"), ("on", "off", "brightness", "color", "color_temp")),
    "fans": ("Fan", ("on", "off"), ("on", "off", "percentage")),
    "switches": ("Outlet", ("on", "off"), ("on", "off")),
    "covers": ("Blinds", ("open", "closed"), ("open", "close")),
    "locks": ("Door Lock", ("locked", "unlocked"), ("lock", "unlock")),
    "media_players": ("TV", ("off", "idle", "playing"), ("on", "off")),
    "climate": ("Thermostat", ("heat", "cool", "off"), ("heat", "cool", "off")),
    "vacuums": ("Vacuum", ("docked", "cleaning"), ("start", "stop")),
    "scenes": ("Scene", ("off", "on"), ("activate",)),
    "scripts": ("Script", ("off", "on"), ("run",)),
    "lawn_mowers": ("Lawn Mower", ("docked", "mowing"), ("start", "stop")),
    "todo_lists": ("Shopping List", ("idle",), ("add",)),
    "buttons": ("Button", ("idle",), ("press",)),
}

_SPECIAL_NAMES: dict[str, tuple[str, ...]] = {
    "lights": ("Kids' Room Light", "O'Malley's Porch Light"),
    "fans": ("Joe's Workshop Fan", "Children's Bedroom Fan"),
    "switches": ("McKay's Office Outlet", "Joe's Desk Outlet"),
    "covers": ("Children's Bedroom Blinds", "O'Malley's Study Blinds"),
    "locks": ("McKay's Front Door Lock", "Children's Door Lock"),
    "media_players": ("Living Room TV", "Bedroom TV"),
}

_KIND_MAP: dict[str, str] = {
    "lights": "light",
    "fans": "fan",
    "switches": "switch",
    "covers": "cover",
    "locks": "lock",
    "media_players": "media_player",
    "climate": "climate",
    "vacuums": "vacuum",
    "scenes": "scene",
    "scripts": "script",
    "lawn_mowers": "lawn_mower",
    "todo_lists": "todo",
    "buttons": "button",
    "timers": "timer",
}


def _slug(name: str) -> str:
    return "".join(char.casefold() if char.isalnum() else "_" for char in name).strip("_")


def _device_class_for(cap: CapabilitySpec) -> str | None:
    mapping = {
        "covers": "blind",
        "locks": "door",
        "switches": "outlet",
        "media_players": "tv",
    }
    return cap.device_class or mapping.get(cap.name)


def make_entity(
    *,
    name: str,
    capability: str,
    area: str,
    floor: str,
    rng: random.Random,
    aliases: list[str] | None = None,
    features: tuple[str, ...] | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    """Create one entity dict aligned with inference context serialization."""
    cap = CAPABILITIES[capability]
    kind = _KIND_MAP[capability]
    noun, states, default_features = _ENTITY_TEMPLATES[capability]
    domain = cap.domain
    device_class = _device_class_for(cap)
    return {
        "entity_id": f"{domain}.{_slug(name)}",
        "name": name,
        "aliases": aliases or [name],
        "domain": domain,
        "kind": kind,
        "capability": capability,
        "device_class": device_class,
        "area": area,
        "floor": floor,
        "state": state or rng.choice(states),
        "capabilities": list(features or default_features),
    }


def _random_entity_name(
    capability: str,
    area: str,
    slot: int,
    index: int,
    rng: random.Random,
) -> str:
    noun, _, _ = _ENTITY_TEMPLATES[capability]
    if slot == 0 and index % 7 == 0 and capability in _SPECIAL_NAMES:
        return _SPECIAL_NAMES[capability][(index // 7) % len(_SPECIAL_NAMES[capability])]
    prefix = _PREFIXES[(index * 5 + slot * 3) % len(_PREFIXES)]
    return f"{area} {prefix} {noun}"


def _capability_slots(size: int, rng: random.Random) -> list[str]:
    """Distribute capability types across entity slots for a home."""
    tier1 = ["lights", "fans", "switches", "covers", "locks", "media_players", "climate", "lights"]
    tier2 = ["vacuums", "scenes", "scripts"]
    tier3 = ["lawn_mowers", "todo_lists", "buttons"]
    pool = tier1 * 3 + tier2 + tier3
    rng.shuffle(pool)
    slots: list[str] = []
    while len(slots) < size:
        slots.extend(pool)
    return slots[:size]


def generate_home(
    index: int,
    size: int,
    rng: random.Random,
    *,
    sayso_entity_area: str | None = None,
) -> dict[str, Any]:
    """Build a coherent synthetic home with distractors."""
    capabilities = _capability_slots(size, rng)
    entities: list[dict[str, Any]] = []
    area_cycle = list(_AREAS)
    rng.shuffle(area_cycle)

    for slot, capability in enumerate(capabilities):
        area = area_cycle[slot % len(area_cycle)]
        floor = _FLOORS[(index + slot) % len(_FLOORS)]
        name = _random_entity_name(capability, area, slot, index, rng)
        noun = _ENTITY_TEMPLATES[capability][0].lower()
        aliases = [f"{area} {noun}", name.split()[-2] + " " + noun if " " in name else noun]
        entities.append(
            make_entity(
                name=name,
                capability=capability,
                area=area,
                floor=floor,
                rng=rng,
                aliases=aliases,
            )
        )

    # Ensure apostrophe name exists
    if not any("'" in e["name"] for e in entities) and entities:
        entity = entities[0]
        cap = entity["capability"]
        noun = _ENTITY_TEMPLATES[cap][0]
        entity["name"] = f"Joe's {entity['area']} {noun}"
        entity["aliases"].append(entity["name"].replace("'", ""))
        entity["entity_id"] = f"{entity['domain']}.{_slug(entity['name'])}"

    # Plausible collisions: second kitchen light, bedroom TV
    if size >= 16:
        kitchen_lights = [e for e in entities if e["capability"] == "lights" and e["area"] == "Kitchen"]
        if kitchen_lights:
            base = kitchen_lights[0]
            entities.append(
                make_entity(
                    name="Kitchen Ceiling Lamp",
                    capability="lights",
                    area="Kitchen",
                    floor=base["floor"],
                    rng=rng,
                    aliases=["kitchen light", "ceiling lamp"],
                )
            )
        tvs = [e for e in entities if e["capability"] == "media_players"]
        if len(tvs) >= 1:
            entities.append(
                make_entity(
                    name="Bedroom TV",
                    capability="media_players",
                    area="Primary Bedroom",
                    floor="Upstairs",
                    rng=rng,
                    aliases=["bedroom tv", "tv"],
                )
            )

    area = sayso_entity_area or entities[index % len(entities)]["area"]
    return {
        "home_id": f"home_{index:06d}_{size}",
        "size": size,
        "sayso_entity_area": area,
        "entities": entities,
        "active_timers": _synthetic_timers(rng) if rng.random() < 0.3 else [],
    }


def _synthetic_timers(rng: random.Random) -> list[dict[str, Any]]:
    names = ("Pizza Timer", "Laundry Timer", "Kids Bath Timer")
    return [
        {
            "name": rng.choice(names),
            "remaining_seconds": rng.randint(60, 1800),
            "area": rng.choice(_AREAS),
        }
        for _ in range(rng.randint(1, 2))
    ]


def entities_of_capability(home: dict[str, Any], capability: str) -> list[dict[str, Any]]:
    return [e for e in home["entities"] if e.get("capability") == capability]


def entities_in_area(home: dict[str, Any], capability: str, area: str) -> list[dict[str, Any]]:
    return [e for e in entities_of_capability(home, capability) if e["area"] == area]
