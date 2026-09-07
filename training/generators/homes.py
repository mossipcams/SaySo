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
    "Attic",
    "Cellar",
    "Pantry",
    "Larder",
    "Scullery",
    "Utility Room",
    "Boot Room",
    "Snug",
    "Study",
    "Library",
    "Den",
    "Parlor",
    "Conservatory",
    "Orangery",
    "Solarium",
    "Atrium",
    "Vestibule",
    "Landing",
    "Stairwell",
    "Breakfast Nook",
    "Butler's Pantry",
    "Mud Porch",
    "Screened Porch",
    "Veranda",
    "Terrace",
    "Balcony",
    "Courtyard",
    "Carport",
    "Potting Shed",
    "Wine Cellar",
    "Home Gym",
    "Media Room",
    "Craft Room",
    "Sewing Room",
    "Music Room",
    "Playroom",
    "Bunk Room",
    "Nanny Suite",
    "In-Law Suite",
    "Loft",
)
_FLOORS = ("Upstairs", "Downstairs", "Main Floor", "Basement")
# Name diversity is load-bearing, not cosmetic. Run 008 memorized entity names
# because 16 areas x 10 prefixes x 13 nouns yields only ~2.1k names across 1.25M
# name slots (median 693 repeats each), so copying a name out of the context was
# never the cheapest thing to learn. Keep the combinatorics far above the row
# count; see TRAINING_LOG "Diagnosis: the model memorized entity names".
_PLACEMENTS = (
    "North", "South", "East", "West", "Main", "Side", "Corner", "Ceiling", "Back", "Front",
    "Upper", "Lower", "Inner", "Outer", "Far", "Near", "Left", "Right", "Center", "Rear",
    "Bay", "Alcove", "Nook", "Bench", "Island", "Counter", "Cabinet", "Shelf", "Mantel", "Sill",
    "Reading", "Accent", "Task", "Desk", "Bedside", "Overhead", "Pendant", "Sconce", "Track",
    "Recessed", "Window", "Doorway", "Stair", "Closet", "Hearth", "Bar", "Buffet", "Console",
)
_NEUTRALS = (
    "Morning", "Evening", "Sunset", "Twilight", "Dawn", "Amber", "Copper", "Slate",
    "Willow", "Cedar", "Birch", "Alder", "Rowan", "Hazel", "Juniper", "Laurel",
    "Quill", "Zephyr", "Marlow", "Cobalt", "Vesper", "Tarn", "Cinder", "Harrow", "Lumen", "Fen",
)
_DESCRIPTORS = _PLACEMENTS + _NEUTRALS
# A lawn mower has no "Overhead" and a shopping list has no "Mantel". Things that
# sit somewhere take either pool; portable and abstract ones take neutrals only.
_PLACEABLE = frozenset(
    {"lights", "fans", "switches", "covers", "locks", "media_players", "climate", "buttons"}
)
_OWNERS = (
    "Joe's", "O'Malley's", "McKay's", "Kids'", "Children's", "Nana's", "Grandad's",
    "Priya's", "Wei's", "Amara's", "Sofia's", "Yusuf's", "Ingrid's", "Mateo's",
    "Rosa's", "Dmitri's", "Fatima's", "Kenji's", "Niamh's", "Tomas's", "Leila's",
    "Bram's", "Oona's", "Hallie's",
)

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
    taken: set[str] | None = None,
) -> str:
    """Draw a name from a vocabulary far larger than the corpus can memorize.

    Four shapes across 56 areas, 78 descriptors, 24 owners and 13 nouns give
    ~180k distinct names, so at 40k rows a name is a poor thing to memorize and
    copying it out of the context is the cheaper rule to learn.
    """
    noun, _, _ = _ENTITY_TEMPLATES[capability]
    taken = taken if taken is not None else set()
    pool = _DESCRIPTORS if capability in _PLACEABLE else _NEUTRALS
    for _ in range(24):
        shape = rng.random()
        if shape < 0.72:
            name = f"{area} {rng.choice(pool)} {noun}"
        elif shape < 0.80:
            # Possessives keep the apostrophe failure class in the data without
            # dominating it; a third of a real home is not named after someone.
            name = f"{rng.choice(_OWNERS)} {rng.choice(pool)} {noun}"
        elif shape < 0.84:
            name = f"{rng.choice(_OWNERS)} {area} {noun}"
        elif capability in _PLACEABLE:
            name = f"{area} {rng.choice(_PLACEMENTS)} {rng.choice(_NEUTRALS)} {noun}"
        else:
            name = f"{rng.choice(_OWNERS)} {area} {rng.choice(_NEUTRALS)} {noun}"
        # Two names that slug alike would collide as script tool names.
        if _slug(name) not in taken:
            return name
    return f"{area} {rng.choice(pool)} {slot}{index % 97} {noun}"


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

    taken: set[str] = set()

    for slot, capability in enumerate(capabilities):
        area = area_cycle[slot % len(area_cycle)]
        floor = _FLOORS[(index + slot) % len(_FLOORS)]
        name = _random_entity_name(capability, area, slot, index, rng, taken)
        taken.add(_slug(name))
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

    # Plausible collisions: a second light sharing an area, and a second TV
    # elsewhere sharing the "tv" alias. Both names are drawn, not fixed — a
    # constant here lands in every home of size >= 16 and would be the single
    # most repeated string in the corpus, which is the habit that broke Run 008.
    # The colliding alias, not the name, is what makes these rows hard.
    if size >= 16:
        lights = [entity for entity in entities if entity["capability"] == "lights"]
        if lights:
            base = rng.choice(lights)
            name = _random_entity_name("lights", base["area"], len(entities), index, rng, taken)
            taken.add(_slug(name))
            entities.append(
                make_entity(
                    name=name,
                    capability="lights",
                    area=base["area"],
                    floor=base["floor"],
                    rng=rng,
                    aliases=[f"{base['area']} light", "light"],
                )
            )
        tvs = [entity for entity in entities if entity["capability"] == "media_players"]
        if tvs:
            elsewhere = rng.choice([a for a in _AREAS if a != tvs[0]["area"]])
            name = _random_entity_name("media_players", elsewhere, len(entities), index, rng, taken)
            taken.add(_slug(name))
            entities.append(
                make_entity(
                    name=name,
                    capability="media_players",
                    area=elsewhere,
                    floor=rng.choice(_FLOORS),
                    rng=rng,
                    aliases=[f"{elsewhere} tv", "tv"],
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
