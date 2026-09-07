"""Synthetic home generation with coherent entities and distractors."""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Any


from generators.capability_registry import CAPABILITIES, CapabilitySpec

# Names are modelled on real Home Assistant homes, not invented words. The
# vocabulary below is grounded in acon96/Home-Assistant-Requests-V2, whose names
# are compositional over natural language -- "Back Attic Light", "Front Lounge
# Light", "Dyson Pure Fan", "Pond aerator switch". That reference pile has only
# ~1,000 distinct names at a median of 401 repeats each, which is the same
# memorization trap Run 008 fell into, so we keep its shapes and its words but
# draw from a much wider pool. See TRAINING_LOG "Diagnosis".
_AREAS = (
    "Kitchen", "Living Room", "Dining Room", "Master Bedroom", "Guest Bedroom",
    "Kids Room", "Nursery", "Bathroom", "Ensuite", "Powder Room", "Washroom",
    "Office", "Study", "Library", "Den", "Lounge", "Sitting Room", "Family Room",
    "Garage", "Workshop", "Shed", "Tool Shed", "Potting Shed", "Basement",
    "Attic", "Loft", "Hallway", "Corridor", "Stairwell", "Landing", "Entryway",
    "Foyer", "Mudroom", "Laundry Room", "Utility Room", "Pantry", "Closet",
    "Cloakroom", "Porch", "Patio", "Deck", "Balcony", "Terrace", "Courtyard",
    "Garden", "Backyard", "Front Yard", "Driveway", "Walkway", "Carport",
    "Greenhouse", "Conservatory", "Sunroom", "Gym", "Home Gym", "Sauna",
    "Pool House", "Wine Cellar", "Cinema", "Home Theater", "Media Room",
    "Games Room", "Arcade", "Man Cave", "Craft Room", "Art Studio", "Music Room",
    "Playroom", "Storage Room", "Boot Room", "Bike Storage", "Boathouse",
    "Observatory", "Guest Suite", "Spare Room", "Breakfast Nook", "Snug",
    "Dining Area", "Kitchen Island", "Kitchen Counter", "Office Desk",
    "Fireplace", "Aquarium Stand", "Bar", "Nook", "Vestibule",
)
_FLOORS = ("Upstairs", "Downstairs", "Main Floor", "Basement")
# Modifiers are per device and split into where-it-is and what-it-is-like,
# because a lawn mower is never "Dim" and "Old X New Y" is not a name anyone
# writes. The reference pile's 4-word names are exactly position + area +
# quality + noun ("Back Bedroom Warm Light").
_MODIFIERS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "lights": (
        ("Ceiling", "Wall", "Floor", "Desk", "Bedside", "Reading", "Accent",
         "Under Cabinet", "Overhead", "Table", "Corner", "Front", "Back", "Side",
         "Main", "Window", "Upstairs", "Downstairs", "Strip"),
        ("Warm", "Cool", "Bright", "Dim", "Smart", "Spare", "Second", "Outdoor", "Indoor"),
    ),
    "fans": (
        ("Ceiling", "Floor", "Desk", "Wall", "Corner", "Overhead", "Window",
         "Upstairs", "Downstairs", "Loft", "Extractor", "Tower"),
        ("Portable", "Outdoor", "Indoor", "Smart", "Old", "Spare", "Quiet"),
    ),
    "switches": (
        ("Wall", "Corner", "Desk", "Counter", "Under Cabinet", "Front", "Back", "Garden"),
        ("Outdoor", "Indoor", "Smart", "Spare", "Backup"),
    ),
    "covers": (
        ("Window", "Front", "Back", "Side", "Left", "Right", "Upper", "Lower", "Main", "Bay"),
        ("Smart", "Blackout", "Sliding", "Spare"),
    ),
    "locks": (
        ("Front", "Back", "Side", "Main", "Entry", "Inner", "Outer", "Patio", "Gate"),
        ("Smart", "Spare", "Deadbolt"),
    ),
    "media_players": (
        ("Main", "Wall", "Corner", "Upstairs", "Downstairs"),
        ("Smart", "Old", "New", "Spare", "Portable", "Big", "Second"),
    ),
    "climate": (
        ("Main", "Upstairs", "Downstairs", "Floor", "Zone"),
        ("Smart", "Old", "New", "Spare", "Backup", "Primary", "Secondary"),
    ),
    "vacuums": (
        ("Upstairs", "Downstairs", "Main", "Corner"),
        ("Spare", "Old", "New", "Portable", "Smart", "Backup", "Robot"),
    ),
    "scenes": (
        ("Morning", "Evening", "Night", "Movie", "Party", "Guest", "Holiday",
         "Weekend", "Bedtime", "Dinner", "Reading", "Relax", "Away", "Welcome"),
        ("Smart", "Old", "Spare"),
    ),
    "scripts": (
        ("Morning", "Evening", "Night", "Away", "Home", "Movie", "Party", "Guest",
         "Holiday", "Weekend", "Cleaning", "Bedtime", "Lockup", "Shutdown", "Startup"),
        ("Smart", "Old", "Spare"),
    ),
    "lawn_mowers": (("Front", "Back", "Side", "Main"), ("Spare", "Old", "New", "Smart", "Robot")),
    "todo_lists": (
        ("Weekly", "Daily", "Grocery", "Hardware", "Holiday", "Party", "Garden",
         "Pharmacy", "Weekend", "Household"),
        ("Shared", "Spare", "Old"),
    ),
    "buttons": (
        ("Front", "Back", "Entry", "Desk", "Wall", "Bedside", "Doorbell", "Panic"),
        ("Smart", "Spare"),
    ),
}
# Brands, per domain -- nobody sells an "Echo Dot Thermostat".
_BRANDS: dict[str, tuple[str, ...]] = {
    "lights": ("Philips Hue", "Ikea Tradfri", "LIFX", "Nanoleaf", "Govee", "Lutron"),
    "fans": ("Dyson", "Honeywell", "Vornado"),
    "media_players": ("Sonos", "Echo Dot", "Apple TV", "Chromecast", "Roku", "Bose",
                      "Samsung Frame", "Vizio", "Panasonic"),
    "climate": ("Nest", "Ecobee", "Honeywell", "Tado"),
    "vacuums": ("Roborock", "Roomba", "Deebot", "Eufy"),
    "covers": ("Chamberlain", "LiftMaster", "Ikea Fyrtur", "Somfy"),
    "locks": ("Yale", "August", "Schlage", "Aqara"),
    "switches": ("Shelly", "Kasa", "Wyze", "Sonoff"),
}
# Purpose-built devices, the pile's most distinctive shape: "Pond aerator
# switch", "Gutter heating control". Outlets and buttons are what these are.
_PURPOSES = (
    "Pond aerator", "Gutter heating", "Hot tub jets", "Steam shower",
    "Plant watering", "Pet feeder", "Aquarium", "Sump pump", "Water heater",
    "Dehumidifier", "Air purifier", "EV charger", "Well pump", "Chicken coop",
    "Wine fridge", "Christmas lights", "Pool filter", "Irrigation",
    "Space heater", "Trickle charger", "Attic vent", "Radon fan",
    "Septic alarm", "Bird bath", "Fish tank", "Kiln", "Espresso machine",
    "Towel rail", "Boot dryer", "Fly trap",
)
_PURPOSED = frozenset({"switches", "buttons"})
_OWNERS = (
    "Joe's", "O'Malley's", "McKay's", "Kids'", "Children's", "Nana's",
    "Grandad's", "Priya's", "Wei's", "Amara's", "Sofia's", "Yusuf's",
    "Ingrid's", "Mateo's", "Rosa's", "Dmitri's", "Fatima's", "Kenji's",
    "Niamh's", "Tomas's", "Leila's", "Bram's", "Oona's", "Hallie's",
)
# A lawn mower lives outside. Areas a device of this kind can plausibly be in.
_OUTDOOR_ONLY = frozenset({"lawn_mowers"})
_OUTDOOR_AREAS = (
    "Garden", "Backyard", "Front Yard", "Driveway", "Walkway", "Patio", "Deck",
    "Courtyard", "Shed", "Tool Shed", "Potting Shed", "Greenhouse", "Carport",
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


@lru_cache(maxsize=1)
def _eval_entity_names() -> frozenset[str]:
    """Names the v3 suites test on. Held out so eval targets stay unseen.

    Areas overlap by design now that both sides use real room names, so the
    guard is on the whole name. Lazy, mirroring pipeline._excluded_prompts().
    """
    try:
        from evals.v3_quality import build_shadow_specs, gold_specs
    except Exception:  # noqa: BLE001 - generation must not depend on the eval package
        return frozenset()
    return frozenset(
        entity["name"]
        for spec in gold_specs() + build_shadow_specs()
        for entity in spec["home"]["entities"]
    )


def _random_entity_name(
    capability: str,
    area: str,
    slot: int,
    index: int,
    rng: random.Random,
    taken: set[str] | None = None,
) -> str:
    """Draw a name the way real Home Assistant users write them.

    Shapes and their weights follow the reference pile: mostly two and three
    words, "Kitchen Light" / "Back Attic Light" / "Bedroom Warm Light", with
    brands and purpose-built devices in the tail. A quarter render in sentence
    case, because real users are inconsistent and the pile is 26% non-title.
    """
    noun, _, _ = _ENTITY_TEMPLATES[capability]
    taken = taken if taken is not None else set()
    positions, qualities = _MODIFIERS[capability]
    brands = _BRANDS.get(capability)
    for _ in range(32):
        shape = rng.random()
        position, quality = rng.choice(positions), rng.choice(qualities)
        if shape < 0.22:
            name = f"{area} {noun}"
        elif shape < 0.46:
            name = f"{position} {area} {noun}"
        elif shape < 0.68:
            name = f"{area} {position} {noun}"
        elif shape < 0.80:
            name = f"{position} {area} {quality} {noun}"
        elif shape < 0.88:
            # Possessives keep the apostrophe failure class in the data without
            # dominating it; a third of a real home is not named after someone.
            name = f"{rng.choice(_OWNERS)} {area} {noun}"
        elif shape < 0.94 and brands:
            name = f"{rng.choice(brands)} {noun}"
        elif capability in _PURPOSED:
            name = f"{rng.choice(_PURPOSES)} {noun}"
        else:
            name = f"{area} {quality} {noun}"
        if rng.random() < 0.26:
            head, *tail = name.split()
            name = " ".join([head] + [word.lower() for word in tail])
        # Two names that slug alike would collide as script tool names, and a
        # name the eval suites use would stop being held out.
        if _slug(name) not in taken and name not in _eval_entity_names():
            return name
    return f"{area} {rng.choice(positions)} {slot}{index % 97} {noun}"


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
        if capability in _OUTDOOR_ONLY:
            area = rng.choice(_OUTDOOR_AREAS)
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
