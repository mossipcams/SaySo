"""Synthetic home generation with coherent entities and distractors."""

from __future__ import annotations

import random
import re
from functools import lru_cache
from typing import Any


from generators.capability_registry import CAPABILITIES, CapabilitySpec

# Repeated names across homes are normal. Vary real fixtures, rooms, brands,
# and household members, not arbitrary adjectives chosen for uniqueness.
_AREAS = (
    "Kitchen", "Living Room", "Dining Room", "Master Bedroom", "Guest Bedroom", "Kids Room",
    "Nursery", "Bathroom", "Ensuite", "Powder Room", "Office", "Study", "Den", "Family Room",
    "Garage", "Workshop", "Shed", "Basement", "Attic", "Hallway", "Landing", "Entryway", "Foyer",
    "Mudroom", "Laundry Room", "Utility Room", "Pantry", "Closet", "Porch", "Patio", "Deck",
    "Balcony", "Garden", "Backyard", "Front Yard", "Driveway", "Sunroom", "Playroom",
    "Guest Suite", "Spare Room", "Home Gym", "Music Room", "Craft Room", "Storage Room",
)
_ROLES = {
    "lights": ("Ceiling Light", "Wall Light", "Lamp", "Ceiling Lights", "Spotlights", "Pendant Light"),
    "fans": ("Ceiling Fan", "Floor Fan", "Desk Fan", "Tower Fan"),
    "switches": ("Plug", "Outlet", "Lamp Plug", "Charger Plug"),
    "covers": ("Window Blinds", "Left Window Blinds", "Right Window Blinds", "Roller Blinds"),
    "locks": ("Front Door Lock", "Back Door Lock", "Side Door Lock", "Entry Door Lock"),
    "media_players": ("TV", "Television", "OLED TV", "Streaming TV",
                      "Speaker", "Soundbar", "Smart Speaker", "Echo Dot"),
    "climate": ("Thermostat", "Heating", "Heat Pump", "Air Conditioner"),
    "vacuums": ("Robot Vacuum", "Vacuum", "Robot Cleaner"),
    "scripts": ("Good Morning", "Good Night", "Leaving Home", "Welcome Home", "Movie Time",
                "Bedtime", "All Lights Off", "Water the Plants", "Feed the Cat", "Evening Cleanup"),
    "scenes": ("Movie Night", "Dinner Time", "Reading Time", "Relax", "Night Lights", "Bedtime Lights"),
    "lawn_mowers": ("Robot Mower", "Lawn Mower", "Mower"),
    "todo_lists": ("Groceries", "Shopping List", "Household Supplies", "Weekend Chores", "Hardware Store"),
    "buttons": ("Wall Button", "Light Button", "Remote Button"),
}
_ROOM_LIGHTS = {
    "Kitchen": ("Ceiling Lights", "Counter Lights", "Under Cabinet Lights", "Island Pendants", "Sink Light"),
    "Dining Room": ("Chandelier", "Pendant Lights", "Wall Lights", "Table Lamp"),
    "Bathroom": ("Mirror Light", "Vanity Lights", "Ceiling Light", "Shower Light"),
    "Bedroom": ("Bedside Lamp", "Reading Lamp", "Ceiling Light", "Left Bedside Lamp", "Right Bedside Lamp"),
    "Living Room": ("Floor Lamp", "Table Lamp", "TV Backlight", "Ceiling Lights", "Reading Lamp"),
    "Office": ("Desk Lamp", "Monitor Backlight", "Ceiling Light", "Reading Lamp"),
    "Outside": ("Wall Light", "Wall Lights", "Floodlight", "Path Lights", "String Lights"),
}
_BRANDS = {
    "lights": ("Hue", "Ikea", "LIFX", "Nanoleaf", "Govee", "Lutron"),
    "fans": ("Dyson", "Honeywell", "Vornado"),
    "switches": ("Shelly", "Kasa", "Wyze", "Sonoff"),
    "covers": ("Ikea", "Somfy"), "locks": ("Yale", "August", "Schlage", "Aqara"),
    "media_players": ("Samsung", "LG", "Sony", "Roku", "Vizio", "Panasonic"),
    "climate": ("Nest", "Ecobee", "Honeywell", "Tado"),
    "vacuums": ("Roborock", "Roomba", "Deebot", "Eufy"),
}
_OWNERS = ("Joe's", "O'Malley's", "McKay's", "Kids'", "Children's", "Nana's", "Grandad's",
           "Priya's", "Wei's", "Amara's", "Sofia's", "Yusuf's", "Ingrid's", "Mateo's",
           "Rosa's", "Dmitri's", "Fatima's", "Kenji's", "Niamh's", "Tomas's", "Leila's",
           "Bram's", "Oona's", "Hallie's")
_OUTDOOR_AREAS = ("Garden", "Backyard", "Front Yard", "Driveway", "Walkway", "Porch", "Patio", "Deck", "Balcony", "Courtyard")
_DEVICE_AREAS = {
    "media_players": ("Living Room", "Family Room", "Den", "Lounge", "Master Bedroom", "Guest Bedroom", "Kids Room", "Media Room", "Home Theater", "Basement"),
    "climate": ("Hallway", "Living Room", "Master Bedroom", "Guest Bedroom", "Office", "Basement"),
    "vacuums": ("Living Room", "Hallway", "Utility Room", "Laundry Room", "Kitchen"),
    "locks": ("Entryway", "Foyer", "Garage", "Porch", "Mudroom"),
    "lawn_mowers": ("Garden", "Backyard", "Front Yard", "Shed", "Garage"),
    "todo_lists": ("Kitchen", "Office", "Utility Room"),
}


def device_areas(capability: str, areas: list[str]) -> list[str]:
    allowed = _DEVICE_AREAS.get(capability)
    if allowed:
        return [area for area in areas if area in allowed]
    if capability in {"covers", "fans"}:
        return [area for area in areas if area not in _OUTDOOR_AREAS]
    return areas


def _roles(capability: str, area: str) -> tuple[str, ...]:
    if capability == "scenes":
        if area in {"Living Room", "Family Room", "Den", "Lounge", "Media Room", "Home Theater"}:
            return ("Movie Night", "Reading Lights", "Relax", "Evening Lights")
        if area in {"Kitchen", "Dining Room", "Dining Area"}:
            return ("Dinner Lights", "Bright Lights", "Evening Lights", "Night Lights")
        return ("Dimmed Lights", "Bright Lights", "Evening Lights", "Night Lights")
    if capability == "lights":
        if area in _OUTDOOR_AREAS:
            return _ROOM_LIGHTS["Outside"]
        if "Bedroom" in area or area in {"Kids Room", "Nursery"}:
            return _ROOM_LIGHTS["Bedroom"]
        if area in {"Bathroom", "Ensuite", "Powder Room", "Washroom"}:
            return _ROOM_LIGHTS["Bathroom"]
        return _ROOM_LIGHTS.get(area, _ROLES[capability])
    if capability == "fans" and area in {"Kitchen", "Bathroom", "Ensuite", "Laundry Room"}:
        return ("Extractor Fan", "Ventilation Fan", "Exhaust Fan")
    if capability == "switches" and area == "Kitchen":
        return ("Coffee Maker Plug", "Kettle Plug", "Counter Outlet", "Toaster Plug")
    return _ROLES[capability]


_ENTITY_TEMPLATES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "lights": ("Light", ("on", "off"), ("on", "off", "brightness", "color", "color_temp")),
    "fans": ("Fan", ("on", "off"), ("on", "off", "percentage")),
    "switches": ("Outlet", ("on", "off"), ("on", "off")),
    "covers": ("Blinds", ("open", "closed"), ("open", "close")),
    "locks": ("Door Lock", ("locked", "unlocked"), ("lock", "unlock")),
    "media_players": ("TV", ("off", "idle", "playing"),
                      ("on", "off", "play", "pause", "volume", "volume_step", "mute",
                       "next", "previous", "search")),
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


# A real home's media players are not interchangeable: a TV has power control and
# no search, a voice speaker has neither power nor a remote. Deriving the profile
# from the fixture's own name keeps the context, the device class and the feature
# set describing one plausible device.
_MEDIA_PROFILES: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
    (
        ("echo", "smart speaker", "google nest", "homepod"),
        "speaker",
        ("play", "pause", "volume", "volume_step", "mute", "next", "previous", "search"),
    ),
    (
        ("speaker", "soundbar", "sonos", "stereo", "receiver"),
        "speaker",
        ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous"),
    ),
    (
        ("streaming tv", "roku", "apple tv", "chromecast", "shield"),
        "tv",
        ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous", "search"),
    ),
)
_MEDIA_DEFAULT = ("on", "off", "play", "pause", "volume", "volume_step", "mute", "next", "previous")


def media_player_profile(name: str) -> tuple[str, tuple[str, ...]]:
    """Return (device_class, features) implied by a media player's name."""
    lowered = name.casefold()
    for keywords, device_class, features in _MEDIA_PROFILES:
        if any(keyword in lowered for keyword in keywords):
            return device_class, features
    return "tv", _MEDIA_DEFAULT


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
    device_class: str | None = None,
) -> dict[str, Any]:
    """Create one entity dict aligned with inference context serialization."""
    cap = CAPABILITIES[capability]
    kind = _KIND_MAP[capability]
    noun, states, default_features = _ENTITY_TEMPLATES[capability]
    domain = cap.domain
    if capability == "media_players" and (device_class is None or features is None):
        profiled_class, profiled_features = media_player_profile(name)
        device_class = device_class or profiled_class
        features = features or profiled_features
    device_class = device_class or _device_class_for(cap)
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
        entity["name"].casefold()
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
    owners: tuple[str, ...] = _OWNERS,
) -> str:
    """Name a real fixture or routine in its room; uniqueness is home-local."""
    taken = taken if taken is not None else set()
    roles = _roles(capability, area)
    brands = _BRANDS.get(capability, ())
    for attempt in range(128):
        role = rng.choice(roles)
        shape = rng.random()
        if capability == "todo_lists":
            name = role
        elif capability == "scripts":
            name = role if shape < 0.65 else f"{area} {rng.choice(('Lights Off', 'Lights On', 'Evening Lighting'))}"
        elif capability == "scenes":
            name = f"{area} {role}"
        elif shape < 0.12:
            name = f"{rng.choice(owners)} {role}"
        elif shape < 0.35 and brands:
            name = f"{area} {rng.choice(brands)} {_ENTITY_TEMPLATES[capability][0]}"
        else:
            name = f"{area} {role}"
        if rng.random() < 0.26:
            name = name[:1] + name[1:].lower()
        if _slug(name) not in taken and name.casefold() not in _eval_entity_names():
            return name
    # Real HA installations commonly number identical fixtures in one room.
    for number in range(2, len(taken) + 3):
        name = f"{area} {roles[0]} {number}"
        if _slug(name) not in taken and name.casefold() not in _eval_entity_names():
            return name
    raise ValueError("no unique fixture name")


def _capability_slots(size: int, rng: random.Random) -> list[str]:
    """Lights and plugs dominate; whole-home appliances remain few."""
    counts = {
        "lights": int(size * .40), "switches": int(size * .22),
        "covers": int(size * .10), "fans": int(size * .06), "media_players": int(size * .04),
        "climate": min(2, max(1, size // 24)), "locks": min(3, max(1, size // 24)),
        "scripts": max(1, size // 20), "scenes": max(1, size // 24),
        "vacuums": int(size >= 32), "buttons": int(size >= 32),
        "todo_lists": int(size >= 32), "lawn_mowers": int(size >= 64),
    }
    slots = [capability for capability, count in counts.items() for _ in range(count)]
    slots += ["lights"] * max(0, size - len(slots))
    rng.shuffle(slots)
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
    # Several devices per room, not a different room for every device.
    rooms = ["Kitchen", "Living Room", "Master Bedroom", "Bathroom", "Hallway", "Entryway", "Garage"]
    extras = [area for area in _AREAS if area not in rooms]
    rooms += rng.sample(extras, min(5, max(1, size // 8)))
    floors = {area: ("Basement" if area == "Basement" else
                     "Upstairs" if "Bedroom" in area or area == "Attic" else "Main Floor") for area in rooms}
    taken: set[str] = set()
    owners = tuple(rng.sample(_OWNERS, 2))
    for slot, capability in enumerate(capabilities):
        area = rng.choice(device_areas(capability, rooms))
        name = _random_entity_name(capability, area, slot, index, rng, taken, owners)
        taken.add(_slug(name))
        noun = _ENTITY_TEMPLATES[capability][0].lower()
        aliases = [name] if capability in {"scripts", "scenes", "todo_lists"} else list(dict.fromkeys([name, f"{area} {noun}"]))
        if capability == "lights":
            synonyms = {"light": "lamp", "lights": "lamps", "lamp": "light", "lamps": "lights",
                        "spotlights": "spots", "backlight": "backlighting"}
            alias = re.sub(r"\b(light|lights|lamp|lamps|spotlights|backlight)\b",
                           lambda match: synonyms[match[0].lower()], name, flags=re.I)
            if alias.casefold() != name.casefold():
                aliases.append(alias)
        entities.append(make_entity(name=name, capability=capability, area=area,
                                    floor=floors[area], rng=rng, aliases=aliases))

    # Plausible collisions: a second light sharing an area, and a second TV
    # elsewhere sharing the "tv" alias. Both names are drawn, not fixed — a
    # constant here lands in every home of size >= 16 and would be the single
    # most repeated string in the corpus, which is the habit that broke Run 008.
    # The colliding alias, not the name, is what makes these rows hard.
    if size >= 16:
        lights = [entity for entity in entities if entity["capability"] == "lights"]
        if lights:
            base = rng.choice(lights)
            name = _random_entity_name("lights", base["area"], len(entities), index, rng, taken, owners)
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
            elsewhere = rng.choice([a for a in device_areas("media_players", rooms) if a != tvs[0]["area"]])
            name = _random_entity_name("media_players", elsewhere, len(entities), index, rng, taken, owners)
            taken.add(_slug(name))
            entities.append(
                make_entity(
                    name=name,
                    capability="media_players",
                    area=elsewhere,
                    floor=floors[elsewhere],
                    rng=rng,
                    aliases=[f"{elsewhere} tv", "tv"],
                )
            )

    remove_canonical_alias_collisions(entities)
    area = sayso_entity_area or entities[index % len(entities)]["area"]
    return {
        "home_id": f"home_{index:06d}_{size}",
        "size": size,
        "sayso_entity_area": area,
        "entities": entities,
        "active_timers": _synthetic_timers(rng) if rng.random() < 0.3 else [],
        "areas": rooms, "area_floors": floors, "owners": owners,
        # Where the entity list came from. A fetched home records whether Home
        # Assistant's Assist exposure list was applied; a synthetic one is exposed
        # by construction. Kept on both so the two shapes stay interchangeable.
        "exposure_source": "synthetic",
    }



def remove_canonical_alias_collisions(entities: list[dict[str, Any]]) -> None:
    """Keep ordinary exact-name targets resolvable while retaining ambiguous distractors."""
    canonical = {entity["name"].casefold() for entity in entities}
    for entity in entities:
        other_names = canonical - {entity["name"].casefold()}
        entity["aliases"] = [alias for alias in entity.get("aliases", [])
                             if alias.casefold() not in other_names
                             and alias.casefold() not in _eval_entity_names()]

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
