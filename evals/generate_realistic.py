from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from evals.cases import CASES_DIR, load_home, normalize_utterance
from evals.runner import production_catalog

SEED = 20261118
HOUSEHOLDS = tuple(range(1, 6))
OUTPUT = CASES_DIR / "realistic_generated.jsonl"
REVIEWED = CASES_DIR / "realistic_v3.jsonl"
SOURCE = "evals/generate_realistic.py"
PROPER_NOUNS = frozenset({"Roomba"})
CATEGORY_CAPS = {
    "ordinary": 22,
    "light_fan_settings": 14,
    "climate": 6,
    "media": 9,
    "timers": 10,
    "routines_vacuum": 8,
    "aliases": 9,
    "multi_action": 8,
    "ambiguity": 5,
    "unavailable": 5,
    "status": 3,
}
CATEGORIES = tuple(CATEGORY_CAPS)
ONOFF_DOMAINS = frozenset({"light", "switch", "fan", "media_player"})
TOTAL_BOUNDS = (422, 922)
WIDE_NEAR_INTENTS = frozenset(
    {"climate__HassClimateSetTemperature", "intent__HassStartTimer"}
)
SHARE_BOUNDS = {"ambiguity": (0.02, 0.06), "unavailable": (0.03, 0.06)}
STATUS_MAX_SHARE = 0.04

ON = ("Turn on {r}.", "Switch on {r}.", "Can you turn on {r}?", "Please turn on {r}.")
OFF = ("Turn off {r}.", "Switch off {r}.", "Can you turn off {r}?", "Turn {r} off.")
LOCK = ("Lock {r}.", "Please lock {r}.", "Can you lock {r}?")
UNLOCK = ("Unlock {r}.", "Please unlock {r}.", "Can you unlock {r}?")
OPEN = ("Open {r}.", "Can you open {r}?", "Please open {r}.")
CLOSE = ("Close {r}.", "Shut {r}.", "Can you close {r}?")
POSITION = ("Set {r} to {n}%.", "Open {r} to {n}%.", "Move {r} to {n}%.")
DIM = ("Dim {r} to {n}%.", "Set {r} to {n}%.", "Turn {r} down to {n}%.")
BRIGHTEN = ("Brighten {r} to {n}%.", "Set {r} to {n}%.", "Turn {r} up to {n}%.")
COLOR = ("Turn {r} {c}.", "Make {r} {c}.", "Set {r} to {c}.")
FAN_LOW = ("Set {r} to {n}%.", "Turn {r} down to {n}%.", "Set {r} speed to {n}%.")
FAN_HIGH = ("Set {r} to {n}%.", "Turn {r} up to {n}%.", "Set {r} speed to {n}%.")
CLIMATE = (
    "Set {r} to {n}.",
    "Set {r} to {n} degrees.",
    "Turn {r} to {n}.",
    "Change {r} to {n} degrees.",
    "Can you set {r} to {n}?",
)
PAUSE = ("Pause {r}.", "Can you pause {r}?")
RESUME = ("Resume {r}.", "Unpause {r}.")
VOLUME = (
    "Set the volume on {r} to {n}%.",
    "Set {r} volume to {n}.",
    "Set {r} volume to {n}%.",
)
VOLUME_UP = ("Turn {r} up a bit.", "Turn up {r}.", "Turn the volume up on {r}.")
VOLUME_DOWN = ("Turn {r} down a bit.", "Turn down {r}.", "Turn the volume down on {r}.")
MUTE = ("Mute {r}.", "Can you mute {r}?")
VACUUM_START = ("Start {r}.", "Run {r}.", "Can you start {r}?")
VACUUM_RETURN = ("Send {r} back to the dock.", "Send {r} home.", "Dock {r}.")
VACUUM_AREA = ("Vacuum the {a}.", "Have {r} clean the {a}.", "Can you vacuum the {a}?")
SCENE = ("Turn on {s}.", "Activate {s}.", "Activate the {s} scene.")
SCRIPT = ("Run my {s} routine.", "Run the {s} script.", "Start the {s} routine.")
AREA_ON = (
    "Turn on the {a} lights.",
    "Switch on the {a} lights.",
    "Turn the {a} lights on.",
)
AREA_OFF = (
    "Turn off the {a} lights.",
    "Switch off the {a} lights.",
    "Turn the {a} lights off.",
)
SAT_ON = ("Switch on the lights.", "Turn the lights on.", "Lights on please.")
SAT_OFF = ("Switch off the lights.", "Turn the lights off.", "Lights off please.")
TODO = ("Add {i} to {l}.", "Put {i} on {l}.", "Add {i} to my {bare}.")
STATUS = {
    "light": ("{be} {r} on?", "Did I leave {r} on?"),
    "lock": ("{be} {r} locked?",),
    "fan": ("{be} {r} running?", "{be} {r} on?"),
    "cover": ("{be} {r} open?",),
    "media_player": ("{be} {r} on?",),
    "switch": ("{be} {r} on?", "Did I leave {r} on?"),
}
COLORS = ("red", "blue", "green", "purple", "orange", "warm white", "yellow", "pink")
ITEMS = ("milk", "bread", "coffee", "bananas", "paper towels", "dish soap", "apples")
TIMER_LABELS = ("pasta", "tea", "laundry", "pizza", "oven", "nap", "cookies", "rice")
ABSENT = (
    (
        "device_absent",
        "garage door",
        ("Open the garage door.", "Close the garage door."),
    ),
    (
        "device_absent",
        "patio lights",
        ("Turn on the patio lights.", "Turn off the patio lights."),
    ),
    (
        "device_absent",
        "hot tub",
        ("Turn on the hot tub.", "Can you turn on the hot tub?"),
    ),
    (
        "device_absent",
        "sprinklers",
        ("Turn on the sprinklers.", "Start the sprinklers."),
    ),
    (
        "device_absent",
        "porch light",
        ("Turn off the porch light.", "Turn on the porch light."),
    ),
    (
        "device_absent",
        "pool pump",
        ("Turn off the pool pump.", "Turn on the pool pump."),
    ),
    (
        "device_absent",
        "attic fan",
        ("Turn on the attic fan.", "Switch on the attic fan."),
    ),
    (
        "device_absent",
        "basement lights",
        ("Turn off the basement lights.", "Turn on the basement lights."),
    ),
    ("device_absent", "dishwasher", ("Start the dishwasher.", "Run the dishwasher.")),
    (
        "device_absent",
        "space heater",
        ("Turn on the space heater.", "Switch off the space heater."),
    ),
    (
        "script_absent",
        "Movie Night routine",
        ("Run my movie night routine.", "Start the movie night routine."),
    ),
    (
        "script_absent",
        "Vacation Mode script",
        ("Start the vacation mode script.", "Run the vacation mode script."),
    ),
    (
        "script_absent",
        "Party Mode routine",
        ("Run my party mode routine.", "Start my party mode routine."),
    ),
    (
        "script_absent",
        "Date Night routine",
        ("Start my date night routine.", "Run my date night routine."),
    ),
    (
        "script_absent",
        "Spa Day routine",
        ("Run the spa day routine.", "Start the spa day routine."),
    ),
)
TIMER_LEFT = (
    "How much time is left on my timer?",
    "How long is left on my timer?",
    "How much time is left on the timer?",
    "How much longer on my timer?",
)
TIMER_PAUSE = ("Pause my timer.", "Pause the timer.", "Can you pause my timer?")
TIMER_CANCEL = (
    "Cancel all my timers.",
    "Cancel all timers.",
    "Cancel all the timers.",
    "Stop all my timers.",
)


class GenerationError(ValueError):
    pass


def household_id(number: int) -> str:
    return f"realistic_eval_household_{number}"


def _args(**values: Any) -> dict[str, Any]:
    return dict(sorted(values.items()))


def _call(tool: str, /, **arguments: Any) -> dict[str, Any]:
    return {"name": tool, "arguments": _args(**arguments)}


def _spoken(name: str) -> str:
    return " ".join(
        word
        if word.endswith("'s") or word.isupper() or word in PROPER_NOUNS
        else word.lower()
        for word in name.split()
    )


def _ref(name: str) -> str:
    first = name.split()[0]
    if first.endswith("'s") or name in PROPER_NOUNS:
        return _spoken(name)
    return f"the {_spoken(name)}"


def _named_ref(entity: dict[str, Any]) -> str:
    if entity["domain"] == "lock":
        return _ref(entity["name"].removesuffix(" Lock"))
    return _ref(entity["name"])


def _plural(text: str) -> bool:
    last = text.split()[-1]
    return last.endswith("s") and not last.endswith("'s")


def _article(number: int) -> str:
    return "an" if str(number).startswith("8") or number in {11, 18} else "a"


def _sentence(text: str) -> str:
    return text[0].upper() + text[1:]


def _labels(home: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for entity in home["entities"]:
        labels = [entity["name"], *(entity.get("aliases") or [])]
        for label in dict.fromkeys(label.casefold() for label in labels):
            index.setdefault(label, []).append(entity)
    return index


def _type_noun(entity: dict[str, Any]) -> str | None:
    domain = entity["domain"]
    last = entity["name"].split()[-1].lower()
    if domain == "light" and last == "lamp":
        return "lamp"
    if domain == "fan":
        return "fan"
    if domain == "cover" and last in {"blinds", "blind", "shade", "curtains"}:
        return last
    if domain == "media_player" and last in {"tv", "television"}:
        return "TV"
    if domain == "vacuum":
        return "vacuum"
    if domain == "climate" and last in {"thermostat", "heating"}:
        return last
    return None


def _refs(entity: dict[str, Any], home: dict[str, Any]) -> list[tuple[str, str]]:
    refs = [(_named_ref(entity), "name")]
    noun = _type_noun(entity)
    area = entity.get("area")
    if noun is None or area is None:
        return refs
    labels = _labels(home)
    same = [
        other
        for other in home["entities"]
        if other["domain"] == entity["domain"] and _type_noun(other) == noun
    ]
    in_room = [other for other in same if other.get("area") == area]

    def free(text: str) -> bool:
        owners = labels.get(text.removeprefix("the ").casefold(), [])
        return all(owner is entity for owner in owners)

    if len(in_room) == 1 and noun != "vacuum":
        room = f"the {area.lower()} {noun}"
        if room not in {text for text, _ in refs} and free(room):
            refs.append((room, "room"))
        if area == home["satellite_area"] and free(f"the {noun}"):
            refs.append((f"the {noun}", "sat"))
            return refs
    if len(same) == 1 and free(f"the {noun}"):
        refs.append((f"the {noun}", "short"))
    return refs


def _onoff_call(
    on: bool, entity: dict[str, Any], form: str, home: dict[str, Any]
) -> dict[str, Any]:
    tool = "intent__HassTurnOn" if on else "intent__HassTurnOff"
    if entity["domain"] in {"lock", "cover"}:
        return _call(tool, name=entity["name"])
    if form == "sat":
        return _call(
            tool,
            area=home["satellite_area"],
            domain=[entity["domain"]],
            name=entity["name"],
        )
    return _call(tool, domain=[entity["domain"]], name=entity["name"])


def _action(category: str, utterance: str, *calls: dict[str, Any], aliases=None):
    return {
        "category": category,
        "utterance": utterance,
        "unavailable": None,
        "response_type": "action",
        "calls": list(calls),
        "clarification": None,
        "spoken_aliases": aliases or {},
    }


def _by_domain(home: dict[str, Any], *domains: str) -> list[dict[str, Any]]:
    return [entity for entity in home["entities"] if entity["domain"] in domains]


def _ordinary(home: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for entity in _by_domain(home, *ONOFF_DOMAINS, "lock", "cover"):
        for text, form in _refs(entity, home):
            if entity["domain"] == "lock":
                pools = ((True, LOCK), (False, UNLOCK))
            elif entity["domain"] == "cover":
                pools = ((True, OPEN), (False, CLOSE))
            else:
                pools = ((True, ON), (False, OFF))
            for on, pool in pools:
                rows.append(
                    _action(
                        "ordinary",
                        rng.choice(pool).format(r=text),
                        _onoff_call(on, entity, form, home),
                    )
                )
            if entity["domain"] == "cover":
                position = rng.choice((20, 30, 40, 50, 60, 75))
                rows.append(
                    _action(
                        "ordinary",
                        rng.choice(POSITION).format(r=text, n=position),
                        _call(
                            "intent__HassSetPosition",
                            name=entity["name"],
                            position=position,
                        ),
                    )
                )
    lit = sorted({entity["area"] for entity in _by_domain(home, "light")})
    for area in lit:
        for on, pool in ((True, AREA_ON), (False, AREA_OFF)):
            tool = "intent__HassTurnOn" if on else "intent__HassTurnOff"
            rows.append(
                _action(
                    "ordinary",
                    rng.choice(pool).format(a=area.lower()),
                    _call(tool, area=area, domain=["light"]),
                )
            )
    if home["satellite_area"] in lit:
        for on, pool in ((True, SAT_ON), (False, SAT_OFF)):
            tool = "intent__HassTurnOn" if on else "intent__HassTurnOff"
            rows.append(
                _action(
                    "ordinary",
                    rng.choice(pool),
                    _call(tool, area=home["satellite_area"], domain=["light"]),
                )
            )
    for entity in _by_domain(home, "todo"):
        bare = _spoken(entity["name"])
        if not bare.endswith("list"):
            bare = f"{bare} list"
        for item in rng.sample(ITEMS, 3):
            rows.append(
                _action(
                    "ordinary",
                    rng.choice(TODO).format(i=item, l=f"the {bare}", bare=bare),
                    _call("todo__HassListAddItem", item=item, name=entity["name"]),
                )
            )
    return rows


def _light_call(entity: dict[str, Any], **values: Any) -> dict[str, Any]:
    return _call("light__HassLightSet", domain=["light"], name=entity["name"], **values)


def _settings(home: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for entity in _by_domain(home, "light"):
        for text, _ in _refs(entity, home):
            level = rng.choice((10, 15, 25, 30, 40, 60, 70, 80, 90))
            pool = DIM if level < 50 else BRIGHTEN
            rows.append(
                _action(
                    "light_fan_settings",
                    rng.choice(pool).format(r=text, n=level),
                    _light_call(entity, brightness=level),
                )
            )
            if _type_noun(entity) == "lamp" or "Ceiling" in entity["name"]:
                color = rng.choice(COLORS)
                rows.append(
                    _action(
                        "light_fan_settings",
                        rng.choice(COLOR).format(r=text, c=color),
                        _light_call(entity, color=color),
                    )
                )
    for entity in _by_domain(home, "fan"):
        for text, _ in _refs(entity, home):
            for speed in rng.sample((20, 30, 45, 55, 70, 85, 100), 2):
                pool = FAN_LOW if speed < 50 else FAN_HIGH
                rows.append(
                    _action(
                        "light_fan_settings",
                        rng.choice(pool).format(r=text, n=speed),
                        _call(
                            "fan__HassFanSetSpeed",
                            domain=["fan"],
                            name=entity["name"],
                            percentage=speed,
                        ),
                    )
                )
    return rows


def _climate(home: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for entity in _by_domain(home, "climate"):
        refs = _refs(entity, home)
        for temperature in rng.sample(range(62, 78), 8):
            text, _ = rng.choice(refs)
            rows.append(
                _action(
                    "climate",
                    rng.choice(CLIMATE).format(r=text, n=temperature),
                    _call(
                        "climate__HassClimateSetTemperature",
                        name=entity["name"],
                        temperature=temperature,
                    ),
                )
            )
    return rows


def _media_rows(
    entity: dict[str, Any], text: str, rng: random.Random, category: str, aliases=None
) -> list[dict[str, Any]]:
    domain = ["media_player"]
    name = entity["name"]
    volume = rng.choice((15, 20, 25, 35, 40, 50, 60))
    commands = (
        (PAUSE, _call("media_player__HassMediaPause", domain=domain, name=name)),
        (RESUME, _call("media_player__HassMediaUnpause", domain=domain, name=name)),
        (
            VOLUME,
            _call(
                "media_player__HassSetVolume",
                domain=domain,
                name=name,
                volume_level=volume,
            ),
        ),
        (
            VOLUME_UP,
            _call("media_player__HassSetVolumeRelative", name=name, volume_step="up"),
        ),
        (
            VOLUME_DOWN,
            _call("media_player__HassSetVolumeRelative", name=name, volume_step="down"),
        ),
        (MUTE, _call("media_player__HassMediaPlayerMute", domain=domain, name=name)),
    )
    return [
        _action(
            category,
            rng.choice(pool).format(r=text, n=volume),
            call,
            aliases=aliases,
        )
        for pool, call in commands
    ]


def _media(home: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for entity in _by_domain(home, "media_player"):
        for text, _ in _refs(entity, home):
            rows.extend(_media_rows(entity, text, rng, "media"))
    return rows


def _timers(home: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    labels = rng.sample(TIMER_LABELS, 4)
    areas = sorted(home["areas"])
    minutes = rng.sample((5, 8, 10, 12, 15, 20, 25, 30, 40), 4)
    timer_area = rng.choice(areas)
    seconds = rng.choice((30, 45, 90))
    rows = [
        _action(
            "timers",
            f"Set {_article(minutes[0])} {minutes[0]} minute {labels[0]} timer.",
            _call("intent__HassStartTimer", minutes=minutes[0], name=labels[0]),
        ),
        _action(
            "timers",
            f"Set a {labels[1]} timer for {minutes[1]} minutes.",
            _call("intent__HassStartTimer", minutes=minutes[1], name=labels[1]),
        ),
        _action(
            "timers",
            f"Start a {seconds} second {labels[2]} timer.",
            _call("intent__HassStartTimer", name=labels[2], seconds=seconds),
        ),
        _action(
            "timers",
            f"Set a 1 hour {labels[3]} timer.",
            _call("intent__HassStartTimer", hours=1, name=labels[3]),
        ),
        _action(
            "timers",
            f"Set a timer for {minutes[2]} minutes.",
            _call("intent__HassStartTimer", minutes=minutes[2]),
        ),
        _action(
            "timers",
            f"Start {_article(minutes[3])} {minutes[3]} minute timer.",
            _call("intent__HassStartTimer", minutes=minutes[3]),
        ),
        _action(
            "timers",
            f"How much time is left on the {labels[0]} timer?",
            _call("intent__HassTimerStatus", name=labels[0]),
        ),
        _action(
            "timers",
            f"How long is left on the {labels[1]} timer?",
            _call("intent__HassTimerStatus", name=labels[1]),
        ),
        _action(
            "timers",
            f"Pause the {labels[3]} timer.",
            _call("intent__HassPauseTimer", name=labels[3]),
        ),
        _action(
            "timers",
            f"Pause the {timer_area.lower()} timer.",
            _call("intent__HassPauseTimer", area=timer_area),
        ),
        _action(
            "timers",
            f"Cancel all timers in the {timer_area.lower()}.",
            _call("intent__HassCancelAllTimers", area=timer_area),
        ),
    ]
    for pool, tool in (
        (TIMER_LEFT, "intent__HassTimerStatus"),
        (TIMER_PAUSE, "intent__HassPauseTimer"),
        (TIMER_CANCEL, "intent__HassCancelAllTimers"),
    ):
        rows.extend(
            _action("timers", text, _call(tool)) for text in rng.sample(pool, len(pool))
        )
    return rows


def _routines(home: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for entity in _by_domain(home, "vacuum"):
        refs = _refs(entity, home)
        for text, _ in refs:
            rows.append(
                _action(
                    "routines_vacuum",
                    rng.choice(VACUUM_START).format(r=text),
                    _call(
                        "vacuum__HassVacuumStart",
                        domain=["vacuum"],
                        name=entity["name"],
                    ),
                )
            )
            rows.append(
                _action(
                    "routines_vacuum",
                    rng.choice(VACUUM_RETURN).format(r=text),
                    _call(
                        "vacuum__HassVacuumReturnToBase",
                        domain=["vacuum"],
                        name=entity["name"],
                    ),
                )
            )
        for area in rng.sample(sorted(home["areas"]), 4):
            text, _ = rng.choice(refs)
            rows.append(
                _action(
                    "routines_vacuum",
                    rng.choice(VACUUM_AREA).format(a=area.lower(), r=text),
                    _call(
                        "vacuum__HassVacuumCleanArea", area=area, name=entity["name"]
                    ),
                )
            )
    for entity in _by_domain(home, "scene"):
        for template in rng.sample(SCENE, 2):
            rows.append(
                _action(
                    "routines_vacuum",
                    template.format(s=_spoken(entity["name"])),
                    _call("intent__HassTurnOn", domain=["scene"], name=entity["name"]),
                )
            )
    for entity in _by_domain(home, "script"):
        tool = f"script__{entity['entity_id'].split('.', 1)[1]}"
        spoken = entity["name"].lower().removesuffix(" routine")
        for template in rng.sample(SCRIPT, 2):
            rows.append(
                _action(
                    "routines_vacuum",
                    template.format(s=spoken),
                    {"name": tool, "arguments": {}},
                )
            )
    return rows


def _aliases(home: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    labels = _labels(home)
    rows = []
    for entity in home["entities"]:
        for alias in entity.get("aliases") or []:
            if (
                len(labels[alias.casefold()]) != 1
                or alias.casefold() == entity["name"].casefold()
            ):
                continue
            text = f"the {alias}"
            spoken = {entity["name"]: alias}
            domain = entity["domain"]
            if domain in ONOFF_DOMAINS:
                for on, pool in ((True, ON), (False, OFF)):
                    rows.append(
                        _action(
                            "aliases",
                            rng.choice(pool).format(r=text),
                            _onoff_call(on, entity, "alias", home),
                            aliases=spoken,
                        )
                    )
            if domain == "light":
                level = rng.choice((20, 35, 50, 65, 80))
                rows.append(
                    _action(
                        "aliases",
                        rng.choice(DIM if level < 50 else BRIGHTEN).format(
                            r=text, n=level
                        ),
                        _light_call(entity, brightness=level),
                        aliases=spoken,
                    )
                )
            if domain == "fan":
                speed = rng.choice((25, 40, 60, 75))
                rows.append(
                    _action(
                        "aliases",
                        rng.choice(FAN_LOW if speed < 50 else FAN_HIGH).format(
                            r=text, n=speed
                        ),
                        _call(
                            "fan__HassFanSetSpeed",
                            domain=["fan"],
                            name=entity["name"],
                            percentage=speed,
                        ),
                        aliases=spoken,
                    )
                )
            if domain == "media_player":
                rows.extend(
                    rng.sample(_media_rows(entity, text, rng, "aliases", spoken), 2)
                )
    return rows


def _fragments(home: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    fragments = []
    for entity in _by_domain(home, *ONOFF_DOMAINS, "lock", "cover"):
        text = _named_ref(entity)
        if entity["domain"] == "lock":
            verbs = (("lock", True), ("unlock", False))
        elif entity["domain"] == "cover":
            verbs = (("open", True), ("close", False))
        else:
            verbs = (("turn on", True), ("turn off", False))
        for verb, on in verbs:
            fragments.append(
                (
                    f"{verb} {text}",
                    _onoff_call(on, entity, "name", home),
                    entity["name"],
                )
            )
    return fragments


def _multi(home: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    fragments = _fragments(home)
    rows = []
    for _ in range(40):
        first, second = rng.sample(fragments, 2)
        if first[2] == second[2]:
            continue
        rows.append(
            _action(
                "multi_action",
                _sentence(f"{first[0]} and {second[0]}."),
                first[1],
                second[1],
            )
        )
    return rows


def _ambiguity(home: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for label, owners in sorted(_labels(home).items()):
        if len(owners) < 2:
            continue
        areas = {owner.get("area") for owner in owners}
        room = areas.pop().lower() if len(areas) == 1 else None
        candidates = sorted(owner["name"] for owner in owners)
        level = rng.choice((20, 30, 40, 60, 70))
        texts = [
            rng.choice(ON).format(r=f"the {label}"),
            rng.choice(OFF).format(r=f"the {label}"),
            rng.choice(DIM if level < 50 else BRIGHTEN).format(
                r=f"the {label}", n=level
            ),
            f"Turn the {label} on.",
            f"Set the {label} to {rng.choice((25, 45, 75))}%.",
        ]
        if room:
            texts.append(f"Turn off the {label} in the {room}.")
            texts.append(f"Switch on the {label} in the {room}.")
        for text in texts:
            rows.append(
                {
                    "category": "ambiguity",
                    "utterance": text,
                    "unavailable": None,
                    "response_type": "clarification",
                    "calls": [],
                    "clarification": {"candidates": candidates},
                    "spoken_aliases": {},
                }
            )
    return rows


def _mentions_home(home: dict[str, Any], utterance: str) -> list[str]:
    spoken = f" {normalize_utterance(utterance)} "
    return sorted(
        label for label in _labels(home) if f" {normalize_utterance(label)} " in spoken
    )


def _unavailable(home: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for kind, capability, phrasings in rng.sample(ABSENT, len(ABSENT)):
        for utterance in phrasings:
            if _mentions_home(home, utterance):
                continue
            rows.append(
                {
                    "category": "unavailable",
                    "utterance": utterance,
                    "unavailable": {
                        "kind": kind,
                        "capability": capability,
                        "removed_tools": [],
                    },
                    "response_type": "refusal",
                    "calls": [],
                    "clarification": None,
                    "spoken_aliases": {},
                }
            )
    return rows


def _status(home: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for entity in _by_domain(home, *STATUS):
        text, _ = rng.choice(_refs(entity, home))
        verb = "Are" if _plural(text) else "Is"
        rows.append(
            {
                "category": "status",
                "utterance": rng.choice(STATUS[entity["domain"]]).format(
                    r=text, be=verb
                ),
                "unavailable": None,
                "response_type": "status",
                "calls": [
                    _call(
                        "homeassistant__GetLiveContext",
                        domain=entity["domain"],
                        name=entity["name"],
                    )
                ],
                "clarification": None,
                "spoken_aliases": {},
            }
        )
    return rows


BUILDERS = {
    "ordinary": _ordinary,
    "light_fan_settings": _settings,
    "climate": _climate,
    "media": _media,
    "timers": _timers,
    "routines_vacuum": _routines,
    "aliases": _aliases,
    "multi_action": _multi,
    "ambiguity": _ambiguity,
    "unavailable": _unavailable,
    "status": _status,
}


def reviewed_utterances() -> set[str]:
    utterances = set()
    for path in sorted(CASES_DIR.glob("*.jsonl")):
        if path == OUTPUT:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                utterances.add(normalize_utterance(json.loads(line)["utterance"]))
    return utterances


def _finish(draft: dict[str, Any], number: int, index: int) -> dict[str, Any]:
    case_id = f"realistic_gen_{number}_{draft['category']}_{index:03d}"
    return {
        "id": case_id,
        "category": draft["category"],
        "tags": ["promotion", "generated"],
        "household": household_id(number),
        "utterance": draft["utterance"],
        "unavailable": draft["unavailable"],
        "expected": {
            "response_type": draft["response_type"],
            "calls": draft["calls"],
            "clarification": draft["clarification"],
            "forbidden_entities": [],
            "spoken_aliases": draft["spoken_aliases"],
        },
        "provenance": {"source": SOURCE, "source_id": case_id, "held_out": True},
    }


def answer_key(row: dict[str, Any]) -> tuple[str, str]:
    expected = row.get("expected", row)
    unavailable = row["unavailable"] or {}
    answer = {
        "response_type": expected["response_type"],
        "calls": expected["calls"],
        "candidates": sorted((expected["clarification"] or {}).get("candidates", [])),
        "unavailable": [unavailable.get("kind"), unavailable.get("capability")],
    }
    return normalize_utterance(row["utterance"]), json.dumps(answer, sort_keys=True)


def _command(text: str) -> str:
    words = normalize_utterance(text).split()
    if "off" in words:
        return "off"
    return "on" if "on" in words else "level"


def _heard(text: str, label: str) -> bool:
    return f" {normalize_utterance(label)} " in f" {normalize_utterance(text)} "


def _owners(home: dict[str, Any], name: str | None) -> list[dict[str, Any]]:
    return _labels(home).get(name.casefold(), []) if name else []


def _form(
    text: str, call: dict[str, Any], spoken: dict[str, str], home: dict[str, Any]
) -> str:
    name = call["arguments"].get("name")
    if name and any(
        _heard(text, label) for label in (name, name.removesuffix(" Lock"))
    ):
        return "name"
    aliases = [
        *spoken.values(),
        *(
            alias
            for owner in _owners(home, name)
            for alias in owner.get("aliases") or []
        ),
    ]
    if any(_heard(text, alias) for alias in aliases):
        return "alias"
    return "short"


def _call_key(
    text: str, call: dict[str, Any], spoken: dict[str, str], home: dict[str, Any]
) -> tuple:
    arguments = call["arguments"]
    if "area" in arguments or "name" in arguments:
        target = (arguments.get("area", ""), arguments.get("name", ""))
    else:
        target = (call["name"], "")
    values = tuple(
        sorted(key for key in arguments if key not in {"area", "name", "domain"})
    )
    return (call["name"], target, _form(text, call, spoken, home), values)


def _target_labels(call: dict[str, Any], home: dict[str, Any]) -> list[str]:
    arguments = call["arguments"]
    name = arguments.get("name")
    area = arguments.get("area")
    owners = _owners(home, name)
    if area is not None:
        if area != home["satellite_area"]:
            return [area]
        if name is None:
            return [area, "the lights"]
        return [
            area,
            *(text for o in owners for text, form in _refs(o, home) if form == "sat"),
        ]
    if name is None:
        return []
    words = normalize_utterance(name).split()
    spans = [
        " ".join(words[start:end])
        for start in range(len(words))
        for end in range(start + 1, len(words) + 1)
    ]
    return [
        *spans,
        *(alias for owner in owners for alias in owner.get("aliases") or []),
        *(text for owner in owners for text, _ in _refs(owner, home)),
    ]


def literal(row: dict[str, Any], home: dict[str, Any]) -> bool:
    for call in row["expected"]["calls"]:
        labels = _target_labels(call, home)
        if labels and not any(_heard(row["utterance"], label) for label in labels):
            return False
    return True


def reviewed_near(number: int) -> Counter:
    home = load_home(household_id(number))
    near: Counter = Counter()
    for line in filter(str.strip, REVIEWED.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if row["household"] == household_id(number) and literal(row, home):
            near.update(item for item, _ in near_keys(row, home))
    return near


def near_keys(
    row: dict[str, Any], home: dict[str, Any]
) -> list[tuple[tuple[Any, ...], int]]:
    expected = row.get("expected", row)
    text = row["utterance"]
    response = expected["response_type"]
    if response == "refusal":
        return [(("refusal", row["unavailable"]["capability"]), 1)]
    if response == "clarification":
        candidates = tuple(sorted(expected["clarification"]["candidates"]))
        return [
            (("clarify", candidates), 2),
            (("clarify", candidates, _command(text)), 1),
        ]
    keys = [
        _call_key(text, call, expected["spoken_aliases"], home)
        for call in expected["calls"]
    ]
    if not keys:
        return []
    if len(keys) > 1:
        return [(("multi", tuple(sorted(keys, key=repr))), 1)]
    (key,) = keys
    return [(key, 2 if key[0] in WIDE_NEAR_INTENTS else 1)]


def generate() -> list[dict[str, Any]]:
    reviewed = reviewed_utterances()
    answered: set[tuple[str, str]] = set()
    rows = []
    for number in HOUSEHOLDS:
        home = load_home(household_id(number))
        seen: set[str] = set()
        near = reviewed_near(number)
        for category in CATEGORIES:
            rng = random.Random(f"{SEED}:{number}:{category}")
            kept = []
            for draft in BUILDERS[category](home, rng):
                key = normalize_utterance(draft["utterance"])
                keys = near_keys(draft, home)
                if key in seen or key in reviewed or answer_key(draft) in answered:
                    continue
                if any(near[item] >= limit for item, limit in keys):
                    continue
                seen.add(key)
                near.update(item for item, _ in keys)
                kept.append(draft)
            cap = CATEGORY_CAPS[category]
            if len(kept) > cap:
                keep = sorted(rng.sample(range(len(kept)), cap))
                for index, draft in enumerate(kept):
                    if index not in keep:
                        seen.discard(normalize_utterance(draft["utterance"]))
                        near.subtract(item for item, _ in near_keys(draft, home))
                kept = [kept[index] for index in keep]
            answered.update(answer_key(draft) for draft in kept)
            rows.extend(
                _finish(draft, number, index)
                for index, draft in enumerate(kept, start=1)
            )
    for row in rows:
        validate_case(row)
    check_composition(rows)
    return rows


def _schema_ok(value: Any, spec: dict[str, Any]) -> bool:
    if "anyOf" in spec:
        return any(_schema_ok(value, option) for option in spec["anyOf"])
    if "enum" in spec and value not in spec["enum"]:
        return False
    kind = spec.get("type")
    checks = {
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ),
        "boolean": lambda: isinstance(value, bool),
        "object": lambda: isinstance(value, dict),
        "array": lambda: (
            isinstance(value, list)
            and all(_schema_ok(item, spec.get("items") or {}) for item in value)
        ),
    }
    if kind is not None and not checks[kind]():
        return False
    if "minimum" in spec and value < spec["minimum"]:
        return False
    return not ("maximum" in spec and value > spec["maximum"])


def _validate_call(
    case_id: str,
    call: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    home: dict[str, Any],
) -> None:
    name = call["name"]
    arguments = call["arguments"]
    if name not in catalog:
        raise GenerationError(f"{case_id}: {name} not in production catalog")
    schema = catalog[name]
    properties = schema.get("properties") or {}
    missing = set(schema.get("required") or []) - set(arguments)
    if missing:
        raise GenerationError(f"{case_id}: {name} missing {sorted(missing)}")
    for key, value in arguments.items():
        if key not in properties or not _schema_ok(value, properties[key]):
            raise GenerationError(f"{case_id}: {name} bad argument {key}={value!r}")
    area = arguments.get("area")
    if area is not None and area.casefold() not in {
        item.casefold() for item in home["areas"]
    }:
        raise GenerationError(f"{case_id}: area {area!r} not in household")
    if "name" not in arguments or (name.startswith("intent__") and "Timer" in name):
        return
    owners = _labels(home).get(arguments["name"].casefold(), [])
    if len(owners) != 1:
        raise GenerationError(
            f"{case_id}: name {arguments['name']!r} resolves to {len(owners)} entities"
        )
    domains = arguments.get("domain")
    domains = [domains] if isinstance(domains, str) else domains or []
    if any(domain != owners[0]["domain"] for domain in domains):
        raise GenerationError(f"{case_id}: domain {domains} != {owners[0]['domain']}")


def validate_case(row: dict[str, Any]) -> None:
    case_id = row["id"]
    home = load_home(row["household"])
    expected = row["expected"]
    if row["category"] not in CATEGORY_CAPS:
        raise GenerationError(f"{case_id}: unknown category {row['category']!r}")
    if (
        row["tags"] != ["promotion", "generated"]
        or row["provenance"]["source"] != SOURCE
    ):
        raise GenerationError(f"{case_id}: wrong tags or provenance")
    if row["provenance"]["held_out"] is not True:
        raise GenerationError(f"{case_id}: provenance.held_out must be true")
    catalog = {
        tool["function"]["name"]: tool["function"].get("parameters") or {}
        for tool in production_catalog(home)
    }
    for call in expected["calls"]:
        _validate_call(case_id, call, catalog, home)
    response_type = expected["response_type"]
    if response_type in {"action", "status"} and not expected["calls"]:
        raise GenerationError(f"{case_id}: {response_type} without calls")
    if response_type == "clarification":
        candidates = (expected["clarification"] or {}).get("candidates") or []
        labels = _labels(home)
        if len(candidates) < 2 or any(
            len(labels.get(candidate.casefold(), [])) != 1 for candidate in candidates
        ):
            raise GenerationError(f"{case_id}: clarify needs 2+ real candidates")
    if response_type == "refusal":
        unavailable = row["unavailable"] or {}
        if unavailable.get("kind") not in {"device_absent", "script_absent"}:
            raise GenerationError(f"{case_id}: unavailable kind must be absent")
        mentioned = _mentions_home(home, row["utterance"]) + _mentions_home(
            home, unavailable["capability"]
        )
        if mentioned:
            raise GenerationError(f"{case_id}: unavailable mentions {mentioned}")
    for entity_name in expected["spoken_aliases"]:
        if len(_labels(home).get(entity_name.casefold(), [])) != 1:
            raise GenerationError(f"{case_id}: spoken alias owner {entity_name!r}")


def check_composition(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    low, high = TOTAL_BOUNDS
    if not low <= total <= high:
        raise GenerationError(f"total {total} outside {TOTAL_BOUNDS}")
    counts = Counter(row["category"] for row in rows)
    missing = set(CATEGORIES) - set(counts)
    if missing:
        raise GenerationError(f"missing categories {sorted(missing)}")
    for category, (share_low, share_high) in SHARE_BOUNDS.items():
        share = counts[category] / total
        if not share_low <= share <= share_high:
            raise GenerationError(f"{category} share {share:.3f} out of bounds")
    if counts["status"] / total > STATUS_MAX_SHARE:
        raise GenerationError("status share above 4%")
    if "exclusion" in counts:
        raise GenerationError("exclusion category is not generated")
    utterances = Counter((row["household"], row["utterance"]) for row in rows)
    duplicates = [key for key, count in utterances.items() if count > 1]
    if duplicates:
        raise GenerationError(f"duplicate utterances {duplicates[:3]}")
    answers = Counter(answer_key(row) for row in rows)
    repeated = [key[0] for key, count in answers.items() if count > 1]
    if repeated:
        raise GenerationError(f"cross-household repeats {repeated[:3]}")
    reviewed = reviewed_utterances()
    collisions = [
        row["id"] for row in rows if normalize_utterance(row["utterance"]) in reviewed
    ]
    if collisions:
        raise GenerationError(f"collide with reviewed cases: {collisions[:3]}")
    crowded = near_duplicates(rows)
    if crowded:
        raise GenerationError(f"near-duplicates over limit {crowded[:3]}")


def near_duplicates(rows: list[dict[str, Any]]) -> list[tuple[str, tuple[Any, ...]]]:
    counts: Counter = Counter(
        {
            (household_id(number), item): count
            for number in HOUSEHOLDS
            for item, count in reviewed_near(number).items()
        }
    )
    limits: dict[tuple[str, tuple[Any, ...]], int] = {}
    for row in rows:
        for item, limit in near_keys(row, load_home(row["household"])):
            counts[(row["household"], item)] += 1
            limits[(row["household"], item)] = limit
    return [key for key, limit in limits.items() if counts[key] > limit]


def render(rows: list[dict[str, Any]] | None = None) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows or generate())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.generate_realistic")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = (
            args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        )
        if current != text:
            print(f"{args.output} is out of date", file=sys.stderr)
            return 1
        return 0
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {text.count(chr(10))} cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
