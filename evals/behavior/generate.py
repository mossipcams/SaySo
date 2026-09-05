"""Generate behavioral voice-command cases from the pinned SaySo tool schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "sayso-tool-schema-v1.json"
CASES_PATH = Path(__file__).resolve().parent / "cases.yaml"

AREAS = (
    "living room",
    "kitchen",
    "bedroom",
    "bathroom",
    "garage",
    "hallway",
    "office",
    "dining room",
    "basement",
    "patio",
    "entryway",
    "laundry room",
    "nursery",
    "guest room",
    "sunroom",
)

FLOORS = ("upstairs", "downstairs", "first floor", "second floor")

LIGHT_NAMES = (
    "ceiling light",
    "desk lamp",
    "floor lamp",
    "chandelier",
    "pendant light",
    "reading light",
    "night light",
    "track lights",
    "under cabinet lights",
    "vanity light",
)

FAN_NAMES = ("ceiling fan", "bedroom fan", "office fan", "patio fan", "bathroom fan")

BRIGHTNESS_LEVELS = (15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)

FAN_SPEEDS = (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)

COLOR_NAMES = ("red", "blue", "green", "warm white", "cool white", "purple", "orange", "pink")

TEMPERATURES = (2700, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500)


def _load_schema_tool_names() -> frozenset[str]:
    payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    names: list[str] = []
    for entry in payload.get("tools", []):
        function = entry.get("function", {})
        name = function.get("name")
        if isinstance(name, str):
            names.append(name)
    return frozenset(names)


def _case(
    case_id: str,
    *,
    category: str,
    utterance: str,
    description: str,
    tool_calls: list[dict[str, Any]],
    checks: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "scenario": utterance,
        "description": description,
        "expect": {"tool_calls": tool_calls},
        "checks": checks or ["tool_name", "tool_args"],
    }


def _turn_on(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassTurnOn", "arguments": arguments}


def _turn_off(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassTurnOff", "arguments": arguments}


def _light_set(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassLightSet", "arguments": arguments}


def _fan_speed(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassFanSetSpeed", "arguments": arguments}


def _cancel_timers(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": "HassCancelAllTimers", "arguments": arguments or {}}


def _live_context(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"name": "GetLiveContext", "arguments": arguments}


def _date_time() -> dict[str, Any]:
    return {"name": "GetDateTime", "arguments": {}}


def generate_cases() -> list[dict[str, Any]]:
    tool_names = _load_schema_tool_names()
    required = {
        "GetDateTime",
        "GetLiveContext",
        "HassCancelAllTimers",
        "HassFanSetSpeed",
        "HassLightSet",
        "HassTurnOff",
        "HassTurnOn",
    }
    missing = required - tool_names
    if missing:
        raise RuntimeError(f"schema missing expected tools: {sorted(missing)}")

    cases: list[dict[str, Any]] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"sayso-behavior-v1-{counter:03d}"

    def add(case: dict[str, Any]) -> None:
        cases.append(case)

    turn_on_phrases = (
        "turn on the {target}",
        "switch on the {target}",
        "please turn on the {target}",
        "activate the {target}",
        "enable the {target}",
        "power on the {target}",
        "start the {target}",
        "flip on the {target}",
    )
    turn_off_phrases = (
        "turn off the {target}",
        "switch off the {target}",
        "please turn off the {target}",
        "deactivate the {target}",
        "disable the {target}",
        "power off the {target}",
        "shut off the {target}",
        "flip off the {target}",
    )

    # Lights on with area (30)
    for area, phrase in zip(AREAS[:15], turn_on_phrases * 2):
        target = f"{area} lights"
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=phrase.format(target=target),
                description="Turn on lights in a named area",
                tool_calls=[_turn_on({"area": area, "domain": ["light"]})],
            )
        )

    # Lights off with area (30)
    for area, phrase in zip(AREAS[:15], turn_off_phrases * 2):
        target = f"{area} lights"
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=phrase.format(target=target),
                description="Turn off lights in a named area",
                tool_calls=[_turn_off({"area": area, "domain": ["light"]})],
            )
        )

    # Lights on by name without area (20)
    for name, phrase in zip(LIGHT_NAMES[:10], turn_on_phrases[:10]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=phrase.format(target=name),
                description="Turn on a named light without area",
                tool_calls=[_turn_on({"name": name, "domain": ["light"]})],
            )
        )
    for name, phrase in zip(LIGHT_NAMES[10:], turn_on_phrases[:5]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=phrase.format(target=name),
                description="Turn on a named light without area",
                tool_calls=[_turn_on({"name": name, "domain": ["light"]})],
            )
        )

    # Lights off by name without area (20)
    for name, phrase in zip(LIGHT_NAMES[:10], turn_off_phrases[:10]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=phrase.format(target=name),
                description="Turn off a named light without area",
                tool_calls=[_turn_off({"name": name, "domain": ["light"]})],
            )
        )
    for name, phrase in zip(LIGHT_NAMES[10:], turn_off_phrases[:5]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=phrase.format(target=name),
                description="Turn off a named light without area",
                tool_calls=[_turn_off({"name": name, "domain": ["light"]})],
            )
        )

    # Global lights on/off without area (10)
    global_light_cmds = (
        ("turn on the lights", _turn_on({"domain": ["light"]})),
        ("switch on all lights", _turn_on({"domain": ["light"]})),
        ("lights on", _turn_on({"domain": ["light"]})),
        ("turn every light on", _turn_on({"domain": ["light"]})),
        ("please turn on all the lights", _turn_on({"domain": ["light"]})),
        ("turn off the lights", _turn_off({"domain": ["light"]})),
        ("switch off all lights", _turn_off({"domain": ["light"]})),
        ("lights off", _turn_off({"domain": ["light"]})),
        ("turn every light off", _turn_off({"domain": ["light"]})),
        ("please turn off all the lights", _turn_off({"domain": ["light"]})),
    )
    for utterance, tool_call in global_light_cmds:
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=utterance,
                description="Control all lights without a specific area",
                tool_calls=[tool_call],
            )
        )

    # TV / media play and pause (24)
    media_cmds = (
        ("turn on the tv", _turn_on({"device_class": ["tv"]})),
        ("switch on the television", _turn_on({"device_class": ["tv"]})),
        ("play the tv", _turn_on({"device_class": ["tv"]})),
        ("start the tv", _turn_on({"device_class": ["tv"]})),
        ("turn on the living room tv", _turn_on({"name": "living room tv", "device_class": ["tv"]})),
        ("play the living room television", _turn_on({"name": "living room tv", "device_class": ["tv"]})),
        ("turn on the bedroom tv", _turn_on({"name": "bedroom tv", "device_class": ["tv"]})),
        ("start the projector", _turn_on({"name": "projector", "device_class": ["projector"]})),
        ("turn on the soundbar", _turn_on({"name": "soundbar", "device_class": ["speaker"]})),
        ("play music on the receiver", _turn_on({"name": "receiver", "device_class": ["receiver"]})),
        ("turn off the tv", _turn_off({"device_class": ["tv"]})),
        ("pause the tv", _turn_off({"device_class": ["tv"]})),
        ("stop the television", _turn_off({"device_class": ["tv"]})),
        ("switch off the tv", _turn_off({"device_class": ["tv"]})),
        ("turn off the living room tv", _turn_off({"name": "living room tv", "device_class": ["tv"]})),
        ("pause the living room television", _turn_off({"name": "living room tv", "device_class": ["tv"]})),
        ("turn off the bedroom tv", _turn_off({"name": "bedroom tv", "device_class": ["tv"]})),
        ("stop the projector", _turn_off({"name": "projector", "device_class": ["projector"]})),
        ("turn off the soundbar", _turn_off({"name": "soundbar", "device_class": ["speaker"]})),
        ("turn off the receiver", _turn_off({"name": "receiver", "device_class": ["receiver"]})),
        ("turn on media in the kitchen", _turn_on({"area": "kitchen", "domain": ["media_player"]})),
        ("pause kitchen media", _turn_off({"area": "kitchen", "device_class": ["tv"]})),
        ("play the office speaker", _turn_on({"area": "office", "device_class": ["speaker"]})),
        ("stop the patio speaker", _turn_off({"area": "patio", "device_class": ["speaker"]})),
    )
    for utterance, tool_call in media_cmds:
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=utterance,
                description="Control TV or media devices",
                tool_calls=[tool_call],
            )
        )

    # Brightness (35)
    brightness_cmds: list[tuple[str, dict[str, Any]]] = []
    for area, level in zip(AREAS[:10], BRIGHTNESS_LEVELS[:10]):
        brightness_cmds.append(
            (
                f"set {area} lights to {level} percent",
                {"area": area, "domain": ["light"], "brightness": level},
            )
        )
    for name, level in zip(LIGHT_NAMES[:10], BRIGHTNESS_LEVELS[10:20]):
        brightness_cmds.append(
            (
                f"dim the {name} to {level} percent",
                {"name": name, "domain": ["light"], "brightness": level},
            )
        )
    for area, level in zip(AREAS[10:15], BRIGHTNESS_LEVELS[20:25]):
        brightness_cmds.append(
            (
                f"brighten the {area} lights to {level}",
                {"area": area, "domain": ["light"], "brightness": level},
            )
        )
    for utterance, arguments in brightness_cmds:
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=utterance,
                description="Set light brightness",
                tool_calls=[_light_set(arguments)],
            )
        )

    # Color and temperature (15)
    for color, area in zip(COLOR_NAMES, AREAS[:8]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=f"set the {area} lights to {color}",
                description="Set light color in an area",
                tool_calls=[_light_set({"area": area, "domain": ["light"], "color": color})],
            )
        )
    for temp, name in zip(TEMPERATURES, LIGHT_NAMES[:7]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=f"set {name} color temperature to {temp}",
                description="Set light color temperature",
                tool_calls=[_light_set({"name": name, "domain": ["light"], "temperature": temp})],
            )
        )

    # Covers open/close (24)
    cover_cmds = (
        ("open the blinds", _turn_on({"name": "blinds", "device_class": ["blind"]})),
        ("close the blinds", _turn_off({"name": "blinds", "device_class": ["blind"]})),
        ("open the curtains", _turn_on({"name": "curtains", "device_class": ["curtain"]})),
        ("close the curtains", _turn_off({"name": "curtains", "device_class": ["curtain"]})),
        ("open the bedroom shades", _turn_on({"name": "bedroom shades", "device_class": ["shade"]})),
        ("close the bedroom shades", _turn_off({"name": "bedroom shades", "device_class": ["shade"]})),
        ("open the garage door", _turn_on({"name": "garage door", "device_class": ["garage"]})),
        ("close the garage door", _turn_off({"name": "garage door", "device_class": ["garage"]})),
        ("open the front gate", _turn_on({"name": "front gate", "device_class": ["gate"]})),
        ("close the front gate", _turn_off({"name": "front gate", "device_class": ["gate"]})),
        ("open the patio awning", _turn_on({"name": "patio awning", "device_class": ["awning"]})),
        ("close the patio awning", _turn_off({"name": "patio awning", "device_class": ["awning"]})),
        ("open the kitchen shutters", _turn_on({"name": "kitchen shutters", "device_class": ["shutter"]})),
        ("close the kitchen shutters", _turn_off({"name": "kitchen shutters", "device_class": ["shutter"]})),
        ("open living room blinds", _turn_on({"area": "living room", "device_class": ["blind"]})),
        ("close living room blinds", _turn_off({"area": "living room", "device_class": ["blind"]})),
        ("open the office window shade", _turn_on({"name": "office window shade", "device_class": ["shade"]})),
        ("close the office window shade", _turn_off({"name": "office window shade", "device_class": ["shade"]})),
        ("raise the dining room curtains", _turn_on({"area": "dining room", "device_class": ["curtain"]})),
        ("lower the dining room curtains", _turn_off({"area": "dining room", "device_class": ["curtain"]})),
        ("open the basement window", _turn_on({"area": "basement", "device_class": ["window"]})),
        ("close the basement window", _turn_off({"area": "basement", "device_class": ["window"]})),
        ("open the patio shade", _turn_on({"area": "patio", "device_class": ["shade"]})),
        ("close the patio shade", _turn_off({"area": "patio", "device_class": ["shade"]})),
    )
    for utterance, tool_call in cover_cmds:
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=utterance,
                description="Open or close covers",
                tool_calls=[tool_call],
            )
        )

    # Locks (12)
    lock_cmds = (
        ("lock the front door", _turn_on({"name": "front door lock", "domain": ["lock"]})),
        ("unlock the front door", _turn_off({"name": "front door lock", "domain": ["lock"]})),
        ("lock the garage", _turn_on({"name": "garage lock", "domain": ["lock"]})),
        ("unlock the garage", _turn_off({"name": "garage lock", "domain": ["lock"]})),
        ("lock the back door", _turn_on({"name": "back door", "domain": ["lock"]})),
        ("unlock the back door", _turn_off({"name": "back door", "domain": ["lock"]})),
        ("lock the side gate", _turn_on({"name": "side gate", "domain": ["lock"]})),
        ("unlock the side gate", _turn_off({"name": "side gate", "domain": ["lock"]})),
        ("lock the kitchen door", _turn_on({"area": "kitchen", "domain": ["lock"]})),
        ("unlock the kitchen door", _turn_off({"area": "kitchen", "domain": ["lock"]})),
        ("secure the entryway", _turn_on({"area": "entryway", "domain": ["lock"]})),
        ("unsecure the entryway", _turn_off({"area": "entryway", "domain": ["lock"]})),
    )
    for utterance, tool_call in lock_cmds:
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=utterance,
                description="Lock or unlock doors",
                tool_calls=[tool_call],
            )
        )

    # Fans on/off and speed (30)
    for name, phrase in zip(FAN_NAMES, turn_on_phrases[:5]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=phrase.format(target=name),
                description="Turn on a fan",
                tool_calls=[_turn_on({"name": name, "domain": ["fan"]})],
            )
        )
    for name, phrase in zip(FAN_NAMES, turn_off_phrases[:5]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=phrase.format(target=name),
                description="Turn off a fan",
                tool_calls=[_turn_off({"name": name, "domain": ["fan"]})],
            )
        )
    for area, speed in zip(AREAS[:10], FAN_SPEEDS[:10]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=f"set the {area} fan to {speed} percent",
                description="Set fan speed in an area",
                tool_calls=[_fan_speed({"area": area, "domain": ["fan"], "percentage": speed})],
            )
        )
    for name, speed in zip(FAN_NAMES, FAN_SPEEDS[10:15]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=f"set {name} speed to {speed}",
                description="Set named fan speed",
                tool_calls=[_fan_speed({"name": name, "domain": ["fan"], "percentage": speed})],
            )
        )
    for floor, speed in zip(FLOORS, FAN_SPEEDS[15:19]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=f"set {floor} fans to {speed} percent",
                description="Set fan speed on a floor",
                tool_calls=[_fan_speed({"floor": floor, "domain": ["fan"], "percentage": speed})],
            )
        )

    # Vacuums and switches (16)
    vacuum_cmds = (
        ("start the robot vacuum", _turn_on({"name": "robot vacuum", "domain": ["vacuum"]})),
        ("stop the robot vacuum", _turn_off({"name": "robot vacuum", "domain": ["vacuum"]})),
        ("start the roomba", _turn_on({"name": "roomba", "domain": ["vacuum"]})),
        ("send the roomba home", _turn_off({"name": "roomba", "domain": ["vacuum"]})),
        ("vacuum the kitchen", _turn_on({"area": "kitchen", "domain": ["vacuum"]})),
        ("stop vacuuming the kitchen", _turn_off({"area": "kitchen", "domain": ["vacuum"]})),
        ("vacuum upstairs", _turn_on({"name": "upstairs vacuum", "domain": ["vacuum"]})),
        ("stop the upstairs vacuum", _turn_off({"name": "upstairs vacuum", "domain": ["vacuum"]})),
    )
    for utterance, tool_call in vacuum_cmds:
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=utterance,
                description="Control vacuum cleaners",
                tool_calls=[tool_call],
            )
        )
    switch_cmds = (
        ("turn on the coffee maker", _turn_on({"name": "coffee maker", "device_class": ["switch"]})),
        ("turn off the coffee maker", _turn_off({"name": "coffee maker", "device_class": ["switch"]})),
        ("turn on the space heater", _turn_on({"name": "space heater", "device_class": ["outlet"]})),
        ("turn off the space heater", _turn_off({"name": "space heater", "device_class": ["outlet"]})),
        ("turn on the garage outlet", _turn_on({"name": "garage outlet", "device_class": ["outlet"]})),
        ("turn off the garage outlet", _turn_off({"name": "garage outlet", "device_class": ["outlet"]})),
        ("turn on the holiday lights", _turn_on({"name": "holiday lights", "device_class": ["switch"]})),
        ("turn off the holiday lights", _turn_off({"name": "holiday lights", "device_class": ["switch"]})),
    )
    for utterance, tool_call in switch_cmds:
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=utterance,
                description="Control switched outlets and appliances",
                tool_calls=[tool_call],
            )
        )

    # Climate via domain (8) - schema allows arbitrary domain strings on turn on/off
    climate_cmds = (
        ("turn on the heat", _turn_on({"domain": ["climate"]})),
        ("turn off the heat", _turn_off({"domain": ["climate"]})),
        ("turn on the air conditioning", _turn_on({"domain": ["climate"]})),
        ("turn off the air conditioning", _turn_off({"domain": ["climate"]})),
        ("set heat in the living room", _turn_on({"area": "living room", "domain": ["climate"]})),
        ("turn off heat in the bedroom", _turn_off({"area": "bedroom", "domain": ["climate"]})),
        ("cool down the office", _turn_on({"area": "office", "domain": ["climate"]})),
        ("stop cooling the office", _turn_off({"area": "office", "domain": ["climate"]})),
    )
    for utterance, tool_call in climate_cmds:
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=utterance,
                description="Control climate devices",
                tool_calls=[tool_call],
            )
        )

    # Timers (12)
    timer_cmds = (
        ("cancel all timers", _cancel_timers()),
        ("clear every timer", _cancel_timers()),
        ("stop all timers", _cancel_timers()),
        ("cancel kitchen timers", _cancel_timers({"area": "kitchen"})),
        ("clear bedroom timers", _cancel_timers({"area": "bedroom"})),
        ("cancel living room timers", _cancel_timers({"area": "living room"})),
        ("stop office timers", _cancel_timers({"area": "office"})),
        ("cancel timers in the garage", _cancel_timers({"area": "garage"})),
        ("clear timers in the nursery", _cancel_timers({"area": "nursery"})),
        ("cancel bathroom timers", _cancel_timers({"area": "bathroom"})),
        ("stop laundry room timers", _cancel_timers({"area": "laundry room"})),
        ("cancel patio timers", _cancel_timers({"area": "patio"})),
    )
    for utterance, tool_call in timer_cmds:
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=utterance,
                description="Cancel household timers",
                tool_calls=[tool_call],
            )
        )

    # Live context queries (24)
    query_cmds = (
        ("what lights are on", {"domain": "light"}),
        ("are any lights on in the kitchen", {"area": "kitchen", "domain": "light"}),
        ("is the bedroom light on", {"area": "bedroom", "domain": "light"}),
        ("what is the garage door state", {"area": "garage", "name": "garage door"}),
        ("is the front door locked", {"name": "front door lock", "domain": "lock"}),
        ("what is the temperature in the living room", {"area": "living room", "domain": "sensor"}),
        ("are the blinds open in the office", {"area": "office", "name": "blinds", "domain": "cover"}),
        ("what fans are running", {"domain": "fan"}),
        ("is the tv on", {"name": "tv", "domain": "media_player"}),
        ("what media is playing in the kitchen", {"area": "kitchen", "domain": "media_player"}),
        ("is the robot vacuum running", {"name": "robot vacuum", "domain": "vacuum"}),
        ("what is the hallway humidity", {"area": "hallway", "domain": "sensor"}),
        ("are any upstairs windows open", {"name": "upstairs window", "domain": "cover"}),
        ("what covers are open in the dining room", {"area": "dining room", "domain": "cover"}),
        ("is the patio shade closed", {"area": "patio", "name": "shade", "domain": "cover"}),
        ("what switches are on", {"domain": "switch"}),
        ("is the nursery night light on", {"area": "nursery", "name": "night light", "domain": "light"}),
        ("what is the office temperature", {"area": "office", "domain": "climate"}),
        ("are the curtains closed in the guest room", {"area": "guest room", "name": "curtains", "domain": "cover"}),
        ("is the basement dehumidifier on", {"area": "basement", "domain": "humidifier"}),
        ("what sensors are in the laundry room", {"area": "laundry room", "domain": "sensor"}),
        ("is the sunroom fan on", {"area": "sunroom", "domain": "fan"}),
        ("what locks are engaged", {"domain": "lock"}),
        ("is the entryway light on", {"area": "entryway", "domain": "light"}),
    )
    for utterance, arguments in query_cmds:
        add(
            _case(
                next_id(),
                category="query",
                utterance=utterance,
                description="Query current device or area state",
                tool_calls=[_live_context(arguments)],
            )
        )

    # Date/time queries (6)
    datetime_cmds = (
        "what time is it",
        "tell me the current time",
        "what is today's date",
        "what day is it",
        "give me the date and time",
        "what is the time right now",
    )
    for utterance in datetime_cmds:
        add(
            _case(
                next_id(),
                category="query",
                utterance=utterance,
                description="Ask for current date or time",
                tool_calls=[_date_time()],
            )
        )

    # Floor-scoped lighting (12)
    for floor, phrase in zip(FLOORS, turn_on_phrases[:4]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=phrase.format(target=f"{floor} lights"),
                description="Turn on lights on a floor",
                tool_calls=[_turn_on({"floor": floor, "domain": ["light"]})],
            )
        )
    for floor, phrase in zip(FLOORS, turn_off_phrases[:4]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=phrase.format(target=f"{floor} lights"),
                description="Turn off lights on a floor",
                tool_calls=[_turn_off({"floor": floor, "domain": ["light"]})],
            )
        )
    for floor, level in zip(FLOORS, BRIGHTNESS_LEVELS[25:29]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=f"set {floor} lights to {level} percent brightness",
                description="Set brightness for lights on a floor",
                tool_calls=[_light_set({"floor": floor, "domain": ["light"], "brightness": level})],
            )
        )
    for floor, color in zip(FLOORS, COLOR_NAMES[4:8]):
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=f"make {floor} lights {color}",
                description="Set light color on a floor",
                tool_calls=[_light_set({"floor": floor, "domain": ["light"], "color": color})],
            )
        )

    # Named area + device combinations (20)
    area_device_cmds = (
        ("turn on the kitchen ceiling light", _turn_on({"area": "kitchen", "name": "ceiling light", "domain": ["light"]})),
        ("turn off the bedroom desk lamp", _turn_off({"area": "bedroom", "name": "desk lamp", "domain": ["light"]})),
        ("switch on the office floor lamp", _turn_on({"area": "office", "name": "floor lamp", "domain": ["light"]})),
        ("switch off the nursery night light", _turn_off({"area": "nursery", "name": "night light", "domain": ["light"]})),
        ("turn on the bathroom vanity light", _turn_on({"area": "bathroom", "name": "vanity light", "domain": ["light"]})),
        ("turn off the hallway track lights", _turn_off({"area": "hallway", "name": "track lights", "domain": ["light"]})),
        ("start the laundry room fan", _turn_on({"area": "laundry room", "domain": ["fan"]})),
        ("stop the guest room fan", _turn_off({"area": "guest room", "domain": ["fan"]})),
        ("open the bedroom curtains", _turn_on({"area": "bedroom", "name": "curtains", "device_class": ["curtain"]})),
        ("close the kitchen blinds", _turn_off({"area": "kitchen", "name": "blinds", "device_class": ["blind"]})),
        ("lock the garage door", _turn_on({"name": "garage door lock", "domain": ["lock"]})),
        ("unlock the garage door", _turn_off({"name": "garage door lock", "domain": ["lock"]})),
        ("turn on the patio string lights", _turn_on({"area": "patio", "name": "holiday lights", "domain": ["light"]})),
        ("turn off the dining room chandelier", _turn_off({"area": "dining room", "name": "chandelier", "domain": ["light"]})),
        ("play the basement tv", _turn_on({"area": "basement", "device_class": ["tv"]})),
        ("pause the sunroom tv", _turn_off({"area": "sunroom", "device_class": ["tv"]})),
        ("start vacuuming the hallway", _turn_on({"area": "hallway", "domain": ["vacuum"]})),
        ("stop the entryway vacuum", _turn_off({"area": "entryway", "domain": ["vacuum"]})),
        ("turn on the sunroom reading light", _turn_on({"area": "sunroom", "name": "reading light", "domain": ["light"]})),
        ("turn off the guest room pendant light", _turn_off({"area": "guest room", "name": "pendant light", "domain": ["light"]})),
    )
    for utterance, tool_call in area_device_cmds:
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=utterance,
                description="Control a named device within an area",
                tool_calls=[tool_call],
            )
        )

    # Additional live-context queries (12)
    extra_query_cmds = (
        ("check if the kitchen lights are on", {"area": "kitchen", "domain": "light"}),
        ("is the office desk lamp on", {"area": "office", "name": "desk lamp", "domain": "light"}),
        ("what is the bedroom fan speed", {"area": "bedroom", "domain": "fan"}),
        ("are any timers running in the kitchen", {"area": "kitchen"}),
        ("is the garage door open", {"area": "garage", "name": "garage door"}),
        ("what is the patio temperature", {"area": "patio", "domain": "sensor"}),
        ("is the laundry room humidifier on", {"area": "laundry room", "domain": "humidifier"}),
        ("what upstairs covers are open", {"name": "upstairs cover", "domain": "cover"}),
        ("is the guest room tv on", {"area": "guest room", "name": "tv", "domain": "media_player"}),
        ("what is the basement humidity", {"area": "basement", "domain": "sensor"}),
        ("are the nursery lights on", {"area": "nursery", "domain": "light"}),
        ("is the dining room chandelier on", {"area": "dining room", "name": "chandelier", "domain": "light"}),
    )
    for utterance, arguments in extra_query_cmds:
        add(
            _case(
                next_id(),
                category="query",
                utterance=utterance,
                description="Query device state in a specific area",
                tool_calls=[_live_context(arguments)],
            )
        )

    # Additional fan speed and brightness cases (12)
    extra_fan_brightness = (
        ("set the hallway fan to 33 percent", _fan_speed({"area": "hallway", "domain": ["fan"], "percentage": 33})),
        ("set the nursery fan to 66 percent", _fan_speed({"area": "nursery", "domain": ["fan"], "percentage": 66})),
        ("slow the patio fan to 12 percent", _fan_speed({"area": "patio", "domain": ["fan"], "percentage": 12})),
        ("boost the garage fan to 88 percent", _fan_speed({"area": "garage", "domain": ["fan"], "percentage": 88})),
        ("set guest room fan to 44 percent", _fan_speed({"area": "guest room", "domain": ["fan"], "percentage": 44})),
        ("set laundry room fan to 77 percent", _fan_speed({"area": "laundry room", "domain": ["fan"], "percentage": 77})),
        ("dim the hallway lights to 22 percent", _light_set({"area": "hallway", "domain": ["light"], "brightness": 22})),
        ("brighten the entryway lights to 88 percent", _light_set({"area": "entryway", "domain": ["light"], "brightness": 88})),
        ("set the sunroom lights to 33 percent", _light_set({"area": "sunroom", "domain": ["light"], "brightness": 33})),
        ("set nursery lights to warm white", _light_set({"area": "nursery", "domain": ["light"], "color": "warm white"})),
        ("make the guest room lights cool white", _light_set({"area": "guest room", "domain": ["light"], "color": "cool white"})),
        ("set the laundry room light temperature to 4000", _light_set({"area": "laundry room", "domain": ["light"], "temperature": 4000})),
    )
    for utterance, tool_call in extra_fan_brightness:
        add(
            _case(
                next_id(),
                category="core_control",
                utterance=utterance,
                description="Adjust fan speed or light appearance",
                tool_calls=[tool_call],
            )
        )

    if len(cases) != 300:
        raise RuntimeError(f"expected 300 generated cases, got {len(cases)}")

    utterances = [case["scenario"] for case in cases]
    if len(utterances) != len(set(utterances)):
        duplicates = {u for u in utterances if utterances.count(u) > 1}
        raise RuntimeError(f"duplicate utterances: {sorted(duplicates)[:5]}")

    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case ids generated")

    for case in cases:
        for tool_call in case["expect"]["tool_calls"]:
            if tool_call["name"] not in tool_names:
                raise RuntimeError(f"unknown tool in case {case['id']}: {tool_call['name']}")

    return cases


def write_cases(path: Path | None = None) -> Path:
    target = path or CASES_PATH
    payload = {"version": 1, "cases": generate_cases()}
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def main() -> None:
    target = write_cases()
    count = len(yaml.safe_load(target.read_text(encoding="utf-8"))["cases"])
    print(f"wrote {count} cases to {target}")


if __name__ == "__main__":
    main()
