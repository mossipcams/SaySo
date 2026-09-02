#!/usr/bin/env python3
"""Generate SaySo-specific English training examples (Home-LLM V2 JSONL shape)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.schema import ALLOWED_HASS_TOOLS  # noqa: E402

FIXTURES = ROOT / "fixtures"

# Composition mix: 65% single, 15% multi, 10% failure, 5% status, 5% no-tool
MIX = (
    ("single", 65),
    ("multi", 15),
    ("failure", 10),
    ("status", 5),
    ("no_tool", 5),
)

AREAS = (
    "Living Room",
    "Kitchen",
    "Bedroom",
    "Office",
    "Garage",
    "Hallway",
    "Dining Room",
    "Nursery",
)
FLOORS = ("Ground Floor", "First Floor", "Basement", "Attic")
LIGHT_NAMES = ("Ceiling Light", "Desk Lamp", "Reading Light", "Pendant", "Sconce")
CLIMATE_NAMES = ("Thermostat", "Heat Pump", "Radiator")
COVER_NAMES = ("Blinds", "Shades", "Garage Door", "Curtain")
MEDIA_NAMES = ("TV", "Speaker", "Soundbar")
COLORS = ("red", "blue", "warm white", "cool white", "green", "amber")
MODES = ("heat", "cool", "auto", "off")


def _load_tools() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "ha_assist_tools.json").read_text(encoding="utf-8"))


def _base_tools() -> list[dict[str, Any]]:
    """Core Hass* tools plus GetLiveContext for status queries."""
    tools = _load_tools()
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "GetLiveContext",
                "description": "Get current device and area states",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )
    for name in (
        "HassToggle",
        "HassSetPosition",
        "HassSetVolume",
        "HassStartTimer",
        "HassListAddItem",
    ):
        if name not in {t["function"]["name"] for t in tools}:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "area": {"type": "string"},
                                "position": {"type": "integer", "minimum": 0, "maximum": 100},
                                "volume_level": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 100,
                                },
                                "hours": {"type": "integer"},
                                "minutes": {"type": "integer"},
                                "item": {"type": "string"},
                            },
                            "required": [],
                        },
                    },
                }
            )
    return [t for t in tools if t["function"]["name"] in ALLOWED_HASS_TOOLS]


def _meta(rng: random.Random, template: str, phrasing: str, seed: int) -> dict[str, Any]:
    return {
        "metadata": {
            "template_family": template,
            "phrasing_family": phrasing,
            "seed": seed,
        }
    }


def _sys() -> dict[str, Any]:
    return {
        "role": "system",
        "content": [{"type": "text", "text": "You are a voice assistant for Home Assistant."}],
        "train_on_turn": False,
    }


def _user(text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "train_on_turn": False,
    }


def _assistant_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [
            {"function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
        ],
        "train_on_turn": True,
    }


def _assistant_text(text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "train_on_turn": True,
    }


def _tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": [{"name": name, "response": result}],
        "train_on_turn": False,
    }


def _tool_results_combined(pairs: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": [{"name": name, "response": result} for name, result in pairs],
        "train_on_turn": False,
    }


def _entry(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {"messages": [_sys(), *messages], "tools": tools, **meta}


def _pick_area(rng: random.Random) -> str:
    return rng.choice(AREAS)


def _pick_light(rng: random.Random) -> tuple[str, str]:
    area = _pick_area(rng)
    return rng.choice(LIGHT_NAMES), area


def _single_turn_on(rng: random.Random, seed: int) -> dict[str, Any]:
    name, area = _pick_light(rng)
    phrasing = rng.choice(["direct", "polite", "short"])
    templates = {
        "direct": f"Turn on the {name} in the {area}",
        "polite": f"Could you turn on the {name} in the {area}?",
        "short": f"{area} {name} on",
    }
    utterance = templates[phrasing]
    target = rng.choice(
        [
            {"name": name, "area": area, "domain": ["light"]},
            {"area": area, "domain": ["light"]},
            {"name": name, "floor": rng.choice(FLOORS), "domain": ["light"]},
            {"name": name, "device_class": ["light"], "domain": ["light"]},
        ]
    )
    tools = _base_tools()
    return _entry(
        [
            _user(utterance),
            _assistant_tool("HassTurnOn", target),
            _tool_result("HassTurnOn", {"result": "Success"}),
            _assistant_text(f"The {name} in the {area} is on."),
        ],
        tools,
        _meta(rng, "turn_on", phrasing, seed),
    )


def _single_brightness(rng: random.Random, seed: int) -> dict[str, Any]:
    name, area = _pick_light(rng)
    pct = rng.randint(10, 100)
    utterance = f"Set the {name} in the {area} to {pct} percent"
    return _entry(
        [
            _user(utterance),
            _assistant_tool("HassLightSet", {"name": name, "area": area, "brightness": pct}),
            _tool_result("HassLightSet", {"result": "Success"}),
            _assistant_text(f"Set {name} to {pct} percent."),
        ],
        _base_tools(),
        _meta(rng, "light_brightness", "direct", seed),
    )


def _single_color(rng: random.Random, seed: int) -> dict[str, Any]:
    name, area = _pick_light(rng)
    color = rng.choice(COLORS)
    utterance = f"Make the {name} in the {area} {color}"
    return _entry(
        [
            _user(utterance),
            _assistant_tool("HassLightSet", {"name": name, "area": area, "color": color}),
            _tool_result("HassLightSet", {"result": "Success"}),
            _assistant_text(f"The {name} is now {color}."),
        ],
        _base_tools(),
        _meta(rng, "light_color", "direct", seed),
    )


def _single_climate(rng: random.Random, seed: int) -> dict[str, Any]:
    name = rng.choice(CLIMATE_NAMES)
    area = _pick_area(rng)
    temp = rng.randint(18, 26)
    utterance = f"Set the {name} in the {area} to {temp} degrees"
    return _entry(
        [
            _user(utterance),
            _assistant_tool(
                "HassClimateSetTemperature",
                {"name": name, "area": area, "temperature": temp},
            ),
            _tool_result("HassClimateSetTemperature", {"result": "Success"}),
            _assistant_text(f"Set {name} to {temp} degrees."),
        ],
        _base_tools(),
        _meta(rng, "climate_temp", "direct", seed),
    )


def _single_cover(rng: random.Random, seed: int) -> dict[str, Any]:
    name = rng.choice(COVER_NAMES)
    area = _pick_area(rng)
    pos = rng.randint(0, 100)
    utterance = f"Open the {name} in the {area} to {pos} percent"
    return _entry(
        [
            _user(utterance),
            _assistant_tool("HassSetPosition", {"name": name, "area": area, "position": pos}),
            _tool_result("HassSetPosition", {"result": "Success"}),
            _assistant_text(f"Set {name} to {pos} percent."),
        ],
        _base_tools(),
        _meta(rng, "cover_position", "direct", seed),
    )


def _single_volume(rng: random.Random, seed: int) -> dict[str, Any]:
    name = rng.choice(MEDIA_NAMES)
    area = _pick_area(rng)
    vol = rng.randint(0, 100)
    utterance = f"Set {name} volume in the {area} to {vol}"
    return _entry(
        [
            _user(utterance),
            _assistant_tool("HassSetVolume", {"name": name, "area": area, "volume_level": vol}),
            _tool_result("HassSetVolume", {"result": "Success"}),
            _assistant_text(f"Volume set to {vol}."),
        ],
        _base_tools(),
        _meta(rng, "media_volume", "direct", seed),
    )


def _single_timer(rng: random.Random, seed: int) -> dict[str, Any]:
    mins = rng.randint(1, 45)
    utterance = f"Start a timer for {mins} minutes"
    return _entry(
        [
            _user(utterance),
            _assistant_tool("HassStartTimer", {"minutes": mins}),
            _tool_result("HassStartTimer", {"result": "Success"}),
            _assistant_text(f"Timer started for {mins} minutes."),
        ],
        _base_tools(),
        _meta(rng, "timer", "direct", seed),
    )


def _single_todo(rng: random.Random, seed: int) -> dict[str, Any]:
    item = rng.choice(["milk", "batteries", "call plumber", "pick up dry cleaning"])
    utterance = f"Add {item} to my shopping list"
    return _entry(
        [
            _user(utterance),
            _assistant_tool("HassListAddItem", {"item": item}),
            _tool_result("HassListAddItem", {"result": "Success"}),
            _assistant_text(f"Added {item} to your list."),
        ],
        _base_tools(),
        _meta(rng, "todo_add", "direct", seed),
    )


def _multi_action(rng: random.Random, seed: int) -> dict[str, Any]:
    area = _pick_area(rng)
    light = rng.choice(LIGHT_NAMES)
    utterance = f"Turn off the {light} in the {area} and turn on the porch light"
    return _entry(
        [
            _user(utterance),
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "tool_calls": [
                    {
                        "function": {
                            "name": "HassTurnOff",
                            "arguments": json.dumps(
                                {"name": light, "area": area, "domain": ["light"]}
                            ),
                        }
                    },
                    {
                        "function": {
                            "name": "HassTurnOn",
                            "arguments": json.dumps(
                                {"name": "Porch Light", "domain": ["light"]}
                            ),
                        }
                    },
                ],
                "train_on_turn": True,
            },
            _tool_results_combined(
                [
                    ("HassTurnOff", {"result": "Success"}),
                    ("HassTurnOn", {"result": "Success"}),
                ]
            ),
            _assistant_text(f"Turned off {light} and turned on the porch light."),
        ],
        _base_tools(),
        _meta(rng, "multi_off_on", "direct", seed),
    )


def _follow_up_off(rng: random.Random, seed: int) -> dict[str, Any]:
    name, area = _pick_light(rng)
    return _entry(
        [
            _user(f"Turn on the {name} in the {area}"),
            _assistant_tool("HassTurnOn", {"name": name, "area": area, "domain": ["light"]}),
            _tool_result("HassTurnOn", {"result": "Success"}),
            _assistant_text(f"The {name} is on."),
            _user("Turn it off"),
            _assistant_tool("HassTurnOff", {"name": name, "area": area, "domain": ["light"]}),
            _tool_result("HassTurnOff", {"result": "Success"}),
            _assistant_text(f"Turned off the {name}."),
        ],
        _base_tools(),
        _meta(rng, "follow_up_off", "contextual", seed),
    )


def _similar_names(rng: random.Random, seed: int) -> dict[str, Any]:
    area = _pick_area(rng)
    utterance = f"Turn on the desk lamp in the {area}, not the floor lamp"
    return _entry(
        [
            _user(utterance),
            _assistant_tool(
                "HassTurnOn",
                {"name": "Desk Lamp", "area": area, "domain": ["light"]},
            ),
            _tool_result("HassTurnOn", {"result": "Success"}),
            _assistant_text("Turned on the desk lamp."),
        ],
        _base_tools(),
        _meta(rng, "similar_names", "disambiguation", seed),
    )


def _partial_failure(rng: random.Random, seed: int) -> dict[str, Any]:
    area = _pick_area(rng)
    utterance = f"Turn on the kitchen light and the backyard floodlight in the {area}"
    return _entry(
        [
            _user(utterance),
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "tool_calls": [
                    {
                        "function": {
                            "name": "HassTurnOn",
                            "arguments": json.dumps(
                                {"name": "Kitchen Light", "domain": ["light"]}
                            ),
                        }
                    },
                    {
                        "function": {
                            "name": "HassTurnOn",
                            "arguments": json.dumps(
                                {"name": "Backyard Floodlight", "domain": ["light"]}
                            ),
                        }
                    },
                ],
                "train_on_turn": True,
            },
            _tool_results_combined(
                [
                    ("HassTurnOn", {"result": "Success"}),
                    ("HassTurnOn", {"result": "Failed", "error": "Entity not found"}),
                ]
            ),
            _assistant_text("Kitchen light is on, but I couldn't find the backyard floodlight."),
        ],
        _base_tools(),
        _meta(rng, "partial_failure", "failure", seed),
    )


def _unsupported(rng: random.Random, seed: int) -> dict[str, Any]:
    utterance = rng.choice(
        [
            "What's the weather tomorrow?",
            "Send a text to Mom",
            "Order pizza",
        ]
    )
    return _entry(
        [
            _user(utterance),
            _assistant_text("I can't help with that from here."),
        ],
        _base_tools(),
        _meta(rng, "unsupported", "refusal", seed),
    )


def _no_match(rng: random.Random, seed: int) -> dict[str, Any]:
    utterance = "Turn on the hologram projector in the lab"
    return _entry(
        [
            _user(utterance),
            _assistant_text("I couldn't find a device matching that."),
        ],
        _base_tools(),
        _meta(rng, "no_match", "failure", seed),
    )


def _status_query(rng: random.Random, seed: int) -> dict[str, Any]:
    area = _pick_area(rng)
    utterance = f"What's on in the {area}?"
    return _entry(
        [
            _user(utterance),
            _assistant_tool("GetLiveContext", {}),
            _tool_result("GetLiveContext", {"areas": {area: {"lights": ["Ceiling Light"]}}}),
            _assistant_text(f"The ceiling light is on in the {area}."),
        ],
        _base_tools(),
        _meta(rng, "status_live_context", "status", seed),
    )


def _no_tool_chat(rng: random.Random, seed: int) -> dict[str, Any]:
    utterance = rng.choice(["Hello", "Thanks", "Good night"])
    reply = rng.choice(["Hi there.", "You're welcome.", "Good night."])
    return _entry(
        [_user(utterance), _assistant_text(reply)],
        _base_tools(),
        _meta(rng, "no_tool_chat", "conversational", seed),
    )


SINGLE_BUILDERS = (
    _single_turn_on,
    _single_brightness,
    _single_color,
    _single_climate,
    _single_cover,
    _single_volume,
    _single_timer,
    _single_todo,
    _similar_names,
    _follow_up_off,
)

MULTI_BUILDERS = (_multi_action,)

FAILURE_BUILDERS = (_partial_failure, _unsupported, _no_match)

STATUS_BUILDERS = (_status_query,)

NO_TOOL_BUILDERS = (_no_tool_chat,)


def _category_builders(category: str) -> tuple[Any, ...]:
    return {
        "single": SINGLE_BUILDERS,
        "multi": MULTI_BUILDERS,
        "failure": FAILURE_BUILDERS,
        "status": STATUS_BUILDERS,
        "no_tool": NO_TOOL_BUILDERS,
    }[category]


def generate_examples(count: int, *, seed: int = 42) -> Iterator[dict[str, Any]]:
    """Yield Home-LLM V2 shaped examples with composition mix."""
    rng = random.Random(seed)
    weights = [weight for _cat, weight in MIX]
    categories = [cat for cat, _weight in MIX]

    for idx in range(count):
        category = rng.choices(categories, weights=weights, k=1)[0]
        builders = _category_builders(category)
        builder = rng.choice(builders)
        yield builder(rng, seed + idx)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=ROOT / "datasets" / "sayso_generated.jsonl")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for example in generate_examples(args.count, seed=args.seed):
            handle.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Wrote {args.count} examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
