"""Locked gold and shadow quality eval rows for the 40k v3 synthetic contract."""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from adapters.schema import tool_schema_map, validate_tool_arguments, v2_openai_tools  # noqa: E402
from build_synthetic_dataset import render_example  # noqa: E402
from generators.utterances import _phrase_for_call, expand_utterance, request_seed_from_spec  # noqa: E402
from evals.metrics import (  # noqa: E402
    extract_assistant_tool_calls,
    parse_tool_arguments,
    score_expected_vs_actual,
)
from evals.recipe_lock import quality_eval_user_prompts as recipe_lock_prompts  # noqa: E402

_BANNED = re.compile(r"evals/cases/|<tool_call>|tool_call_start", re.I)
_GOLD_AREAS = (
    "Great Room",
    "Family Room",
    "Home Theater",
    "Rec Room",
    "Study",
    "Dining Room",
    "Sunroom",
    "Mudroom",
    "Guest Suite",
    "Library",
)
_SHADOW_AREAS = (
    "Annex",
    "Atrium",
    "Conservatory",
    "Solarium",
    "Studio",
    "Terrace",
    "Balcony",
    "Cellar",
    "Attic",
    "Porch",
    "Veranda",
    "Courtyard",
)
SHADOW_MIN = 80
SHADOW_MAX = 120
DEFAULT_SHADOW_COUNT = 100


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("'", "'")))


def _slug(name: str) -> str:
    return "".join(char.casefold() if char.isalnum() else "_" for char in name).strip("_")


def _entity(
    *,
    name: str,
    kind: str,
    area: str,
    aliases: list[str] | None = None,
    state: str = "off",
    floor: str = "Main Floor",
) -> dict[str, Any]:
    domain_map = {
        "light": "light",
        "fan": "fan",
        "climate": "climate",
        "media_player": "media_player",
        "vacuum": "vacuum",
        "scene": "scene",
        "script": "script",
    }
    device_class_map = {
        "media_player": "tv",
    }
    capabilities_map = {
        "light": ("on", "off", "brightness"),
        "fan": ("on", "off", "percentage"),
        "climate": ("heat", "cool", "off"),
        "media_player": ("on", "off"),
        "vacuum": ("start", "stop"),
        "scene": ("activate",),
        "script": ("run",),
    }
    domain = domain_map[kind]
    return {
        "entity_id": f"{domain}.{_slug(name)}",
        "name": name,
        "aliases": aliases or [name],
        "domain": domain,
        "kind": kind,
        "device_class": device_class_map.get(kind),
        "area": area,
        "floor": floor,
        "state": state,
        "capabilities": list(capabilities_map[kind]),
    }


def _home(*entities: dict[str, Any], sayso_entity_area: str, home_id: str) -> dict[str, Any]:
    return {
        "home_id": home_id,
        "sayso_entity_area": sayso_entity_area,
        "entities": list(entities),
    }


def _action(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "action", "calls": list(calls)}


def _no_action(response: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": "no_action", "response": response, "calls": []}
    payload.update(extra)
    return payload


def _status(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {"name": entity["name"]}
    if entity["domain"] in {"light", "fan", "climate", "media_player", "vacuum", "scene", "script"}:
        args["domain"] = [entity["domain"]]
    return {
        "kind": "status",
        "calls": [{"name": "GetLiveContext", "arguments": args}],
        "state": entity["state"],
    }


def _turn_on(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {"name": entity["name"], "domain": [entity["domain"]]}
    return {"name": "HassTurnOn", "arguments": args}


def _turn_off(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {"name": entity["name"], "domain": [entity["domain"]]}
    return {"name": "HassTurnOff", "arguments": args}


def _spec(
    *,
    row_id: str,
    category: str,
    subcategory: str,
    utterance: str,
    home: dict[str, Any],
    expected: dict[str, Any],
    target_names: list[str] | None = None,
    request_hint: str = "",
) -> dict[str, Any]:
    calls = expected.get("calls") or []
    names = target_names or [
        call["arguments"]["name"]
        for call in calls
        if isinstance(call.get("arguments"), dict) and call["arguments"].get("name")
    ]
    return {
        "candidate_id": f"v3_quality_gold_{row_id}",
        "seed": 0,
        "category": category,
        "subcategory": subcategory,
        "home": home,
        "expected": expected,
        "target_names": names,
        "spoken_targets": {},
        "excluded_names": [],
        "contrastive_group": None,
        "request_hint": request_hint,
        "stt_corruption": None,
        "utterance": utterance,
        "quality_eval": True,
        "dataset": "v3_quality_gold",
    }


def gold_specs() -> list[dict[str, Any]]:
    """Return authoritative locked gold rows for the 40k v3 quality gate."""
    great_room_thermostat = _entity(name="Great Room Thermostat", kind="climate", area="Great Room", state="heat")
    family_room_tv = _entity(name="Family Room TV", kind="media_player", area="Family Room", aliases=["tv"])
    home_theater_tv = _entity(name="Home Theater TV", kind="media_player", area="Home Theater", aliases=["tv"])
    rec_room_tv = _entity(name="Rec Room TV", kind="media_player", area="Rec Room")
    bedroom_tv = _entity(name="Guest Suite TV", kind="media_player", area="Guest Suite")
    upstairs_vacuum = _entity(name="Upstairs Robot Vacuum", kind="vacuum", area="Sunroom", state="docked")
    mudroom_vacuum = _entity(name="Mudroom Robot Vacuum", kind="vacuum", area="Mudroom", state="cleaning")
    movie_scene = _entity(name="Movie Night Scene", kind="scene", area="Home Theater", state="off")
    bedtime_scene = _entity(name="Bedtime Scene", kind="scene", area="Guest Suite", state="off")
    morning_script = _entity(name="Good Morning Script", kind="script", area="Kitchen", state="off")
    away_script = _entity(name="Leave Home Script", kind="script", area="Foyer", state="off")
    dining_light = _entity(name="Dining Room Pendant Light", kind="light", area="Dining Room", aliases=["light"])
    study_fan = _entity(name="Study Desk Fan", kind="fan", area="Study", aliases=["fan"])
    library_light_a = _entity(name="Library Reading Lamp", kind="light", area="Library", aliases=["light"])
    library_light_b = _entity(name="Library Desk Lamp", kind="light", area="Library")
    sunroom_light = _entity(name="Sunroom Accent Light", kind="light", area="Sunroom", aliases=["light"])

    rows: list[dict[str, Any]] = [
        _spec(
            row_id="climate_a",
            category="climate_setpoint",
            subcategory="named_device",
            utterance="Set Great Room Thermostat to 72 degrees",
            home=_home(great_room_thermostat, sayso_entity_area="Great Room", home_id="v3_gold_climate_a"),
            expected=_action(
                {
                    "name": "HassClimateSetTemperature",
                    "arguments": {"name": "Great Room Thermostat", "temperature": 72},
                }
            ),
        ),
        _spec(
            row_id="climate_b",
            category="climate_setpoint",
            subcategory="conversational",
            utterance="Could you set the family room thermostat to 68 degrees for me?",
            home=_home(
                _entity(name="Family Room Thermostat", kind="climate", area="Family Room", state="cool"),
                sayso_entity_area="Family Room",
                home_id="v3_gold_climate_b",
            ),
            expected=_action(
                {
                    "name": "HassClimateSetTemperature",
                    "arguments": {"name": "Family Room Thermostat", "temperature": 68},
                }
            ),
        ),
        _spec(
            row_id="media_play",
            category="media_play",
            subcategory="named_device",
            utterance="Play Home Theater TV",
            home=_home(home_theater_tv, sayso_entity_area="Home Theater", home_id="v3_gold_media_play"),
            expected=_action({"name": "HassMediaUnpause", "arguments": {"name": "Home Theater TV", "domain": ["media_player"], "device_class": ["tv"]}}),
        ),
        _spec(
            row_id="media_pause",
            category="media_pause",
            subcategory="named_device",
            utterance="Pause Rec Room TV",
            home=_home(rec_room_tv, sayso_entity_area="Rec Room", home_id="v3_gold_media_pause"),
            expected=_action({"name": "HassMediaPause", "arguments": {"name": "Rec Room TV", "domain": ["media_player"], "device_class": ["tv"]}}),
        ),
        _spec(
            row_id="media_volume",
            category="media_volume",
            subcategory="absolute",
            utterance="Set Family Room TV volume to 45 percent",
            home=_home(family_room_tv, sayso_entity_area="Family Room", home_id="v3_gold_media_volume"),
            expected=_action(
                {
                    "name": "HassSetVolume",
                    "arguments": {"name": "Family Room TV", "domain": ["media_player"], "device_class": ["tv"], "volume_level": 45},
                }
            ),
        ),
        _spec(
            row_id="media_volume_rel",
            category="media_volume",
            subcategory="relative",
            utterance="Turn up Guest Suite TV volume",
            home=_home(bedroom_tv, sayso_entity_area="Guest Suite", home_id="v3_gold_media_volume_rel"),
            expected=_action(
                {
                    "name": "HassSetVolumeRelative",
                    "arguments": {"name": "Guest Suite TV", "volume_step": "up"},
                }
            ),
        ),
        _spec(
            row_id="media_mute",
            category="media_mute",
            subcategory="named_device",
            utterance="Mute Home Theater TV",
            home=_home(home_theater_tv, sayso_entity_area="Home Theater", home_id="v3_gold_media_mute"),
            expected=_action({"name": "HassMediaPlayerMute", "arguments": {"name": "Home Theater TV", "domain": ["media_player"], "device_class": ["tv"]}}),
        ),
        _spec(
            row_id="timer_start",
            category="timer_start",
            subcategory="minutes",
            utterance="Start a 15 minute timer",
            home=_home(sunroom_light, sayso_entity_area="Sunroom", home_id="v3_gold_timer_start"),
            expected=_action({"name": "HassStartTimer", "arguments": {"minutes": 15}}),
            target_names=[],
        ),
        _spec(
            row_id="timer_start_named",
            category="timer_start",
            subcategory="hours",
            utterance="Start a 1 hour timer",
            home=_home(sunroom_light, sayso_entity_area="Sunroom", home_id="v3_gold_timer_start_named"),
            expected=_action({"name": "HassStartTimer", "arguments": {"hours": 1}}),
            target_names=[],
        ),
        _spec(
            row_id="timer_pause",
            category="timer_pause",
            subcategory="generic",
            utterance="Pause the timer",
            home=_home(sunroom_light, sayso_entity_area="Sunroom", home_id="v3_gold_timer_pause"),
            expected=_action({"name": "HassPauseTimer", "arguments": {}}),
            target_names=[],
        ),
        _spec(
            row_id="timer_status",
            category="timer_status",
            subcategory="generic",
            utterance="What is the timer status?",
            home=_home(sunroom_light, sayso_entity_area="Sunroom", home_id="v3_gold_timer_status"),
            expected=_action({"name": "HassTimerStatus", "arguments": {}}),
            target_names=[],
        ),
        _spec(
            row_id="timer_cancel_all",
            category="timer_cancel",
            subcategory="all",
            utterance="Cancel all timers",
            home=_home(sunroom_light, sayso_entity_area="Sunroom", home_id="v3_gold_timer_cancel_all"),
            expected=_action({"name": "HassCancelAllTimers", "arguments": {}}),
            target_names=[],
        ),
        _spec(
            row_id="timer_cancel_named",
            category="timer_cancel",
            subcategory="area",
            utterance="Cancel all timers in the Sunroom",
            home=_home(sunroom_light, sayso_entity_area="Sunroom", home_id="v3_gold_timer_cancel_named"),
            expected=_action({"name": "HassCancelAllTimers", "arguments": {"area": "Sunroom"}}),
            target_names=[],
        ),
        _spec(
            row_id="vacuum_start",
            category="vacuum_start",
            subcategory="named_device",
            utterance="Start Upstairs Robot Vacuum",
            home=_home(upstairs_vacuum, sayso_entity_area="Sunroom", home_id="v3_gold_vacuum_start"),
            expected=_action({"name": "HassVacuumStart", "arguments": {"name": "Upstairs Robot Vacuum", "domain": ["vacuum"]}}),
        ),
        _spec(
            row_id="vacuum_return",
            category="vacuum_return",
            subcategory="named_device",
            utterance="Send Mudroom Robot Vacuum home",
            home=_home(mudroom_vacuum, sayso_entity_area="Mudroom", home_id="v3_gold_vacuum_return"),
            expected=_action({"name": "HassVacuumReturnToBase", "arguments": {"name": "Mudroom Robot Vacuum", "domain": ["vacuum"]}}),
        ),
        _spec(
            row_id="vacuum_clean_area",
            category="vacuum_clean_area",
            subcategory="area",
            utterance="Vacuum the Sunroom",
            home=_home(upstairs_vacuum, sayso_entity_area="Sunroom", home_id="v3_gold_vacuum_clean_area"),
            expected=_action(
                {
                    "name": "HassVacuumCleanArea",
                    "arguments": {"name": "Upstairs Robot Vacuum", "area": "Sunroom"},
                }
            ),
        ),
        _spec(
            row_id="scene_a",
            category="scene_activate",
            subcategory="named_scene",
            utterance="Activate Movie Night Scene",
            home=_home(movie_scene, sayso_entity_area="Home Theater", home_id="v3_gold_scene_a"),
            expected=_action(_turn_on(movie_scene)),
        ),
        _spec(
            row_id="scene_b",
            category="scene_activate",
            subcategory="conversational",
            utterance="Run the bedtime scene in the guest suite",
            home=_home(bedtime_scene, sayso_entity_area="Guest Suite", home_id="v3_gold_scene_b"),
            expected=_action(_turn_on(bedtime_scene)),
        ),
        _spec(
            row_id="script_a",
            category="script_run",
            subcategory="named_script",
            utterance="Run Good Morning Script",
            home=_home(morning_script, sayso_entity_area="Kitchen", home_id="v3_gold_script_a"),
            expected=_action(_turn_on(morning_script)),
        ),
        _spec(
            row_id="script_b",
            category="script_run",
            subcategory="conversational",
            utterance="Please start the leave home script",
            home=_home(away_script, sayso_entity_area="Foyer", home_id="v3_gold_script_b"),
            expected=_action(_turn_on(away_script)),
        ),
        _spec(
            row_id="ordinary_on",
            category="ordinary_on",
            subcategory="light",
            utterance="Turn on Dining Room Pendant Light",
            home=_home(dining_light, sayso_entity_area="Dining Room", home_id="v3_gold_ordinary_on"),
            expected=_action(_turn_on(dining_light)),
        ),
        _spec(
            row_id="ordinary_off",
            category="ordinary_off",
            subcategory="fan",
            utterance="Turn off Study Desk Fan",
            home=_home(study_fan, sayso_entity_area="Study", home_id="v3_gold_ordinary_off"),
            expected=_action(_turn_off(study_fan)),
        ),
        _spec(
            row_id="status_media",
            category="status",
            subcategory="media_player",
            utterance="What is the status of Home Theater TV?",
            home=_home(home_theater_tv, sayso_entity_area="Home Theater", home_id="v3_gold_status_media"),
            expected=_status(home_theater_tv),
            target_names=["Home Theater TV"],
        ),
        _spec(
            row_id="status_climate",
            category="status",
            subcategory="climate",
            utterance="Is the Great Room Thermostat heating?",
            home=_home(great_room_thermostat, sayso_entity_area="Great Room", home_id="v3_gold_status_climate"),
            expected=_status(great_room_thermostat),
            target_names=["Great Room Thermostat"],
        ),
        _spec(
            row_id="ambiguity_resolve",
            category="ambiguity",
            subcategory="one_default_light",
            utterance="Turn on the reading lamp",
            home=_home(library_light_a, library_light_b, sayso_entity_area="Library", home_id="v3_gold_ambiguity_resolve"),
            expected=_action(_turn_on(library_light_a)),
        ),
        _spec(
            row_id="ambiguity_clarify",
            category="ambiguity",
            subcategory="two_lights",
            utterance="Switch on the accent light",
            home=_home(
                _entity(name="Sunroom Table Lamp", kind="light", area="Sunroom", aliases=["accent light"]),
                _entity(name="Sunroom Floor Lamp", kind="light", area="Sunroom", aliases=["accent light"]),
                sayso_entity_area="Sunroom",
                home_id="v3_gold_ambiguity_clarify",
            ),
            expected=_no_action("clarify"),
            target_names=[],
            request_hint="switch on the accent light",
        ),
        _spec(
            row_id="ambiguity_area_unavailable",
            category="ambiguity",
            subcategory="zero_lights",
            utterance="Turn on the desk light",
            home=_home(study_fan, sayso_entity_area="Study", home_id="v3_gold_ambiguity_area"),
            expected=_no_action(
                "area_unavailable",
                unavailable={"area": "study", "type": "lights"},
            ),
            target_names=[],
            request_hint="turn on the desk light",
        ),
        _spec(
            row_id="unsupported_lawn",
            category="unsupported_no_action",
            subcategory="lawn_mower",
            utterance="Start the lawn mower in the courtyard",
            home=_home(sunroom_light, sayso_entity_area="Sunroom", home_id="v3_gold_unsupported_lawn"),
            expected=_no_action("unsupported"),
            target_names=[],
            request_hint="start the lawn mower in the courtyard",
        ),
        _spec(
            row_id="unsupported_todo",
            category="unsupported_no_action",
            subcategory="todo",
            utterance="Add milk to the shopping list",
            home=_home(sunroom_light, sayso_entity_area="Sunroom", home_id="v3_gold_unsupported_todo"),
            expected=_no_action("unsupported"),
            target_names=[],
            request_hint="add milk to the shopping list",
        ),
        _spec(
            row_id="no_call_clarify",
            category="unsupported_no_action",
            subcategory="incomplete",
            utterance="Set the thermostat to",
            home=_home(great_room_thermostat, sayso_entity_area="Great Room", home_id="v3_gold_no_call_clarify"),
            expected=_no_action("clarify"),
            target_names=[],
            request_hint="set the thermostat to",
        ),
    ]
    return rows


def _utterance_for_spec(spec: dict[str, Any]) -> str:
    utterance = expand_utterance(spec)
    if isinstance(utterance, str) and utterance.strip():
        return utterance.strip()
    expected = spec.get("expected") or {}
    if expected.get("kind") == "no_action":
        return str(spec.get("request_hint") or "unsupported request")
    calls = expected.get("calls") or []
    if not calls:
        return str(spec.get("request_hint") or "")
    target_names = spec.get("target_names") or []
    target = target_names[0] if target_names else ""
    phrase = _phrase_for_call(target, calls[0])
    if phrase:
        return phrase[0].upper() + phrase[1:]
    return request_seed_from_spec(spec)


def _shadow_spec_shell(
    *,
    candidate_id: str,
    seed: int,
    category: str,
    subcategory: str,
    home: dict[str, Any],
    expected: dict[str, Any],
    target_names: list[str],
    request_hint: str = "",
    utterance: str | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "category": category,
        "subcategory": subcategory,
        "home": home,
        "expected": expected,
        "target_names": target_names,
        "spoken_targets": {},
        "excluded_names": [],
        "contrastive_group": None,
        "request_hint": request_hint,
        "stt_corruption": None,
        "utterance": utterance,
        "dataset": "v3_quality_shadow",
    }


def build_shadow_specs(seed: int = 20260906, count: int = DEFAULT_SHADOW_COUNT) -> list[dict[str, Any]]:
    """Build shadow rows that mirror gold concepts with fresh homes and phrasing."""
    if not SHADOW_MIN <= count <= SHADOW_MAX:
        raise ValueError(f"shadow count must be {SHADOW_MIN}-{SHADOW_MAX}, got {count}")
    slots = {
        "climate_setpoint": max(2, count // 12),
        "media_play": max(1, count // 20),
        "media_pause": max(1, count // 20),
        "media_volume": max(2, count // 15),
        "media_mute": max(1, count // 20),
        "timer_start": max(2, count // 15),
        "timer_pause": max(1, count // 25),
        "timer_status": max(1, count // 25),
        "timer_cancel": max(2, count // 15),
        "vacuum_start": max(1, count // 20),
        "vacuum_return": max(1, count // 20),
        "vacuum_clean_area": max(1, count // 20),
        "scene_activate": max(2, count // 15),
        "script_run": max(2, count // 15),
        "ordinary_on": max(2, count // 15),
        "ordinary_off": max(1, count // 20),
        "status": max(2, count // 15),
        "ambiguity": max(3, count // 10),
        "unsupported_no_action": max(2, count // 15),
    }
    while sum(slots.values()) > count:
        key = max(slots, key=lambda name: slots[name])
        slots[key] -= 1
    while sum(slots.values()) < count:
        slots["ambiguity"] += 1

    specs: list[dict[str, Any]] = []
    slot = 0

    for index in range(slots["climate_setpoint"]):
        rng = random.Random((seed << 20) ^ index)
        area = _SHADOW_AREAS[index % len(_SHADOW_AREAS)]
        thermostat = _entity(name=f"{area} Comfort Thermostat", kind="climate", area=area, state="heat")
        temp = 66 + (index * 3) % 8
        home = _home(thermostat, sayso_entity_area=area, home_id=f"v3_shadow_climate_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_climate_{slot:04d}",
                seed=seed,
                category="climate_setpoint",
                subcategory="named_device",
                home=home,
                expected=_action(
                    {
                        "name": "HassClimateSetTemperature",
                        "arguments": {"name": thermostat["name"], "temperature": temp},
                    }
                ),
                target_names=[thermostat["name"]],
            )
        )
        slot += 1

    for index in range(slots["media_play"]):
        rng = random.Random((seed << 18) ^ (index + 50))
        area = _SHADOW_AREAS[(index + 2) % len(_SHADOW_AREAS)]
        tv = _entity(name=f"{area} Wall TV", kind="media_player", area=area, aliases=["tv"])
        home = _home(tv, sayso_entity_area=area, home_id=f"v3_shadow_media_play_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_media_play_{slot:04d}",
                seed=seed,
                category="media_play",
                subcategory="named_device",
                home=home,
                expected=_action(
                    {
                        "name": "HassMediaUnpause",
                        "arguments": {"name": tv["name"], "domain": ["media_player"], "device_class": ["tv"]},
                    }
                ),
                target_names=[tv["name"]],
            )
        )
        slot += 1

    for index in range(slots["media_pause"]):
        area = _SHADOW_AREAS[(index + 4) % len(_SHADOW_AREAS)]
        tv = _entity(name=f"{area} Corner TV", kind="media_player", area=area)
        home = _home(tv, sayso_entity_area=area, home_id=f"v3_shadow_media_pause_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_media_pause_{slot:04d}",
                seed=seed,
                category="media_pause",
                subcategory="named_device",
                home=home,
                expected=_action(
                    {
                        "name": "HassMediaPause",
                        "arguments": {"name": tv["name"], "domain": ["media_player"], "device_class": ["tv"]},
                    }
                ),
                target_names=[tv["name"]],
            )
        )
        slot += 1

    for index in range(slots["media_volume"]):
        area = _SHADOW_AREAS[(index + 1) % len(_SHADOW_AREAS)]
        tv = _entity(name=f"{area} Soundbar TV", kind="media_player", area=area)
        home = _home(tv, sayso_entity_area=area, home_id=f"v3_shadow_media_volume_{slot:04d}")
        if index % 2 == 0:
            level = 25 + index * 7
            expected = _action(
                {
                    "name": "HassSetVolume",
                    "arguments": {
                        "name": tv["name"],
                        "domain": ["media_player"],
                        "device_class": ["tv"],
                        "volume_level": level,
                    },
                }
            )
            subcategory = "absolute"
        else:
            expected = _action(
                {
                    "name": "HassSetVolumeRelative",
                    "arguments": {"name": tv["name"], "volume_step": "down"},
                }
            )
            subcategory = "relative"
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_media_volume_{slot:04d}",
                seed=seed,
                category="media_volume",
                subcategory=subcategory,
                home=home,
                expected=expected,
                target_names=[tv["name"]],
            )
        )
        slot += 1

    for index in range(slots["media_mute"]):
        area = _SHADOW_AREAS[(index + 6) % len(_SHADOW_AREAS)]
        tv = _entity(name=f"{area} Bedroom TV", kind="media_player", area=area)
        home = _home(tv, sayso_entity_area=area, home_id=f"v3_shadow_media_mute_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_media_mute_{slot:04d}",
                seed=seed,
                category="media_mute",
                subcategory="named_device",
                home=home,
                expected=_action(
                    {
                        "name": "HassMediaPlayerMute",
                        "arguments": {"name": tv["name"], "domain": ["media_player"], "device_class": ["tv"]},
                    }
                ),
                target_names=[tv["name"]],
            )
        )
        slot += 1

    timer_light = _entity(name="Porch Accent Light", kind="light", area="Porch")
    for index in range(slots["timer_start"]):
        area = _SHADOW_AREAS[(index + 3) % len(_SHADOW_AREAS)]
        minutes = 7 + index * 9
        home = _home(timer_light, sayso_entity_area=area, home_id=f"v3_shadow_timer_start_{slot:04d}")
        args: dict[str, Any] = {"minutes": minutes}
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_timer_start_{slot:04d}",
                seed=seed,
                category="timer_start",
                subcategory="minutes",
                home=home,
                expected=_action({"name": "HassStartTimer", "arguments": args}),
                target_names=[],
                utterance=f"Set a {minutes} minute timer in the {area}",
            )
        )
        slot += 1

    for index in range(slots["timer_pause"]):
        area = _SHADOW_AREAS[(index + 5) % len(_SHADOW_AREAS)]
        home = _home(timer_light, sayso_entity_area=area, home_id=f"v3_shadow_timer_pause_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_timer_pause_{slot:04d}",
                seed=seed,
                category="timer_pause",
                subcategory="generic",
                home=home,
                expected=_action({"name": "HassPauseTimer", "arguments": {}}),
                target_names=[],
                utterance=f"Hold the {area} countdown timer",
            )
        )
        slot += 1

    for index in range(slots["timer_status"]):
        area = _SHADOW_AREAS[(index + 7) % len(_SHADOW_AREAS)]
        home = _home(timer_light, sayso_entity_area=area, home_id=f"v3_shadow_timer_status_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_timer_status_{slot:04d}",
                seed=seed,
                category="timer_status",
                subcategory="generic",
                home=home,
                expected=_action({"name": "HassTimerStatus", "arguments": {}}),
                target_names=[],
                utterance=f"How much time is left on the {area} timer?",
            )
        )
        slot += 1

    for index in range(slots["timer_cancel"]):
        area = _SHADOW_AREAS[(index + 8) % len(_SHADOW_AREAS)]
        home = _home(timer_light, sayso_entity_area=area, home_id=f"v3_shadow_timer_cancel_{slot:04d}")
        if index % 2 == 0:
            expected = _action({"name": "HassCancelAllTimers", "arguments": {}})
            subcategory = "all"
            utterance = f"Stop every running timer in the {area}"
        else:
            expected = _action({"name": "HassCancelAllTimers", "arguments": {"area": area}})
            subcategory = "area"
            utterance = f"Clear all timers in the {area}"
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_timer_cancel_{slot:04d}",
                seed=seed,
                category="timer_cancel",
                subcategory=subcategory,
                home=home,
                expected=expected,
                target_names=[],
                utterance=utterance,
            )
        )
        slot += 1

    for index in range(slots["vacuum_start"]):
        area = _SHADOW_AREAS[(index + 9) % len(_SHADOW_AREAS)]
        vacuum = _entity(name=f"{area} Floor Vacuum", kind="vacuum", area=area)
        home = _home(vacuum, sayso_entity_area=area, home_id=f"v3_shadow_vacuum_start_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_vacuum_start_{slot:04d}",
                seed=seed,
                category="vacuum_start",
                subcategory="named_device",
                home=home,
                expected=_action({"name": "HassVacuumStart", "arguments": {"name": vacuum["name"], "domain": ["vacuum"]}}),
                target_names=[vacuum["name"]],
            )
        )
        slot += 1

    for index in range(slots["vacuum_return"]):
        area = _SHADOW_AREAS[(index + 10) % len(_SHADOW_AREAS)]
        vacuum = _entity(name=f"{area} Robot Vacuum", kind="vacuum", area=area, state="cleaning")
        home = _home(vacuum, sayso_entity_area=area, home_id=f"v3_shadow_vacuum_return_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_vacuum_return_{slot:04d}",
                seed=seed,
                category="vacuum_return",
                subcategory="named_device",
                home=home,
                expected=_action(
                    {"name": "HassVacuumReturnToBase", "arguments": {"name": vacuum["name"], "domain": ["vacuum"]}}
                ),
                target_names=[vacuum["name"]],
            )
        )
        slot += 1

    for index in range(slots["vacuum_clean_area"]):
        area = _SHADOW_AREAS[(index + 11) % len(_SHADOW_AREAS)]
        vacuum = _entity(name=f"{area} Robot Vacuum", kind="vacuum", area=area)
        home = _home(vacuum, sayso_entity_area=area, home_id=f"v3_shadow_vacuum_area_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_vacuum_area_{slot:04d}",
                seed=seed,
                category="vacuum_clean_area",
                subcategory="area",
                home=home,
                expected=_action(
                    {"name": "HassVacuumCleanArea", "arguments": {"name": vacuum["name"], "area": area}}
                ),
                target_names=[vacuum["name"]],
            )
        )
        slot += 1

    for index in range(slots["scene_activate"]):
        area = _SHADOW_AREAS[index % len(_SHADOW_AREAS)]
        scene = _entity(name=f"{area} Relax Scene", kind="scene", area=area)
        home = _home(scene, sayso_entity_area=area, home_id=f"v3_shadow_scene_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_scene_{slot:04d}",
                seed=seed,
                category="scene_activate",
                subcategory="named_scene",
                home=home,
                expected=_action(_turn_on(scene)),
                target_names=[scene["name"]],
            )
        )
        slot += 1

    for index in range(slots["script_run"]):
        area = _SHADOW_AREAS[(index + 2) % len(_SHADOW_AREAS)]
        script = _entity(name=f"{area} Away Script", kind="script", area=area)
        home = _home(script, sayso_entity_area=area, home_id=f"v3_shadow_script_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_script_{slot:04d}",
                seed=seed,
                category="script_run",
                subcategory="named_script",
                home=home,
                expected=_action(_turn_on(script)),
                target_names=[script["name"]],
            )
        )
        slot += 1

    for index in range(slots["ordinary_on"]):
        area = _SHADOW_AREAS[(index + 4) % len(_SHADOW_AREAS)]
        light = _entity(name=f"{area} Task Light", kind="light", area=area, aliases=["light"])
        home = _home(light, sayso_entity_area=area, home_id=f"v3_shadow_on_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_on_{slot:04d}",
                seed=seed,
                category="ordinary_on",
                subcategory="light",
                home=home,
                expected=_action(_turn_on(light)),
                target_names=[light["name"]],
            )
        )
        slot += 1

    for index in range(slots["ordinary_off"]):
        area = _SHADOW_AREAS[(index + 6) % len(_SHADOW_AREAS)]
        fan = _entity(name=f"{area} Ceiling Fan", kind="fan", area=area, aliases=["fan"])
        home = _home(fan, sayso_entity_area=area, home_id=f"v3_shadow_off_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_off_{slot:04d}",
                seed=seed,
                category="ordinary_off",
                subcategory="fan",
                home=home,
                expected=_action(_turn_off(fan)),
                target_names=[fan["name"]],
            )
        )
        slot += 1

    for index in range(slots["status"]):
        area = _SHADOW_AREAS[(index + 1) % len(_SHADOW_AREAS)]
        if index % 2 == 0:
            target = _entity(name=f"{area} Lounge TV", kind="media_player", area=area)
            subcategory = "media_player"
        else:
            target = _entity(name=f"{area} Thermostat", kind="climate", area=area)
            subcategory = "climate"
        home = _home(target, sayso_entity_area=area, home_id=f"v3_shadow_status_{slot:04d}")
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_status_{slot:04d}",
                seed=seed,
                category="status",
                subcategory=subcategory,
                home=home,
                expected=_status(target),
                target_names=[target["name"]],
            )
        )
        slot += 1

    for index in range(slots["ambiguity"]):
        area = _SHADOW_AREAS[(index + 3) % len(_SHADOW_AREAS)]
        utterance = ""
        if index % 3 == 0:
            light = _entity(name=f"{area} Reading Lamp", kind="light", area=area, aliases=["reading lamp"])
            home = _home(light, sayso_entity_area=area, home_id=f"v3_shadow_ambiguity_{slot:04d}")
            expected = _action(_turn_on(light))
            hint = "turn on the reading lamp"
            utterance = f"Please switch on the {area} reading lamp"
        elif index % 3 == 1:
            a = _entity(name=f"{area} Lamp A", kind="light", area=area, aliases=["accent light"])
            b = _entity(name=f"{area} Lamp B", kind="light", area=area, aliases=["accent light"])
            home = _home(a, b, sayso_entity_area=area, home_id=f"v3_shadow_ambiguity_{slot:04d}")
            expected = _no_action("clarify")
            hint = "enable the mood light"
            utterance = f"Enable the mood light in the {area}"
        else:
            fan = _entity(name=f"{area} Desk Fan", kind="fan", area=area)
            home = _home(fan, sayso_entity_area=area, home_id=f"v3_shadow_ambiguity_{slot:04d}")
            expected = _no_action("area_unavailable", unavailable={"area": area.casefold(), "type": "lights"})
            hint = "turn on the desk light"
            utterance = f"Turn on the {area} desk light"
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_ambiguity_{slot:04d}",
                seed=seed,
                category="ambiguity",
                subcategory="generic",
                home=home,
                expected=expected,
                target_names=[],
                request_hint=hint,
                utterance=utterance,
            )
        )
        slot += 1

    for index in range(slots["unsupported_no_action"]):
        area = _SHADOW_AREAS[(index + 5) % len(_SHADOW_AREAS)]
        light = _entity(name=f"{area} Hall Light", kind="light", area=area)
        home = _home(light, sayso_entity_area=area, home_id=f"v3_shadow_unsupported_{slot:04d}")
        hints = (
            f"mow the front lawn with the robot mower in the {area}",
            f"press the panic button in the {area}",
            f"add batteries to the {area} shopping list",
        )
        hint = hints[index % len(hints)]
        specs.append(
            _shadow_spec_shell(
                candidate_id=f"v3_shadow_unsupported_{slot:04d}",
                seed=seed,
                category="unsupported_no_action",
                subcategory="unsupported",
                home=home,
                expected=_no_action("unsupported"),
                target_names=[],
                request_hint=hint,
                utterance=hint,
            )
        )
        slot += 1

    for spec in specs:
        if not spec.get("utterance"):
            spec["utterance"] = _utterance_for_spec(spec)
    return specs[:count]


def gold_user_prompts() -> set[str]:
    return {spec["utterance"] for spec in gold_specs()}


def shadow_user_prompts() -> set[str]:
    return {spec["utterance"] for spec in build_shadow_specs()}


def excluded_train_prompts() -> set[str]:
    """Normalized user prompts that must not appear in train JSONL overlap checks."""
    excluded = {_normalized(text) for text in gold_user_prompts()}
    excluded.update(_normalized(text) for text in shadow_user_prompts())
    excluded.update(_normalized(text) for text in recipe_lock_prompts())
    return excluded


def build_gold_examples() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for spec in gold_specs():
        example = render_example(spec)
        example["metadata"]["quality_eval"] = True
        example["metadata"]["v3_quality_gold"] = True
        examples.append(example)
    return examples


def build_shadow_examples(seed: int = 20260906, count: int = DEFAULT_SHADOW_COUNT) -> list[dict[str, Any]]:
    specs = build_shadow_specs(seed=seed, count=count)
    rows = [render_example(spec) for spec in specs]
    for row in rows:
        row["metadata"]["shadow_eval"] = True
        row["metadata"]["v3_quality_shadow"] = True
    return rows


def expected_tool_calls(example: dict[str, Any]) -> list[dict[str, Any]]:
    batches = extract_assistant_tool_calls(example.get("messages") or [])
    return [call for batch in batches for call in batch]


def score_quality_gold(example: dict[str, Any], actual_messages: list[dict[str, Any]]) -> dict[str, Any]:
    expected_messages = example.get("messages") or []
    expected_calls = expected_tool_calls(example)
    actual_calls = expected_tool_calls({"messages": actual_messages})
    name_ok, args_ok, _multi_ok, category = score_expected_vs_actual(expected_messages, actual_messages)
    no_call_expected = not expected_calls
    no_call_actual = not actual_calls
    no_call_ok = no_call_expected == no_call_actual
    if no_call_expected and actual_calls:
        category = category or "unexpected_tool_call"
    if not no_call_expected and not actual_calls:
        category = category or "missing_tool_call"
    return {
        "tool_name_exact": name_ok,
        "args_exact": args_ok,
        "no_call_when_expected": no_call_ok,
        "failure_category": category,
        "pass": name_ok and args_ok and no_call_ok,
    }


def assert_quality_eval_contract(example: dict[str, Any]) -> None:
    blob = json.dumps(example, ensure_ascii=False)
    if _BANNED.search(blob):
        raise ValueError("v3 quality eval contains banned eval or ChatML tool-call markers")
    metadata = example.get("metadata") or {}
    if str(metadata.get("candidate_id", "")).startswith("evals/cases"):
        raise ValueError("v3 quality eval must not use evals/cases IDs")
    schemas = tool_schema_map(v2_openai_tools())
    for call in expected_tool_calls(example):
        args = call.get("function", {}).get("arguments")
        if not isinstance(args, str):
            raise ValueError("v3 quality eval keeps function.arguments as JSON strings")
        parsed = parse_tool_arguments(args)
        if parsed is None:
            raise ValueError("v3 quality eval arguments must parse as JSON objects")
        reason = validate_tool_arguments(call["function"]["name"], parsed, schemas)
        if reason:
            raise ValueError(f"v3 quality eval arguments failed schema validation: {reason}")


def v3_quality_summary() -> dict[str, Any]:
    gold = gold_specs()
    shadow = build_shadow_specs()
    return {
        "gold_count": len(gold),
        "shadow_count": len(shadow),
        "gold_categories": sorted({spec["category"] for spec in gold}),
        "shadow_categories": sorted({spec["category"] for spec in shadow}),
        "gold_user_prompts": sorted(gold_user_prompts()),
    }
