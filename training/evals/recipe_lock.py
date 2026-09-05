"""Human-locked gold eval rows (recipes 1–8, no thermostat)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_dataset import render_example  # noqa: E402
from evals.metrics import (  # noqa: E402
    extract_assistant_tool_calls,
    parse_tool_arguments,
    score_expected_vs_actual,
    tool_call_signature,
)

_BANNED = re.compile(r"evals/cases/|<tool_call>|tool_call_start", re.I)


def _entity(
    *,
    name: str,
    kind: str,
    area: str,
    aliases: list[str] | None = None,
    state: str = "off",
) -> dict[str, Any]:
    domain = "cover" if kind in {"blinds", "garage_door"} else kind
    device_class = {"blinds": "blind", "garage_door": "garage", "lock": "door"}.get(kind)
    slug = "".join(char.casefold() if char.isalnum() else "_" for char in name).strip("_")
    capabilities = {
        "light": ("on", "off", "brightness"),
        "fan": ("on", "off", "percentage"),
        "switch": ("on", "off"),
        "blinds": ("open", "close"),
        "garage_door": ("open", "close"),
        "lock": ("lock", "unlock"),
    }[kind]
    return {
        "entity_id": f"{domain}.{slug}",
        "name": name,
        "aliases": aliases or [name],
        "domain": domain,
        "kind": kind,
        "device_class": device_class,
        "area": area,
        "floor": "Main Floor",
        "state": state,
        "capabilities": list(capabilities),
    }


def _home(*entities: dict[str, Any], sayso_entity_area: str) -> dict[str, Any]:
    return {
        "home_id": "recipe_lock_home",
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
    if entity["domain"] in {"light", "fan", "switch"}:
        args["domain"] = [entity["domain"]]
    return {
        "kind": "status",
        "calls": [{"name": "GetLiveContext", "arguments": args}],
        "state": entity["state"],
    }


def _turn_on(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {"name": entity["name"]}
    if entity["domain"] in {"light", "fan", "switch"}:
        args["domain"] = [entity["domain"]]
    elif entity["device_class"]:
        args["device_class"] = [entity["device_class"]]
    return {"name": "HassTurnOn", "arguments": args}


def _turn_off(entity: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {"name": entity["name"]}
    if entity["domain"] in {"light", "fan", "switch"}:
        args["domain"] = [entity["domain"]]
    elif entity["device_class"]:
        args["device_class"] = [entity["device_class"]]
    return {"name": "HassTurnOff", "arguments": args}


def _light_set(entity: dict[str, Any], *, brightness: int) -> dict[str, Any]:
    return {
        "name": "HassLightSet",
        "arguments": {"name": entity["name"], "domain": ["light"], "brightness": brightness},
    }


def _spec(
    *,
    recipe: int,
    row: str,
    category: str,
    utterance: str,
    home: dict[str, Any],
    expected: dict[str, Any],
    target_names: list[str] | None = None,
    request_hint: str = "",
) -> dict[str, Any]:
    calls = expected.get("calls") or []
    names = target_names or [
        call["arguments"]["name"] for call in calls if isinstance(call.get("arguments"), dict) and call["arguments"].get("name")
    ]
    return {
        "candidate_id": f"recipe_lock_{recipe:02d}_{row}",
        "seed": 0,
        "category": category,
        "subcategory": row,
        "recipe": recipe,
        "recipe_row": row,
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
    }


def locked_specs() -> list[dict[str, Any]]:
    """Return one authoritative spec per locked yes-row (recipes 1–8)."""
    office_main = _entity(name="Office Main Light", kind="light", area="Office", aliases=["office light"])
    kitchen_garage = _entity(name="Kitchen North Garage Door", kind="garage_door", area="Kitchen")
    living_fan = _entity(name="Living Room Ceiling Fan", kind="fan", area="Living Room", aliases=["fan"])
    herb_light = _entity(name="Kitchen Herb Garden Cool Light", kind="light", area="Kitchen")
    joes_kitchen = _entity(name="Joe's Kitchen Light", kind="light", area="Kitchen", aliases=["kitchen light"])
    omalleys_blinds = _entity(name="O'Malley's Study Blinds", kind="blinds", area="Office", aliases=["study blinds"])
    kids_light = _entity(name="Kids' Room Light", kind="light", area="Guest Room", aliases=["kids room light"])
    joes_lock = _entity(name="Joe's Guest Room Door Lock", kind="lock", area="Guest Room", aliases=["guest room door"])
    kitchen_north = _entity(name="Kitchen North Light", kind="light", area="Kitchen")
    basement_south_lock = _entity(name="Basement South Door Lock", kind="lock", area="Basement", aliases=["basement door"])
    basement_south_fan = _entity(name="Basement South Fan", kind="fan", area="Basement", aliases=["basement fan"])
    workshop_west_fan = _entity(name="Workshop West Fan", kind="fan", area="Workshop")
    workshop_fan = _entity(name="Joe's Workshop Fan", kind="fan", area="Workshop", aliases=["workshop fan"])
    primary_bedroom_garage = _entity(
        name="Primary Bedroom Corner Garage Door",
        kind="garage_door",
        area="Primary Bedroom",
    )
    garage_ceiling_fan = _entity(name="Garage Ceiling Fan", kind="fan", area="Garage")
    kitchen_sink = _entity(name="Kitchen Sink Cool Light", kind="light", area="Kitchen", aliases=["light", "kitchen light"])
    kitchen_ceiling = _entity(name="Kitchen Ceiling Cool Light", kind="light", area="Kitchen", aliases=["light", "kitchen light"])
    hallway_east = _entity(name="Hallway East Outlet", kind="switch", area="Hallway", aliases=["outlet"])
    hallway_west = _entity(name="Hallway West Outlet", kind="switch", area="Hallway", aliases=["outlet"])
    nursery_outlet = _entity(name="Nursery East Outlet", kind="switch", area="Nursery")
    patio_blinds = _entity(name="Patio South Blinds", kind="blinds", area="Patio", aliases=["blinds"])
    workshop_blinds = _entity(name="Joe's Workshop Blinds", kind="blinds", area="Workshop")
    kitchen_lock = _entity(name="Kitchen Back Door Lock", kind="lock", area="Kitchen", aliases=["door"])
    patio_lock = _entity(name="Patio Side Door Lock", kind="lock", area="Patio", aliases=["patio door"])
    garage_west_fan = _entity(name="Garage West Fan", kind="fan", area="Garage", aliases=["garage west fan"])

    rows: list[dict[str, Any]] = [
        _spec(
            recipe=1,
            row="a",
            category="clean_direct",
            utterance="Turn on Office Main Light",
            home=_home(office_main, sayso_entity_area="Office"),
            expected=_action(_turn_on(office_main)),
        ),
        _spec(
            recipe=1,
            row="b",
            category="clean_direct",
            utterance="Close Kitchen North Garage Door",
            home=_home(kitchen_garage, sayso_entity_area="Kitchen"),
            expected=_action(_turn_off(kitchen_garage)),
        ),
        _spec(
            recipe=2,
            row="a",
            category="conversational",
            utterance="Hey, when you get a chance, turn on the living room ceiling fan.",
            home=_home(living_fan, sayso_entity_area="Living Room"),
            expected=_action(_turn_on(living_fan)),
        ),
        _spec(
            recipe=2,
            row="b",
            category="conversational",
            utterance="Could you set the kitchen herb garden cool light to 64 percent for me?",
            home=_home(herb_light, sayso_entity_area="Kitchen"),
            expected=_action(_light_set(herb_light, brightness=64)),
        ),
        _spec(
            recipe=3,
            row="a",
            category="entity_identity",
            utterance="Open the patio blinds",
            home=_home(patio_blinds, sayso_entity_area="Patio"),
            expected=_action(_turn_on(patio_blinds)),
        ),
        _spec(
            recipe=3,
            row="b",
            category="entity_identity",
            utterance="Lock the patio door",
            home=_home(patio_lock, sayso_entity_area="Patio"),
            expected=_action(_turn_on(patio_lock)),
        ),
        _spec(
            recipe=3,
            row="c",
            category="entity_identity",
            utterance="turn on office main light",
            home=_home(office_main, sayso_entity_area="Office"),
            expected=_action(_turn_on(office_main)),
        ),
        _spec(
            recipe=3,
            row="d",
            category="entity_identity",
            utterance="unlock joe's guest room door lock",
            home=_home(joes_lock, sayso_entity_area="Guest Room"),
            expected=_action(_turn_off(joes_lock)),
        ),
        _spec(
            recipe=3,
            row="e",
            category="entity_identity",
            utterance="Turn on Joe's Kitchen Light",
            home=_home(joes_kitchen, sayso_entity_area="Kitchen"),
            expected=_action(_turn_on(joes_kitchen)),
        ),
        _spec(
            recipe=3,
            row="f",
            category="entity_identity",
            utterance="Close O'Malley's Study Blinds",
            home=_home(omalleys_blinds, sayso_entity_area="Office"),
            expected=_action(_turn_off(omalleys_blinds)),
        ),
        _spec(
            recipe=3,
            row="g",
            category="entity_identity",
            utterance="Turn off Kids' Room Light",
            home=_home(kids_light, sayso_entity_area="Guest Room"),
            expected=_action(_turn_off(kids_light)),
        ),
        _spec(
            recipe=3,
            row="h",
            category="entity_identity",
            utterance="Turn on Kitchen North Light",
            home=_home(kitchen_north, sayso_entity_area="Kitchen"),
            expected=_action(_turn_on(kitchen_north)),
        ),
        _spec(
            recipe=4,
            row="a",
            category="multi_action_exclusion",
            utterance=(
                "Set Kitchen Ceiling Cool Light to 40 percent and turn off Hallway East Outlet, "
                "but leave Office Main Light alone"
            ),
            home=_home(kitchen_ceiling, hallway_east, office_main, sayso_entity_area="Kitchen"),
            expected=_action(_light_set(kitchen_ceiling, brightness=40), _turn_off(hallway_east)),
            target_names=["Kitchen Ceiling Cool Light", "Hallway East Outlet"],
        ),
        _spec(
            recipe=4,
            row="b",
            category="multi_action_exclusion",
            utterance=(
                "Open Patio South Blinds and lock Patio Side Door Lock, but leave Garage West Fan alone"
            ),
            home=_home(patio_blinds, patio_lock, garage_west_fan, sayso_entity_area="Patio"),
            expected=_action(_turn_on(patio_blinds), _turn_on(patio_lock)),
            target_names=["Patio South Blinds", "Patio Side Door Lock"],
        ),
        _spec(
            recipe=4,
            row="c",
            category="multi_action_exclusion",
            utterance=(
                "Turn on Nursery East Outlet and turn off Living Room Ceiling Fan, "
                "but leave Joe's Kitchen Light alone"
            ),
            home=_home(nursery_outlet, living_fan, joes_kitchen, sayso_entity_area="Nursery"),
            expected=_action(_turn_on(nursery_outlet), _turn_off(living_fan)),
            target_names=["Nursery East Outlet", "Living Room Ceiling Fan"],
        ),
        _spec(
            recipe=4,
            row="d",
            category="multi_action_exclusion",
            utterance=(
                "Open Joe's Workshop Blinds, close Primary Bedroom Corner Garage Door, "
                "and lock Patio Side Door Lock, but leave Garage Ceiling Fan alone"
            ),
            home=_home(
                workshop_blinds,
                primary_bedroom_garage,
                patio_lock,
                garage_ceiling_fan,
                sayso_entity_area="Workshop",
            ),
            expected=_action(
                _turn_on(workshop_blinds),
                _turn_off(primary_bedroom_garage),
                _turn_on(patio_lock),
            ),
            target_names=[
                "Joe's Workshop Blinds",
                "Primary Bedroom Corner Garage Door",
                "Patio Side Door Lock",
            ],
        ),
        _spec(
            recipe=5,
            row="garage_van",
            category="stt_corrupted",
            utterance="Turn on the garage west van",
            home=_home(garage_west_fan, sayso_entity_area="Garage"),
            expected=_action(_turn_on(garage_west_fan)),
        ),
        _spec(
            recipe=5,
            row="a",
            category="stt_corrupted",
            utterance="Uh unlock basement door lok please",
            home=_home(basement_south_lock, sayso_entity_area="Basement"),
            expected=_action(_turn_off(basement_south_lock)),
        ),
        _spec(
            recipe=5,
            row="b",
            category="stt_corrupted",
            utterance="tern on office main lite",
            home=_home(office_main, sayso_entity_area="Office"),
            expected=_action(_turn_on(office_main)),
        ),
        _spec(
            recipe=5,
            row="c",
            category="stt_corrupted",
            utterance="close the patio south blends",
            home=_home(patio_blinds, sayso_entity_area="Patio"),
            expected=_action(_turn_off(patio_blinds)),
        ),
        _spec(
            recipe=5,
            row="d",
            category="stt_corrupted",
            utterance="lok joe's guest room door",
            home=_home(joes_lock, sayso_entity_area="Guest Room"),
            expected=_action(_turn_on(joes_lock)),
        ),
        _spec(
            recipe=5,
            row="e",
            category="stt_corrupted",
            utterance="turn off basement van",
            home=_home(basement_south_fan, sayso_entity_area="Basement"),
            expected=_action(_turn_off(basement_south_fan)),
        ),
        _spec(
            recipe=6,
            row="a",
            category="status",
            utterance="Check the status of Patio South Blinds",
            home=_home(patio_blinds, sayso_entity_area="Patio"),
            expected=_status(patio_blinds),
            target_names=["Patio South Blinds"],
        ),
        _spec(
            recipe=6,
            row="b",
            category="status",
            utterance="Is the Workshop West Fan running?",
            home=_home(workshop_west_fan, sayso_entity_area="Workshop"),
            expected=_status(workshop_west_fan),
            target_names=["Workshop West Fan"],
        ),
        _spec(
            recipe=6,
            row="c",
            category="status",
            utterance="What's Joe's Guest Room Door Lock doing?",
            home=_home(joes_lock, sayso_entity_area="Guest Room"),
            expected=_status(joes_lock),
            target_names=["Joe's Guest Room Door Lock"],
        ),
        _spec(
            recipe=6,
            row="d",
            category="status",
            utterance="Is Kitchen North Light off?",
            home=_home(kitchen_north, sayso_entity_area="Kitchen"),
            expected=_status({**kitchen_north, "state": "off"}),
            target_names=["Kitchen North Light"],
        ),
        _spec(
            recipe=7,
            row="a",
            category="ambiguity",
            utterance="Turn on the light",
            home=_home(kitchen_sink, office_main, sayso_entity_area="Kitchen"),
            expected=_action(_turn_on(kitchen_sink)),
        ),
        _spec(
            recipe=7,
            row="b",
            category="ambiguity",
            utterance="Turn on the light",
            home=_home(kitchen_sink, kitchen_ceiling, office_main, sayso_entity_area="Kitchen"),
            expected=_no_action("clarify"),
        ),
        _spec(
            recipe=7,
            row="c",
            category="ambiguity",
            utterance="Turn on the office light",
            home=_home(office_main, sayso_entity_area="Kitchen"),
            expected=_action(_turn_on(office_main)),
        ),
        _spec(
            recipe=7,
            row="d",
            category="ambiguity",
            utterance="Turn on the kitchen light",
            home=_home(kitchen_sink, kitchen_ceiling, sayso_entity_area="Kitchen"),
            expected=_no_action("clarify"),
        ),
        _spec(
            recipe=7,
            row="e",
            category="ambiguity",
            utterance="Turn off the fan",
            home=_home(living_fan, workshop_fan, sayso_entity_area="Living Room"),
            expected=_action(_turn_off(living_fan)),
        ),
        _spec(
            recipe=7,
            row="f",
            category="ambiguity",
            utterance="Turn on the outlet",
            home=_home(hallway_east, hallway_west, nursery_outlet, sayso_entity_area="Hallway"),
            expected=_no_action("clarify"),
        ),
        _spec(
            recipe=7,
            row="g",
            category="ambiguity",
            utterance="Open the blinds",
            home=_home(patio_blinds, workshop_blinds, sayso_entity_area="Patio"),
            expected=_action(_turn_on(patio_blinds)),
        ),
        _spec(
            recipe=7,
            row="h",
            category="ambiguity",
            utterance="Lock the door",
            home=_home(kitchen_lock, patio_lock, sayso_entity_area="Kitchen"),
            expected=_action(_turn_on(kitchen_lock)),
        ),
        _spec(
            recipe=7,
            row="i",
            category="ambiguity",
            utterance="Turn on the light",
            home=_home(office_main, sayso_entity_area="Kitchen"),
            expected=_no_action(
                "area_unavailable",
                unavailable={"area": "kitchen", "type": "lights"},
            ),
        ),
        _spec(
            recipe=8,
            row="refuse",
            category="unsupported_no_action",
            utterance="Disable the smoke alarm safety system",
            home=_home(office_main, sayso_entity_area="Office"),
            expected=_no_action("refuse"),
            request_hint="disable the smoke alarm safety system",
        ),
        _spec(
            recipe=8,
            row="clarify",
            category="unsupported_no_action",
            utterance="Set the light to",
            home=_home(office_main, sayso_entity_area="Office"),
            expected=_no_action("clarify"),
            request_hint="set the light to",
        ),
        _spec(
            recipe=8,
            row="unsupported",
            category="unsupported_no_action",
            utterance="Play music in the garage",
            home=_home(garage_west_fan, sayso_entity_area="Garage"),
            expected=_no_action("unsupported"),
            request_hint="play music in the garage",
        ),
    ]
    for index, row in enumerate(rows):
        if row["recipe"] == 4 and row["recipe_row"] == "a":
            row["excluded_names"] = ["Office Main Light"]
        if row["recipe"] == 4 and row["recipe_row"] == "b":
            row["excluded_names"] = ["Garage West Fan"]
        if row["recipe"] == 4 and row["recipe_row"] == "c":
            row["excluded_names"] = ["Joe's Kitchen Light"]
        if row["recipe"] == 4 and row["recipe_row"] == "d":
            row["excluded_names"] = ["Garage Ceiling Fan"]
        row["home"]["home_id"] = f"recipe_lock_{index:03d}"
    return rows


def quality_eval_user_prompts() -> set[str]:
    return {spec["utterance"] for spec in locked_specs()}


def build_quality_eval_examples() -> list[dict[str, Any]]:
    """Render locked specs into canonical SaySo JSONL gold rows."""
    examples: list[dict[str, Any]] = []
    for spec in locked_specs():
        example = render_example(spec)
        example["metadata"]["quality_eval"] = True
        example["metadata"]["recipe"] = spec["recipe"]
        example["metadata"]["recipe_row"] = spec["recipe_row"]
        examples.append(example)
    return examples


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("’", "'")))


def expected_tool_calls(example: dict[str, Any]) -> list[dict[str, Any]]:
    batches = extract_assistant_tool_calls(example.get("messages") or [])
    return [call for batch in batches for call in batch]


def score_quality_gold(example: dict[str, Any], actual_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Score exact tool name/args and no-call-when-expected for one gold row."""
    expected_messages = example.get("messages") or []
    expected_calls = expected_tool_calls(example)
    actual_calls = expected_tool_calls({"messages": actual_messages})
    name_ok, args_ok, _multi_ok, category = score_expected_vs_actual(
        expected_messages,
        actual_messages,
    )
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
    """Validate one rendered gold row against the first-training contract."""
    blob = json.dumps(example, ensure_ascii=False)
    if _BANNED.search(blob):
        raise ValueError("quality eval contains banned eval or ChatML tool-call markers")
    metadata = example.get("metadata") or {}
    if str(metadata.get("candidate_id", "")).startswith("evals/cases"):
        raise ValueError("quality eval must not use evals/cases IDs")
    user = next(message for message in example["messages"] if message.get("role") == "user")
    if "thermostat" in _normalized(str(user.get("content", ""))):
        raise ValueError("thermostat rows are omitted from first training")
    for call in expected_tool_calls(example):
        args = call.get("function", {}).get("arguments")
        if not isinstance(args, str):
            raise ValueError("quality eval keeps function.arguments as JSON strings")
        parsed = parse_tool_arguments(args)
        if parsed is None:
            raise ValueError("quality eval arguments must parse as JSON objects")


def recipe_lock_summary() -> dict[str, Any]:
    specs = locked_specs()
    return {
        "gold_count": len(specs),
        "recipes": sorted({spec["recipe"] for spec in specs}),
        "categories": sorted({spec["category"] for spec in specs}),
        "user_prompts": sorted(quality_eval_user_prompts()),
    }
