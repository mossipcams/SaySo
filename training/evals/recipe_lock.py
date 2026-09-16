"""Human-locked gold eval rows (recipes 1–8, no thermostat)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_synthetic_dataset import render_example  # noqa: E402
from evals.specs import (  # noqa: E402
    action as _action,
    assert_row_contract,
    entity as _entity,
    expected_tool_calls,
    fan_speed as _fan_speed,
    home as _home_base,
    light_set,
    no_action as _no_action,
    normalized as _normalized,
    score_quality_gold,
    spec as _spec_base,
    status as _status,
    turn_off as _turn_off,
    turn_on as _turn_on,
)

_LOCK_HOME_ID = "recipe_lock_home"


def _home(*entities: dict[str, Any], sayso_entity_area: str) -> dict[str, Any]:
    """Every locked row uses one home id; only its contents vary."""
    return _home_base(
        *entities, sayso_entity_area=sayso_entity_area, home_id=_LOCK_HOME_ID
    )


def _light_set(entity: dict[str, Any], *, brightness: int) -> dict[str, Any]:
    return light_set(entity, brightness)


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
    """One locked row, addressed by the recipe and lettered row it came from."""
    return _spec_base(
        candidate_id=f"recipe_lock_{recipe:02d}_{row}",
        category=category,
        subcategory=row,
        utterance=utterance,
        home=home,
        expected=expected,
        target_names=target_names or None,
        request_hint=request_hint,
        recipe=recipe,
        recipe_row=row,
    )


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


def assert_quality_eval_contract(example: dict[str, Any]) -> None:
    """Validate one rendered gold row against the first-training contract."""
    assert_row_contract(example, "quality eval")
    user = next(m for m in example["messages"] if m.get("role") == "user")
    if "thermostat" in _normalized(str(user.get("content", ""))):
        raise ValueError("thermostat rows are omitted from first training")


def recipe_lock_summary() -> dict[str, Any]:
    specs = locked_specs()
    return {
        "gold_count": len(specs),
        "recipes": sorted({spec["recipe"] for spec in specs}),
        "categories": sorted({spec["category"] for spec in specs}),
        "user_prompts": sorted(quality_eval_user_prompts()),
    }
