"""Core evaluation corpus loader, authorship, and validation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sayso_server.control_plan import ActionPlan, ControlPlan, QueryPlan
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.models import Scope, ScopeKind
from sayso_server.resolver import resolve_entity_ids

from evals.schema import EvalCase, ExpectedOutcome, load_eval_cases_jsonl

EVALS_DIR = Path(__file__).resolve().parent
CORE_DATASET_PATH = EVALS_DIR / "datasets" / "core.jsonl"
HOME_GRAPH_PATH = EVALS_DIR / "fixtures" / "home_graph.json"

CORE_CATEGORY_COUNTS: dict[str, int] = {
    "simple_control": 20,
    "room_relative": 20,
    "multi_device": 20,
    "scene": 15,
    "script": 15,
    "climate": 15,
    "query": 15,
}
CORE_CASE_COUNT = sum(CORE_CATEGORY_COUNTS.values())

_HOME = "eval-home"
_ORIGIN = "area_living_room"
_ALL_ENTITIES = (
    "light.living_room_ceiling",
    "light.floor_lamp",
    "climate.downstairs",
    "binary_sensor.front_door",
    "scene.movie_time",
    "script.good_night",
)
_LIVING_ROOM_ENTITIES = (
    "light.living_room_ceiling",
    "light.floor_lamp",
    "climate.downstairs",
    "binary_sensor.front_door",
    "scene.movie_time",
)
_LIVING_ROOM_LIGHTS = ("light.living_room_ceiling", "light.floor_lamp")


def load_home_graph() -> HomeGraphSnapshot:
    data = json.loads(HOME_GRAPH_PATH.read_text())
    return HomeGraphSnapshot.model_validate(data)


def load_home_graph_entity_ids() -> frozenset[str]:
    graph = load_home_graph()
    return frozenset(
        item.entity_id
        for item in (*graph.entities, *graph.scenes, *graph.scripts)
    )


def load_home_graph_origin_areas() -> frozenset[str]:
    graph = load_home_graph()
    return frozenset(area.id for area in graph.areas)


def load_core_corpus() -> list[EvalCase]:
    return load_eval_cases_jsonl(CORE_DATASET_PATH.read_text())


def validate_core_corpus(cases: Iterable[EvalCase]) -> None:
    case_list = list(cases)
    if len(case_list) != CORE_CASE_COUNT:
        msg = f"case count must be {CORE_CASE_COUNT}, got {len(case_list)}"
        raise ValueError(msg)

    counts = Counter(case.category for case in case_list)
    for category, expected in CORE_CATEGORY_COUNTS.items():
        if counts[category] != expected:
            msg = f"category {category!r} expected {expected} cases, got {counts[category]}"
            raise ValueError(msg)

    unknown = set(counts) - set(CORE_CATEGORY_COUNTS)
    if unknown:
        msg = f"unknown categories: {sorted(unknown)}"
        raise ValueError(msg)

    case_ids = [case.case_id for case in case_list]
    if len(case_ids) != len(set(case_ids)):
        msg = "case_id values must be unique"
        raise ValueError(msg)

    valid_origins = load_home_graph_origin_areas()
    valid_entities = load_home_graph_entity_ids()
    for case in case_list:
        if case.home != _HOME:
            msg = f"{case.case_id}: home must be {_HOME!r}, got {case.home!r}"
            raise ValueError(msg)
        if case.origin not in valid_origins:
            msg = f"{case.case_id}: unknown origin {case.origin!r}"
            raise ValueError(msg)
        for entity_id in (
            *case.expected_candidate_entities,
            *case.expected_resolved_entities,
        ):
            if entity_id not in valid_entities:
                msg = f"{case.case_id}: unknown entity {entity_id!r}"
                raise ValueError(msg)


def verify_expected_resolutions(cases: Iterable[EvalCase]) -> None:
    graph = load_home_graph()
    for case in cases:
        if case.expected_outcome not in {
            ExpectedOutcome.VALID_ACTION,
            ExpectedOutcome.VALID_QUERY,
        }:
            continue

        plan = ControlPlan.model_validate(case.expected_control_plan)
        if isinstance(plan, ActionPlan):
            scope = plan.scope
            if scope is None and (plan.targets or plan.include or plan.exclude):
                scope = Scope(kind=ScopeKind.CURRENT_AREA)
            resolved = resolve_entity_ids(
                graph,
                origin_area_id=case.origin,
                scope=scope,
                domain=plan.domain,
                targets=plan.targets,
                include=plan.include,
                exclude=plan.exclude,
            )
        elif isinstance(plan, QueryPlan):
            scope = plan.scope
            if scope is None and (plan.targets or plan.include or plan.exclude):
                scope = Scope(kind=ScopeKind.CURRENT_AREA)
            resolved = resolve_entity_ids(
                graph,
                origin_area_id=case.origin,
                scope=scope,
                domain=plan.domain,
                targets=plan.targets,
                include=plan.include,
                exclude=plan.exclude,
            )
        else:
            continue

        expected = frozenset(case.expected_resolved_entities)
        if resolved != expected:
            msg = (
                f"{case.case_id}: expected_resolved_entities {sorted(expected)} "
                f"!= resolver output {sorted(resolved)}"
            )
            raise ValueError(msg)


def _case(
    *,
    case_id: str,
    category: str,
    turns: list[str],
    plan: dict[str, Any],
    candidates: list[str],
    resolved: list[str],
    outcome: str,
    execution_allowed: bool = True,
    origin: str = _ORIGIN,
) -> dict[str, Any]:
    missing = set(resolved) - set(candidates)
    if missing:
        msg = f"{case_id}: candidates missing resolved entities {sorted(missing)}"
        raise ValueError(msg)
    return {
        "case_id": case_id,
        "category": category,
        "home": _HOME,
        "origin": origin,
        "turns": turns,
        "expected_control_plan": plan,
        "expected_candidate_entities": candidates,
        "expected_resolved_entities": resolved,
        "expected_outcome": outcome,
        "execution_allowed": execution_allowed,
    }


def _author_simple_control_cases() -> list[dict[str, Any]]:
    specs: list[tuple[str, list[str], dict[str, Any], list[str], list[str]]] = [
        (
            "Turn off the ceiling lights",
            ["Turn off the ceiling lights"],
            {
                "outcome": "action",
                "intent": "turn off the ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "off",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.living_room_ceiling"],
        ),
        (
            "Turn on the floor lamp",
            ["Turn on the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn on the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "on",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.floor_lamp"],
        ),
        (
            "Switch off the lamp",
            ["Switch off the lamp"],
            {
                "outcome": "action",
                "intent": "switch off the lamp",
                "domain": "light",
                "targets": ["lamp"],
                "state": "off",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.floor_lamp"],
        ),
        (
            "Turn on the reading lamp",
            ["Turn on the reading lamp"],
            {
                "outcome": "action",
                "intent": "turn on the reading lamp",
                "domain": "light",
                "targets": ["reading lamp"],
                "state": "on",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.floor_lamp"],
        ),
        (
            "Dim the ceiling lights to fifty percent",
            ["Dim the ceiling lights to fifty percent"],
            {
                "outcome": "action",
                "intent": "dim the ceiling lights to fifty percent",
                "domain": "light",
                "targets": ["ceiling lights"],
                "value": 50,
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.living_room_ceiling"],
        ),
        (
            "Set the floor lamp brightness to seventy five",
            ["Set the floor lamp brightness to seventy five"],
            {
                "outcome": "action",
                "intent": "set the floor lamp brightness to seventy five",
                "domain": "light",
                "targets": ["floor lamp"],
                "value": 75,
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.floor_lamp"],
        ),
        (
            "Toggle the floor lamp",
            ["Toggle the floor lamp"],
            {
                "outcome": "action",
                "intent": "toggle the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "toggle",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.floor_lamp"],
        ),
        (
            "Turn off the overhead lights",
            ["Turn off the overhead lights"],
            {
                "outcome": "action",
                "intent": "turn off the overhead lights",
                "domain": "light",
                "targets": ["overhead lights"],
                "state": "off",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.living_room_ceiling"],
        ),
        (
            "Turn on living room ceiling",
            ["Turn on living room ceiling"],
            {
                "outcome": "action",
                "intent": "turn on living room ceiling",
                "domain": "light",
                "targets": ["living room ceiling"],
                "state": "on",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.living_room_ceiling"],
        ),
        (
            "Switch off ceiling lights",
            ["Switch off ceiling lights"],
            {
                "outcome": "action",
                "intent": "switch off ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "off",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.living_room_ceiling"],
        ),
        (
            "Brighten the floor lamp to forty",
            ["Brighten the floor lamp to forty"],
            {
                "outcome": "action",
                "intent": "brighten the floor lamp to forty",
                "domain": "light",
                "targets": ["floor lamp"],
                "value": 40,
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.floor_lamp"],
        ),
        (
            "Turn the lamp on",
            ["Turn the lamp on"],
            {
                "outcome": "action",
                "intent": "turn the lamp on",
                "domain": "light",
                "targets": ["lamp"],
                "state": "on",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.floor_lamp"],
        ),
        (
            "Shut off the floor lamp",
            ["Shut off the floor lamp"],
            {
                "outcome": "action",
                "intent": "shut off the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "off",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.floor_lamp"],
        ),
        (
            "Set ceiling lights to twenty five percent",
            ["Set ceiling lights to twenty five percent"],
            {
                "outcome": "action",
                "intent": "set ceiling lights to twenty five percent",
                "domain": "light",
                "targets": ["ceiling lights"],
                "value": 25,
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.living_room_ceiling"],
        ),
        (
            "Turn on the ceiling",
            ["Turn on the ceiling"],
            {
                "outcome": "action",
                "intent": "turn on the ceiling",
                "domain": "light",
                "targets": ["living room ceiling"],
                "state": "on",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.living_room_ceiling"],
        ),
        (
            "Turn off living room ceiling light",
            ["Turn off living room ceiling light"],
            {
                "outcome": "action",
                "intent": "turn off living room ceiling light",
                "domain": "light",
                "targets": ["living room ceiling"],
                "state": "off",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.living_room_ceiling"],
        ),
        (
            "Dim floor lamp to ten",
            ["Dim floor lamp to ten"],
            {
                "outcome": "action",
                "intent": "dim floor lamp to ten",
                "domain": "light",
                "targets": ["floor lamp"],
                "value": 10,
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.floor_lamp"],
        ),
        (
            "Toggle ceiling lights",
            ["Toggle ceiling lights"],
            {
                "outcome": "action",
                "intent": "toggle ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "toggle",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.living_room_ceiling"],
        ),
        (
            "Switch on the overhead lights",
            ["Switch on the overhead lights"],
            {
                "outcome": "action",
                "intent": "switch on the overhead lights",
                "domain": "light",
                "targets": ["overhead lights"],
                "state": "on",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.living_room_ceiling"],
        ),
        (
            "Turn the floor lamp off",
            ["Turn the floor lamp off"],
            {
                "outcome": "action",
                "intent": "turn the floor lamp off",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "off",
            },
            list(_LIVING_ROOM_LIGHTS),
            ["light.floor_lamp"],
        ),
    ]
    return [
        _case(
            case_id=f"simple_control-{index:03d}",
            category="simple_control",
            turns=turns,
            plan=plan,
            candidates=candidates,
            resolved=resolved,
            outcome="valid_action",
        )
        for index, (_title, turns, plan, candidates, resolved) in enumerate(specs, start=1)
    ]


def _author_room_relative_cases() -> list[dict[str, Any]]:
    living_candidates = list(_LIVING_ROOM_ENTITIES)
    living_lights = list(_LIVING_ROOM_LIGHTS)
    all_candidates = list(_ALL_ENTITIES)
    specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
        (
            ["Turn off the lights in here"],
            {
                "outcome": "action",
                "intent": "turn off the lights in here",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn on the lights in the living room"],
            {
                "outcome": "action",
                "intent": "turn on the lights in the living room",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn off lounge lights"],
            {
                "outcome": "action",
                "intent": "turn off lounge lights",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "lounge"},
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn on family room lights"],
            {
                "outcome": "action",
                "intent": "turn on family room lights",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "family room"},
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn off lights downstairs"],
            {
                "outcome": "action",
                "intent": "turn off lights downstairs",
                "domain": "light",
                "scope": {"kind": "floor", "name": "downstairs"},
                "state": "off",
            },
            all_candidates,
            living_lights,
        ),
        (
            ["Run good night upstairs"],
            {
                "outcome": "action",
                "intent": "run good night upstairs",
                "domain": "script",
                "scope": {"kind": "floor", "name": "upstairs"},
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Turn off everything in here"],
            {
                "outcome": "action",
                "intent": "turn off everything in here",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Dim the lights in this room to thirty"],
            {
                "outcome": "action",
                "intent": "dim the lights in this room to thirty",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "value": 30,
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Activate movie time in the living room"],
            {
                "outcome": "action",
                "intent": "activate movie time in the living room",
                "domain": "scene",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "targets": ["movie time"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Run bedtime in the primary bedroom"],
            {
                "outcome": "action",
                "intent": "run bedtime in the primary bedroom",
                "domain": "script",
                "scope": {"kind": "named_area", "name": "Primary Bedroom"},
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Turn off lights on the ground floor"],
            {
                "outcome": "action",
                "intent": "turn off lights on the ground floor",
                "domain": "light",
                "scope": {"kind": "floor", "name": "ground floor"},
                "state": "off",
            },
            all_candidates,
            living_lights,
        ),
        (
            ["Set the thermostat in here to seventy two"],
            {
                "outcome": "action",
                "intent": "set the thermostat in here to seventy two",
                "domain": "climate",
                "scope": {"kind": "current_area"},
                "targets": ["thermostat"],
                "value": 72,
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Turn off lights in area_living_room"],
            {
                "outcome": "action",
                "intent": "turn off lights in area_living_room",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "area_living_room"},
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn on lights in this room"],
            {
                "outcome": "action",
                "intent": "turn on lights in this room",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn off lights on floor_ground"],
            {
                "outcome": "action",
                "intent": "turn off lights on floor_ground",
                "domain": "light",
                "scope": {"kind": "floor", "name": "floor_ground"},
                "state": "off",
            },
            all_candidates,
            living_lights,
        ),
        (
            ["Run good night on floor_upper"],
            {
                "outcome": "action",
                "intent": "run good night on floor_upper",
                "domain": "script",
                "scope": {"kind": "floor", "name": "floor_upper"},
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Turn off the lights here"],
            {
                "outcome": "action",
                "intent": "turn off the lights here",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn on lights in the lounge"],
            {
                "outcome": "action",
                "intent": "turn on lights in the lounge",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "lounge"},
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn off lights in the living room"],
            {
                "outcome": "action",
                "intent": "turn off lights in the living room",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "living room"},
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Start movie mode in here"],
            {
                "outcome": "action",
                "intent": "start movie mode in here",
                "domain": "scene",
                "scope": {"kind": "current_area"},
                "targets": ["movie mode"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
    ]
    return [
        _case(
            case_id=f"room_relative-{index:03d}",
            category="room_relative",
            turns=turns,
            plan=plan,
            candidates=candidates,
            resolved=resolved,
            outcome="valid_action",
        )
        for index, (turns, plan, candidates, resolved) in enumerate(specs, start=1)
    ]


def _author_multi_device_cases() -> list[dict[str, Any]]:
    living_candidates = list(_LIVING_ROOM_ENTITIES)
    living_lights = list(_LIVING_ROOM_LIGHTS)
    specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
        (
            ["Turn off all the lights in the living room"],
            {
                "outcome": "action",
                "intent": "turn off all the lights in the living room",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn on all lights in here"],
            {
                "outcome": "action",
                "intent": "turn on all lights in here",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn off the ceiling and floor lamp"],
            {
                "outcome": "action",
                "intent": "turn off the ceiling and floor lamp",
                "domain": "light",
                "targets": ["ceiling lights", "floor lamp"],
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn on both lights"],
            {
                "outcome": "action",
                "intent": "turn on both lights",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Dim all lights in here to forty"],
            {
                "outcome": "action",
                "intent": "dim all lights in here to forty",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "value": 40,
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn off every light in the lounge"],
            {
                "outcome": "action",
                "intent": "turn off every light in the lounge",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "lounge"},
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Switch on all living room lights"],
            {
                "outcome": "action",
                "intent": "switch on all living room lights",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "living room"},
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn off all lights downstairs"],
            {
                "outcome": "action",
                "intent": "turn off all lights downstairs",
                "domain": "light",
                "scope": {"kind": "floor", "name": "downstairs"},
                "state": "off",
            },
            list(_ALL_ENTITIES),
            living_lights,
        ),
        (
            ["Turn on the ceiling lights and floor lamp"],
            {
                "outcome": "action",
                "intent": "turn on the ceiling lights and floor lamp",
                "domain": "light",
                "targets": ["ceiling lights", "floor lamp"],
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn off all the lights here except none"],
            {
                "outcome": "action",
                "intent": "turn off all the lights here",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn on overhead and reading lights"],
            {
                "outcome": "action",
                "intent": "turn on overhead and reading lights",
                "domain": "light",
                "targets": ["overhead lights", "reading lamp"],
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Shut off all lights in this room"],
            {
                "outcome": "action",
                "intent": "shut off all lights in this room",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn on all lights in the family room"],
            {
                "outcome": "action",
                "intent": "turn on all lights in the family room",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "family room"},
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn off lights in here except the lamp"],
            {
                "outcome": "action",
                "intent": "turn off lights in here except the lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn off all lights except the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn off all lights except the floor lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["floor lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Include ceiling lights and floor lamp"],
            {
                "outcome": "action",
                "intent": "include ceiling lights and floor lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "include": ["ceiling lights", "floor lamp"],
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn on the ceiling lights and reading lamp"],
            {
                "outcome": "action",
                "intent": "turn on the ceiling lights and reading lamp",
                "domain": "light",
                "targets": ["ceiling lights", "reading lamp"],
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Turn off every light downstairs"],
            {
                "outcome": "action",
                "intent": "turn off every light downstairs",
                "domain": "light",
                "scope": {"kind": "floor", "name": "downstairs"},
                "state": "off",
            },
            list(_ALL_ENTITIES),
            living_lights,
        ),
        (
            ["Turn on all lights on the ground floor"],
            {
                "outcome": "action",
                "intent": "turn on all lights on the ground floor",
                "domain": "light",
                "scope": {"kind": "floor", "name": "ground floor"},
                "state": "on",
            },
            list(_ALL_ENTITIES),
            living_lights,
        ),
        (
            ["Turn off both living room lights"],
            {
                "outcome": "action",
                "intent": "turn off both living room lights",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "state": "off",
            },
            living_candidates,
            living_lights,
        ),
    ]
    return [
        _case(
            case_id=f"multi_device-{index:03d}",
            category="multi_device",
            turns=turns,
            plan=plan,
            candidates=candidates,
            resolved=resolved,
            outcome="valid_action",
        )
        for index, (turns, plan, candidates, resolved) in enumerate(specs, start=1)
    ]


def _author_scene_cases() -> list[dict[str, Any]]:
    living_candidates = list(_LIVING_ROOM_ENTITIES)
    specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
        (
            ["Start movie time"],
            {
                "outcome": "action",
                "intent": "start movie time",
                "domain": "scene",
                "targets": ["movie time"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Activate movie mode"],
            {
                "outcome": "action",
                "intent": "activate movie mode",
                "domain": "scene",
                "targets": ["movie mode"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Run cinema scene"],
            {
                "outcome": "action",
                "intent": "run cinema scene",
                "domain": "scene",
                "targets": ["cinema"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Start the movie time scene"],
            {
                "outcome": "action",
                "intent": "start the movie time scene",
                "domain": "scene",
                "targets": ["movie time"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Activate scene movie time"],
            {
                "outcome": "action",
                "intent": "activate scene movie time",
                "domain": "scene",
                "targets": ["movie time"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Begin movie mode"],
            {
                "outcome": "action",
                "intent": "begin movie mode",
                "domain": "scene",
                "targets": ["movie mode"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Launch cinema"],
            {
                "outcome": "action",
                "intent": "launch cinema",
                "domain": "scene",
                "targets": ["cinema"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Run movie time"],
            {
                "outcome": "action",
                "intent": "run movie time",
                "domain": "scene",
                "targets": ["movie time"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Start cinema scene"],
            {
                "outcome": "action",
                "intent": "start cinema scene",
                "domain": "scene",
                "targets": ["cinema"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Activate the movie time scene"],
            {
                "outcome": "action",
                "intent": "activate the movie time scene",
                "domain": "scene",
                "targets": ["movie time"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Enable movie mode"],
            {
                "outcome": "action",
                "intent": "enable movie mode",
                "domain": "scene",
                "targets": ["movie mode"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Trigger movie time"],
            {
                "outcome": "action",
                "intent": "trigger movie time",
                "domain": "scene",
                "targets": ["movie time"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Play cinema scene"],
            {
                "outcome": "action",
                "intent": "play cinema scene",
                "domain": "scene",
                "targets": ["cinema"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Start movie mode in here"],
            {
                "outcome": "action",
                "intent": "start movie mode in here",
                "domain": "scene",
                "scope": {"kind": "current_area"},
                "targets": ["movie mode"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
        (
            ["Activate movie time in the living room"],
            {
                "outcome": "action",
                "intent": "activate movie time in the living room",
                "domain": "scene",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "targets": ["movie time"],
                "state": "activate",
            },
            living_candidates,
            ["scene.movie_time"],
        ),
    ]
    return [
        _case(
            case_id=f"scene-{index:03d}",
            category="scene",
            turns=turns,
            plan=plan,
            candidates=candidates,
            resolved=resolved,
            outcome="valid_action",
        )
        for index, (turns, plan, candidates, resolved) in enumerate(specs, start=1)
    ]


def _author_script_cases() -> list[dict[str, Any]]:
    all_candidates = list(_ALL_ENTITIES)
    specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
        (
            ["Run good night"],
            {
                "outcome": "action",
                "intent": "run good night",
                "domain": "script",
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Start bedtime"],
            {
                "outcome": "action",
                "intent": "start bedtime",
                "domain": "script",
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Execute good night script"],
            {
                "outcome": "action",
                "intent": "execute good night script",
                "domain": "script",
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Run the bedtime script"],
            {
                "outcome": "action",
                "intent": "run the bedtime script",
                "domain": "script",
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Activate good night"],
            {
                "outcome": "action",
                "intent": "activate good night",
                "domain": "script",
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Trigger bedtime"],
            {
                "outcome": "action",
                "intent": "trigger bedtime",
                "domain": "script",
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Run good night upstairs"],
            {
                "outcome": "action",
                "intent": "run good night upstairs",
                "domain": "script",
                "scope": {"kind": "floor", "name": "upstairs"},
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Start good night in the primary bedroom"],
            {
                "outcome": "action",
                "intent": "start good night in the primary bedroom",
                "domain": "script",
                "scope": {"kind": "named_area", "name": "Primary Bedroom"},
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Run bedtime upstairs"],
            {
                "outcome": "action",
                "intent": "run bedtime upstairs",
                "domain": "script",
                "scope": {"kind": "floor", "name": "upstairs"},
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Execute bedtime on the upper floor"],
            {
                "outcome": "action",
                "intent": "execute bedtime on the upper floor",
                "domain": "script",
                "scope": {"kind": "floor", "name": "upper floor"},
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Launch good night"],
            {
                "outcome": "action",
                "intent": "launch good night",
                "domain": "script",
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Run the good night script"],
            {
                "outcome": "action",
                "intent": "run the good night script",
                "domain": "script",
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Start the bedtime routine"],
            {
                "outcome": "action",
                "intent": "start the bedtime routine",
                "domain": "script",
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Activate bedtime script"],
            {
                "outcome": "action",
                "intent": "activate bedtime script",
                "domain": "script",
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
        (
            ["Run good night on floor_upper"],
            {
                "outcome": "action",
                "intent": "run good night on floor_upper",
                "domain": "script",
                "scope": {"kind": "floor", "name": "floor_upper"},
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
        ),
    ]
    cases: list[dict[str, Any]] = []
    for index, (turns, plan, candidates, resolved) in enumerate(specs, start=1):
        if "scope" not in plan:
            plan = {**plan, "scope": {"kind": "all"}}
        cases.append(
            _case(
                case_id=f"script-{index:03d}",
                category="script",
                turns=turns,
                plan=plan,
                candidates=candidates,
                resolved=resolved,
                outcome="valid_action",
            )
        )
    return cases


def _author_climate_cases() -> list[dict[str, Any]]:
    living_candidates = list(_LIVING_ROOM_ENTITIES)
    specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
        (
            ["Set the thermostat to seventy two"],
            {
                "outcome": "action",
                "intent": "set the thermostat to seventy two",
                "domain": "climate",
                "targets": ["thermostat"],
                "value": 72,
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Turn on heat"],
            {
                "outcome": "action",
                "intent": "turn on heat",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "heat",
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Set thermostat to sixty eight"],
            {
                "outcome": "action",
                "intent": "set thermostat to sixty eight",
                "domain": "climate",
                "targets": ["thermostat"],
                "value": 68,
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Set the hvac to cool"],
            {
                "outcome": "action",
                "intent": "set the hvac to cool",
                "domain": "climate",
                "targets": ["hvac"],
                "mode": "cool",
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Turn off the thermostat"],
            {
                "outcome": "action",
                "intent": "turn off the thermostat",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "off",
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Set downstairs thermostat to seventy"],
            {
                "outcome": "action",
                "intent": "set downstairs thermostat to seventy",
                "domain": "climate",
                "targets": ["downstairs thermostat"],
                "value": 70,
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Set heat to seventy four"],
            {
                "outcome": "action",
                "intent": "set heat to seventy four",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "heat",
                "value": 74,
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Switch thermostat to auto"],
            {
                "outcome": "action",
                "intent": "switch thermostat to auto",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "auto",
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Set the temperature to seventy one"],
            {
                "outcome": "action",
                "intent": "set the temperature to seventy one",
                "domain": "climate",
                "targets": ["thermostat"],
                "value": 71,
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Turn on cooling"],
            {
                "outcome": "action",
                "intent": "turn on cooling",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "cool",
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Set thermostat in here to sixty nine"],
            {
                "outcome": "action",
                "intent": "set thermostat in here to sixty nine",
                "domain": "climate",
                "scope": {"kind": "current_area"},
                "targets": ["thermostat"],
                "value": 69,
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Set hvac to heat at seventy three"],
            {
                "outcome": "action",
                "intent": "set hvac to heat at seventy three",
                "domain": "climate",
                "targets": ["hvac"],
                "mode": "heat",
                "value": 73,
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Turn the thermostat off"],
            {
                "outcome": "action",
                "intent": "turn the thermostat off",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "off",
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Set downstairs to sixty seven degrees"],
            {
                "outcome": "action",
                "intent": "set downstairs to sixty seven degrees",
                "domain": "climate",
                "targets": ["downstairs"],
                "value": 67,
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Switch the hvac to auto mode"],
            {
                "outcome": "action",
                "intent": "switch the hvac to auto mode",
                "domain": "climate",
                "targets": ["hvac"],
                "mode": "auto",
            },
            living_candidates,
            ["climate.downstairs"],
        ),
    ]
    return [
        _case(
            case_id=f"climate-{index:03d}",
            category="climate",
            turns=turns,
            plan=plan,
            candidates=candidates,
            resolved=resolved,
            outcome="valid_action",
        )
        for index, (turns, plan, candidates, resolved) in enumerate(specs, start=1)
    ]


def _author_query_cases() -> list[dict[str, Any]]:
    living_candidates = list(_LIVING_ROOM_ENTITIES)
    living_lights = list(_LIVING_ROOM_LIGHTS)
    specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
        (
            ["Is the door closed"],
            {
                "outcome": "query",
                "intent": "is the door closed",
                "domain": "binary_sensor",
                "targets": ["door"],
            },
            living_candidates,
            ["binary_sensor.front_door"],
        ),
        (
            ["Are any lights on in here"],
            {
                "outcome": "query",
                "intent": "are any lights on in here",
                "domain": "light",
                "scope": {"kind": "current_area"},
            },
            living_candidates,
            living_lights,
        ),
        (
            ["What is the thermostat set to"],
            {
                "outcome": "query",
                "intent": "what is the thermostat set to",
                "domain": "climate",
                "targets": ["thermostat"],
                "attribute": "temperature",
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Is the front door open"],
            {
                "outcome": "query",
                "intent": "is the front door open",
                "domain": "binary_sensor",
                "targets": ["front door"],
            },
            living_candidates,
            ["binary_sensor.front_door"],
        ),
        (
            ["Are all lights on in the living room"],
            {
                "outcome": "query",
                "intent": "are all lights on in the living room",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "Living Room"},
            },
            living_candidates,
            living_lights,
        ),
        (
            ["What is the current temperature"],
            {
                "outcome": "query",
                "intent": "what is the current temperature",
                "domain": "climate",
                "targets": ["thermostat"],
                "attribute": "current_temperature",
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Is the floor lamp on"],
            {
                "outcome": "query",
                "intent": "is the floor lamp on",
                "domain": "light",
                "targets": ["floor lamp"],
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
        (
            ["Are any lights on downstairs"],
            {
                "outcome": "query",
                "intent": "are any lights on downstairs",
                "domain": "light",
                "scope": {"kind": "floor", "name": "downstairs"},
            },
            list(_ALL_ENTITIES),
            living_lights,
        ),
        (
            ["Is the ceiling light on"],
            {
                "outcome": "query",
                "intent": "is the ceiling light on",
                "domain": "light",
                "targets": ["ceiling lights"],
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["What is the hvac mode"],
            {
                "outcome": "query",
                "intent": "what is the hvac mode",
                "domain": "climate",
                "targets": ["hvac"],
                "attribute": "hvac_mode",
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Is the door open or closed"],
            {
                "outcome": "query",
                "intent": "is the door open or closed",
                "domain": "binary_sensor",
                "targets": ["door"],
            },
            living_candidates,
            ["binary_sensor.front_door"],
        ),
        (
            ["Check if any lights are on"],
            {
                "outcome": "query",
                "intent": "check if any lights are on",
                "domain": "light",
                "scope": {"kind": "current_area"},
            },
            living_candidates,
            living_lights,
        ),
        (
            ["Is the living room ceiling on"],
            {
                "outcome": "query",
                "intent": "is the living room ceiling on",
                "domain": "light",
                "targets": ["living room ceiling"],
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Are all the lights off in here"],
            {
                "outcome": "query",
                "intent": "are all the lights off in here",
                "domain": "light",
                "scope": {"kind": "current_area"},
            },
            living_candidates,
            living_lights,
        ),
        (
            ["What temperature is the thermostat reading"],
            {
                "outcome": "query",
                "intent": "what temperature is the thermostat reading",
                "domain": "climate",
                "targets": ["thermostat"],
                "attribute": "current_temperature",
            },
            living_candidates,
            ["climate.downstairs"],
        ),
    ]
    return [
        _case(
            case_id=f"query-{index:03d}",
            category="query",
            turns=turns,
            plan=plan,
            candidates=candidates,
            resolved=resolved,
            outcome="valid_query",
            execution_allowed=False,
        )
        for index, (turns, plan, candidates, resolved) in enumerate(specs, start=1)
    ]


def author_core_cases() -> list[dict[str, Any]]:
    cases = [
        *_author_simple_control_cases(),
        *_author_room_relative_cases(),
        *_author_multi_device_cases(),
        *_author_scene_cases(),
        *_author_script_cases(),
        *_author_climate_cases(),
        *_author_query_cases(),
    ]
    if len(cases) != CORE_CASE_COUNT:
        msg = f"authored {len(cases)} cases, expected {CORE_CASE_COUNT}"
        raise RuntimeError(msg)
    counts = Counter(case["category"] for case in cases)
    if dict(counts) != CORE_CATEGORY_COUNTS:
        msg = f"authored category counts {dict(counts)} != {CORE_CATEGORY_COUNTS}"
        raise RuntimeError(msg)
    parsed = [EvalCase.model_validate(case) for case in cases]
    validate_core_corpus(parsed)
    verify_expected_resolutions(parsed)
    return cases


def write_core_dataset(path: Path | None = None) -> Path:
    target = path or CORE_DATASET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    cases = author_core_cases()
    lines = [json.dumps(case, separators=(",", ":")) for case in cases]
    target.write_text("\n".join(lines) + "\n")
    return target


if __name__ == "__main__":
    written = write_core_dataset()
    print(f"wrote {CORE_CASE_COUNT} cases to {written}")
