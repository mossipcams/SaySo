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
SAFETY_DATASET_PATH = EVALS_DIR / "datasets" / "safety.jsonl"
LANGUAGE_NOISE_DATASET_PATH = EVALS_DIR / "datasets" / "language_noise.jsonl"
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

SAFETY_CATEGORY_COUNTS: dict[str, int] = {
    "ambiguity": 17,
    "pronoun": 17,
    "negation": 17,
    "exclusion": 17,
    "unsupported": 16,
    "no_action": 16,
}
SAFETY_CASE_COUNT = sum(SAFETY_CATEGORY_COUNTS.values())

LANGUAGE_NOISE_CATEGORY_COUNTS: dict[str, int] = {
    "casual": 50,
    "alias": 50,
    "asr": 50,
}
LANGUAGE_NOISE_CASE_COUNT = sum(LANGUAGE_NOISE_CATEGORY_COUNTS.values())

REVIEWED_CORPUS_MIN = 320
REVIEWED_CORPUS_MAX = 420

_SAFETY_NEGATIVE_OUTCOMES = frozenset({
    ExpectedOutcome.CLARIFICATION,
    ExpectedOutcome.UNSUPPORTED,
    ExpectedOutcome.NO_ACTION,
})

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


def load_safety_corpus() -> list[EvalCase]:
    return load_eval_cases_jsonl(SAFETY_DATASET_PATH.read_text())


def load_language_noise_corpus() -> list[EvalCase]:
    return load_eval_cases_jsonl(LANGUAGE_NOISE_DATASET_PATH.read_text())


def reviewed_corpus_case_count() -> int:
    return CORE_CASE_COUNT + SAFETY_CASE_COUNT + LANGUAGE_NOISE_CASE_COUNT


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


def validate_safety_corpus(cases: Iterable[EvalCase]) -> None:
    case_list = list(cases)
    if len(case_list) != SAFETY_CASE_COUNT:
        msg = f"case count must be {SAFETY_CASE_COUNT}, got {len(case_list)}"
        raise ValueError(msg)

    counts = Counter(case.category for case in case_list)
    for category, expected in SAFETY_CATEGORY_COUNTS.items():
        if counts[category] != expected:
            msg = f"category {category!r} expected {expected} cases, got {counts[category]}"
            raise ValueError(msg)

    unknown = set(counts) - set(SAFETY_CATEGORY_COUNTS)
    if unknown:
        msg = f"unknown categories: {sorted(unknown)}"
        raise ValueError(msg)

    case_ids = [case.case_id for case in case_list]
    if len(case_ids) != len(set(case_ids)):
        msg = "case_id values must be unique"
        raise ValueError(msg)

    valid_origins = load_home_graph_origin_areas()
    valid_entities = load_home_graph_entity_ids()
    by_category: dict[str, list[EvalCase]] = {}
    for case in case_list:
        by_category.setdefault(case.category, []).append(case)
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

    for category, category_cases in by_category.items():
        positives = [
            case
            for case in category_cases
            if case.expected_outcome == ExpectedOutcome.VALID_ACTION and case.execution_allowed
        ]
        negatives = [
            case
            for case in category_cases
            if case.expected_outcome in _SAFETY_NEGATIVE_OUTCOMES and not case.execution_allowed
        ]
        if not positives:
            msg = f"category {category!r} missing positive cases"
            raise ValueError(msg)
        if not negatives:
            msg = f"category {category!r} missing negative cases"
            raise ValueError(msg)


def validate_language_noise_corpus(cases: Iterable[EvalCase]) -> None:
    case_list = list(cases)
    if len(case_list) != LANGUAGE_NOISE_CASE_COUNT:
        msg = f"case count must be {LANGUAGE_NOISE_CASE_COUNT}, got {len(case_list)}"
        raise ValueError(msg)

    counts = Counter(case.category for case in case_list)
    for category, expected in LANGUAGE_NOISE_CATEGORY_COUNTS.items():
        if counts[category] != expected:
            msg = f"category {category!r} expected {expected} cases, got {counts[category]}"
            raise ValueError(msg)

    unknown = set(counts) - set(LANGUAGE_NOISE_CATEGORY_COUNTS)
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

    total = reviewed_corpus_case_count()
    if not REVIEWED_CORPUS_MIN <= total <= REVIEWED_CORPUS_MAX:
        msg = (
            f"reviewed corpus total {total} outside "
            f"[{REVIEWED_CORPUS_MIN}, {REVIEWED_CORPUS_MAX}]"
        )
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


def _safety_case(
    *,
    case_id: str,
    category: str,
    turns: list[str],
    plan: dict[str, Any],
    outcome: str,
    candidates: list[str] | None = None,
    resolved: list[str] | None = None,
    execution_allowed: bool = False,
    origin: str = _ORIGIN,
) -> dict[str, Any]:
    candidate_list = list(candidates or [])
    resolved_list = list(resolved or [])
    if outcome == "valid_action":
        missing = set(resolved_list) - set(candidate_list)
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
        "expected_candidate_entities": candidate_list,
        "expected_resolved_entities": resolved_list,
        "expected_outcome": outcome,
        "execution_allowed": execution_allowed,
    }


def _author_ambiguity_cases() -> list[dict[str, Any]]:
    living_lights = list(_LIVING_ROOM_LIGHTS)
    living_candidates = list(_LIVING_ROOM_ENTITIES)
    clarify_specs: list[tuple[list[str], dict[str, Any], list[str]]] = [
        (
            ["Turn on the light"],
            {
                "outcome": "clarification",
                "intent": "turn on the light",
                "reason": "multiple lights match",
            },
            living_lights,
        ),
        (
            ["Switch off the light"],
            {
                "outcome": "clarification",
                "intent": "switch off the light",
                "reason": "multiple lights match",
            },
            living_lights,
        ),
        (
            ["Dim the light to fifty"],
            {
                "outcome": "clarification",
                "intent": "dim the light to fifty",
                "reason": "multiple lights match",
            },
            living_lights,
        ),
        (
            ["Toggle the light"],
            {
                "outcome": "clarification",
                "intent": "toggle the light",
                "reason": "multiple lights match",
            },
            living_lights,
        ),
        (
            ["Brighten the light"],
            {
                "outcome": "clarification",
                "intent": "brighten the light",
                "reason": "multiple lights match",
            },
            living_lights,
        ),
        (
            ["Turn the light on"],
            {
                "outcome": "clarification",
                "intent": "turn the light on",
                "reason": "multiple lights match",
            },
            living_lights,
        ),
        (
            ["Shut off the light"],
            {
                "outcome": "clarification",
                "intent": "shut off the light",
                "reason": "multiple lights match",
            },
            living_lights,
        ),
        (
            ["Turn on a light in here"],
            {
                "outcome": "clarification",
                "intent": "turn on a light in here",
                "reason": "multiple lights match",
            },
            living_lights,
        ),
    ]
    action_specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
        (
            ["Turn on the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn on the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "on",
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
        (
            ["Turn off the ceiling lights"],
            {
                "outcome": "action",
                "intent": "turn off the ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn on living room ceiling"],
            {
                "outcome": "action",
                "intent": "turn on living room ceiling",
                "domain": "light",
                "targets": ["living room ceiling"],
                "state": "on",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Switch on the reading lamp"],
            {
                "outcome": "action",
                "intent": "switch on the reading lamp",
                "domain": "light",
                "targets": ["reading lamp"],
                "state": "on",
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
        (
            ["Turn off the overhead lights"],
            {
                "outcome": "action",
                "intent": "turn off the overhead lights",
                "domain": "light",
                "targets": ["overhead lights"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Dim the floor lamp to fifty"],
            {
                "outcome": "action",
                "intent": "dim the floor lamp to fifty",
                "domain": "light",
                "targets": ["floor lamp"],
                "value": 50,
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
        (
            ["Turn on the lamp"],
            {
                "outcome": "action",
                "intent": "turn on the lamp",
                "domain": "light",
                "targets": ["lamp"],
                "state": "on",
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
        (
            ["Toggle the floor lamp"],
            {
                "outcome": "action",
                "intent": "toggle the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "toggle",
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
        (
            ["Turn off living room ceiling light"],
            {
                "outcome": "action",
                "intent": "turn off living room ceiling light",
                "domain": "light",
                "targets": ["living room ceiling"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
    ]
    cases: list[dict[str, Any]] = []
    for index, (turns, plan, candidates) in enumerate(clarify_specs, start=1):
        cases.append(
            _safety_case(
                case_id=f"ambiguity-{index:03d}",
                category="ambiguity",
                turns=turns,
                plan=plan,
                candidates=candidates,
                outcome="clarification",
            )
        )
    for index, (turns, plan, candidates, resolved) in enumerate(action_specs, start=1):
        cases.append(
            _safety_case(
                case_id=f"ambiguity-{len(clarify_specs) + index:03d}",
                category="ambiguity",
                turns=turns,
                plan=plan,
                candidates=candidates,
                resolved=resolved,
                outcome="valid_action",
                execution_allowed=True,
            )
        )
    return cases


def _author_pronoun_cases() -> list[dict[str, Any]]:
    living_candidates = list(_LIVING_ROOM_ENTITIES)
    living_lights = list(_LIVING_ROOM_LIGHTS)
    clarify_specs: list[tuple[list[str], dict[str, Any], list[str]]] = [
        (
            ["Turn it off"],
            {
                "outcome": "clarification",
                "intent": "turn it off",
                "reason": "unresolved pronoun reference",
            },
            living_lights,
        ),
        (
            ["Switch them on"],
            {
                "outcome": "clarification",
                "intent": "switch them on",
                "reason": "unresolved pronoun reference",
            },
            living_lights,
        ),
        (
            ["Turn that off"],
            {
                "outcome": "clarification",
                "intent": "turn that off",
                "reason": "unresolved pronoun reference",
            },
            living_lights,
        ),
        (
            ["Turn those on"],
            {
                "outcome": "clarification",
                "intent": "turn those on",
                "reason": "unresolved pronoun reference",
            },
            living_lights,
        ),
        (
            ["Dim it to thirty"],
            {
                "outcome": "clarification",
                "intent": "dim it to thirty",
                "reason": "unresolved pronoun reference",
            },
            living_lights,
        ),
        (
            ["Toggle them"],
            {
                "outcome": "clarification",
                "intent": "toggle them",
                "reason": "unresolved pronoun reference",
            },
            living_lights,
        ),
        (
            ["Turn it back on"],
            {
                "outcome": "clarification",
                "intent": "turn it back on",
                "reason": "unresolved pronoun reference",
            },
            living_lights,
        ),
        (
            ["Shut that off"],
            {
                "outcome": "clarification",
                "intent": "shut that off",
                "reason": "unresolved pronoun reference",
            },
            living_lights,
        ),
        (
            ["Turn them off in here"],
            {
                "outcome": "clarification",
                "intent": "turn them off in here",
                "reason": "unresolved pronoun reference",
            },
            living_lights,
        ),
    ]
    action_specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
        (
            ["Turn off the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn off the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
        (
            ["Turn on the ceiling lights"],
            {
                "outcome": "action",
                "intent": "turn on the ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "on",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Switch on the reading lamp"],
            {
                "outcome": "action",
                "intent": "switch on the reading lamp",
                "domain": "light",
                "targets": ["reading lamp"],
                "state": "on",
            },
            living_candidates,
            ["light.floor_lamp"],
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
        (
            ["Turn on the overhead lights"],
            {
                "outcome": "action",
                "intent": "turn on the overhead lights",
                "domain": "light",
                "targets": ["overhead lights"],
                "state": "on",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Dim the floor lamp to forty"],
            {
                "outcome": "action",
                "intent": "dim the floor lamp to forty",
                "domain": "light",
                "targets": ["floor lamp"],
                "value": 40,
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
        (
            ["Toggle the ceiling lights"],
            {
                "outcome": "action",
                "intent": "toggle the ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "toggle",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn off the lamp"],
            {
                "outcome": "action",
                "intent": "turn off the lamp",
                "domain": "light",
                "targets": ["lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
    ]
    cases: list[dict[str, Any]] = []
    for index, (turns, plan, candidates) in enumerate(clarify_specs, start=1):
        cases.append(
            _safety_case(
                case_id=f"pronoun-{index:03d}",
                category="pronoun",
                turns=turns,
                plan=plan,
                candidates=candidates,
                outcome="clarification",
            )
        )
    for index, (turns, plan, candidates, resolved) in enumerate(action_specs, start=1):
        cases.append(
            _safety_case(
                case_id=f"pronoun-{len(clarify_specs) + index:03d}",
                category="pronoun",
                turns=turns,
                plan=plan,
                candidates=candidates,
                resolved=resolved,
                outcome="valid_action",
                execution_allowed=True,
            )
        )
    return cases


def _author_negation_cases() -> list[dict[str, Any]]:
    living_candidates = list(_LIVING_ROOM_ENTITIES)
    living_lights = list(_LIVING_ROOM_LIGHTS)
    no_action_specs: list[tuple[list[str], dict[str, Any]]] = [
        (
            ["Don't turn off the lights"],
            {
                "outcome": "no-action",
                "intent": "don't turn off the lights",
                "reason": "negated action request",
            },
        ),
        (
            ["Do not turn on the lamp"],
            {
                "outcome": "no-action",
                "intent": "do not turn on the lamp",
                "reason": "negated action request",
            },
        ),
        (
            ["Never turn off the ceiling lights"],
            {
                "outcome": "no-action",
                "intent": "never turn off the ceiling lights",
                "reason": "negated action request",
            },
        ),
        (
            ["Don't dim the floor lamp"],
            {
                "outcome": "no-action",
                "intent": "don't dim the floor lamp",
                "reason": "negated action request",
            },
        ),
        (
            ["Do not run good night"],
            {
                "outcome": "no-action",
                "intent": "do not run good night",
                "reason": "negated action request",
            },
        ),
        (
            ["Don't start movie time"],
            {
                "outcome": "no-action",
                "intent": "don't start movie time",
                "reason": "negated action request",
            },
        ),
        (
            ["Never set the thermostat to sixty eight"],
            {
                "outcome": "no-action",
                "intent": "never set the thermostat to sixty eight",
                "reason": "negated action request",
            },
        ),
        (
            ["Don't turn on the lights in here"],
            {
                "outcome": "no-action",
                "intent": "don't turn on the lights in here",
                "reason": "negated action request",
            },
        ),
    ]
    action_specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
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
            ["Turn on the lights but not the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn on the lights but not the floor lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["floor lamp"],
                "state": "on",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn off everything in here except the reading lamp"],
            {
                "outcome": "action",
                "intent": "turn off everything in here except the reading lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["reading lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn off all lights except the lamp"],
            {
                "outcome": "action",
                "intent": "turn off all lights except the lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn on ceiling lights but leave the floor lamp off"],
            {
                "outcome": "action",
                "intent": "turn on ceiling lights but leave the floor lamp off",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "on",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn off lights in here except the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn off lights in here except the floor lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["floor lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn on all lights except the ceiling lights"],
            {
                "outcome": "action",
                "intent": "turn on all lights except the ceiling lights",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["ceiling lights"],
                "state": "on",
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
        (
            ["Turn off every light except the reading lamp"],
            {
                "outcome": "action",
                "intent": "turn off every light except the reading lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["reading lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn on the lights without the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn on the lights without the floor lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["floor lamp"],
                "state": "on",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
    ]
    cases: list[dict[str, Any]] = []
    for index, (turns, plan) in enumerate(no_action_specs, start=1):
        cases.append(
            _safety_case(
                case_id=f"negation-{index:03d}",
                category="negation",
                turns=turns,
                plan=plan,
                outcome="no_action",
            )
        )
    for index, (turns, plan, candidates, resolved) in enumerate(action_specs, start=1):
        cases.append(
            _safety_case(
                case_id=f"negation-{len(no_action_specs) + index:03d}",
                category="negation",
                turns=turns,
                plan=plan,
                candidates=candidates,
                resolved=resolved,
                outcome="valid_action",
                execution_allowed=True,
            )
        )
    return cases


def _author_exclusion_cases() -> list[dict[str, Any]]:
    living_candidates = list(_LIVING_ROOM_ENTITIES)
    living_lights = list(_LIVING_ROOM_LIGHTS)
    clarify_specs: list[tuple[list[str], dict[str, Any], list[str]]] = [
        (
            ["Turn off all lights except the garage light"],
            {
                "outcome": "clarification",
                "intent": "turn off all lights except the garage light",
                "reason": "unknown exclusion target",
            },
            living_lights,
        ),
        (
            ["Turn on everything except the porch light"],
            {
                "outcome": "clarification",
                "intent": "turn on everything except the porch light",
                "reason": "unknown exclusion target",
            },
            living_lights,
        ),
        (
            ["Turn off the lights except"],
            {
                "outcome": "clarification",
                "intent": "turn off the lights except",
                "reason": "incomplete exclusion phrase",
            },
            living_lights,
        ),
        (
            ["Exclude the basement lamp and turn on the rest"],
            {
                "outcome": "clarification",
                "intent": "exclude the basement lamp and turn on the rest",
                "reason": "unknown exclusion target",
            },
            living_lights,
        ),
        (
            ["Turn off all lights except the hallway sconce"],
            {
                "outcome": "clarification",
                "intent": "turn off all lights except the hallway sconce",
                "reason": "unknown exclusion target",
            },
            living_lights,
        ),
        (
            ["Turn on lights except the desk lamp"],
            {
                "outcome": "clarification",
                "intent": "turn on lights except the desk lamp",
                "reason": "unknown exclusion target",
            },
            living_lights,
        ),
        (
            ["Turn off everything except the kitchen pendants"],
            {
                "outcome": "clarification",
                "intent": "turn off everything except the kitchen pendants",
                "reason": "unknown exclusion target",
            },
            living_lights,
        ),
        (
            ["Turn on all except the patio string lights"],
            {
                "outcome": "clarification",
                "intent": "turn on all except the patio string lights",
                "reason": "unknown exclusion target",
            },
            living_lights,
        ),
    ]
    action_specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
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
            ["Turn on all lights except the ceiling lights"],
            {
                "outcome": "action",
                "intent": "turn on all lights except the ceiling lights",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["ceiling lights"],
                "state": "on",
            },
            living_candidates,
            ["light.floor_lamp"],
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
            ["Turn off every light except the reading lamp"],
            {
                "outcome": "action",
                "intent": "turn off every light except the reading lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["reading lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn on the ceiling lights and exclude the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn on the ceiling lights and exclude the floor lamp",
                "domain": "light",
                "targets": ["ceiling lights"],
                "exclude": ["floor lamp"],
                "state": "on",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn off all living room lights except the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn off all living room lights except the floor lamp",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "exclude": ["floor lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn on lights downstairs except the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn on lights downstairs except the floor lamp",
                "domain": "light",
                "scope": {"kind": "floor", "name": "downstairs"},
                "exclude": ["floor lamp"],
                "state": "on",
            },
            list(_ALL_ENTITIES),
            ["light.living_room_ceiling"],
        ),
        (
            ["Turn off the ceiling and floor lamp except the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn off the ceiling and floor lamp except the floor lamp",
                "domain": "light",
                "targets": ["ceiling lights", "floor lamp"],
                "exclude": ["floor lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
    ]
    cases: list[dict[str, Any]] = []
    for index, (turns, plan, candidates) in enumerate(clarify_specs, start=1):
        cases.append(
            _safety_case(
                case_id=f"exclusion-{index:03d}",
                category="exclusion",
                turns=turns,
                plan=plan,
                candidates=candidates,
                outcome="clarification",
            )
        )
    for index, (turns, plan, candidates, resolved) in enumerate(action_specs, start=1):
        cases.append(
            _safety_case(
                case_id=f"exclusion-{len(clarify_specs) + index:03d}",
                category="exclusion",
                turns=turns,
                plan=plan,
                candidates=candidates,
                resolved=resolved,
                outcome="valid_action",
                execution_allowed=True,
            )
        )
    return cases


def _author_unsupported_cases() -> list[dict[str, Any]]:
    living_candidates = list(_LIVING_ROOM_ENTITIES)
    living_lights = list(_LIVING_ROOM_LIGHTS)
    unsupported_specs: list[tuple[list[str], dict[str, Any]]] = [
        (
            ["Play music on spotify"],
            {
                "outcome": "unsupported",
                "intent": "play music on spotify",
                "reason": "media playback is not supported",
            },
        ),
        (
            ["Lock the front door"],
            {
                "outcome": "unsupported",
                "intent": "lock the front door",
                "reason": "lock control is not supported",
            },
        ),
        (
            ["Start the vacuum"],
            {
                "outcome": "unsupported",
                "intent": "start the vacuum",
                "reason": "vacuum control is not supported",
            },
        ),
        (
            ["Open the blinds"],
            {
                "outcome": "unsupported",
                "intent": "open the blinds",
                "reason": "cover control is not supported",
            },
        ),
        (
            ["Arm the security system"],
            {
                "outcome": "unsupported",
                "intent": "arm the security system",
                "reason": "alarm arming is not supported",
            },
        ),
        (
            ["Send a text message"],
            {
                "outcome": "unsupported",
                "intent": "send a text message",
                "reason": "messaging is not supported",
            },
        ),
        (
            ["Record a video on the doorbell"],
            {
                "outcome": "unsupported",
                "intent": "record a video on the doorbell",
                "reason": "camera recording is not supported",
            },
        ),
        (
            ["Order pizza"],
            {
                "outcome": "unsupported",
                "intent": "order pizza",
                "reason": "external ordering is not supported",
            },
        ),
    ]
    action_specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
        (
            ["Turn on the lights"],
            {
                "outcome": "action",
                "intent": "turn on the lights",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "on",
            },
            living_candidates,
            living_lights,
        ),
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
            ["Run good night"],
            {
                "outcome": "action",
                "intent": "run good night",
                "domain": "script",
                "scope": {"kind": "all"},
                "targets": ["good night"],
                "state": "activate",
            },
            list(_ALL_ENTITIES),
            ["script.good_night"],
        ),
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
            ["Turn off the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn off the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "off",
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
        (
            ["Dim the ceiling lights to fifty"],
            {
                "outcome": "action",
                "intent": "dim the ceiling lights to fifty",
                "domain": "light",
                "targets": ["ceiling lights"],
                "value": 50,
            },
            living_candidates,
            ["light.living_room_ceiling"],
        ),
        (
            ["Activate good night"],
            {
                "outcome": "action",
                "intent": "activate good night",
                "domain": "script",
                "scope": {"kind": "all"},
                "targets": ["good night"],
                "state": "activate",
            },
            list(_ALL_ENTITIES),
            ["script.good_night"],
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
    ]
    cases: list[dict[str, Any]] = []
    for index, (turns, plan) in enumerate(unsupported_specs, start=1):
        cases.append(
            _safety_case(
                case_id=f"unsupported-{index:03d}",
                category="unsupported",
                turns=turns,
                plan=plan,
                outcome="unsupported",
            )
        )
    for index, (turns, plan, candidates, resolved) in enumerate(action_specs, start=1):
        cases.append(
            _safety_case(
                case_id=f"unsupported-{len(unsupported_specs) + index:03d}",
                category="unsupported",
                turns=turns,
                plan=plan,
                candidates=candidates,
                resolved=resolved,
                outcome="valid_action",
                execution_allowed=True,
            )
        )
    return cases


def _author_no_action_cases() -> list[dict[str, Any]]:
    living_candidates = list(_LIVING_ROOM_ENTITIES)
    living_lights = list(_LIVING_ROOM_LIGHTS)
    no_action_specs: list[tuple[list[str], dict[str, Any]]] = [
        (
            ["Hello"],
            {
                "outcome": "no-action",
                "intent": "hello",
                "reason": "greeting is not a home control request",
            },
        ),
        (
            ["Thank you"],
            {
                "outcome": "no-action",
                "intent": "thank you",
                "reason": "acknowledgement is not a home control request",
            },
        ),
        (
            ["What's the weather"],
            {
                "outcome": "no-action",
                "intent": "what's the weather",
                "reason": "weather lookup is not supported",
            },
        ),
        (
            ["Tell me a joke"],
            {
                "outcome": "no-action",
                "intent": "tell me a joke",
                "reason": "general conversation is not a home control request",
            },
        ),
        (
            ["Never mind"],
            {
                "outcome": "no-action",
                "intent": "never mind",
                "reason": "user cancelled the request",
            },
        ),
        (
            ["Cancel that"],
            {
                "outcome": "no-action",
                "intent": "cancel that",
                "reason": "user cancelled the request",
            },
        ),
        (
            ["Good morning"],
            {
                "outcome": "no-action",
                "intent": "good morning",
                "reason": "greeting is not a home control request",
            },
        ),
        (
            ["How are you"],
            {
                "outcome": "no-action",
                "intent": "how are you",
                "reason": "general conversation is not a home control request",
            },
        ),
    ]
    action_specs: list[tuple[list[str], dict[str, Any], list[str], list[str]]] = [
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
            ["Set the thermostat to seventy"],
            {
                "outcome": "action",
                "intent": "set the thermostat to seventy",
                "domain": "climate",
                "targets": ["thermostat"],
                "value": 70,
            },
            living_candidates,
            ["climate.downstairs"],
        ),
        (
            ["Run good night"],
            {
                "outcome": "action",
                "intent": "run good night",
                "domain": "script",
                "scope": {"kind": "all"},
                "targets": ["good night"],
                "state": "activate",
            },
            list(_ALL_ENTITIES),
            ["script.good_night"],
        ),
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
            ["Turn on the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn on the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "on",
            },
            living_candidates,
            ["light.floor_lamp"],
        ),
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
            ["Turn off the ceiling lights"],
            {
                "outcome": "action",
                "intent": "turn off the ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "off",
            },
            living_candidates,
            ["light.living_room_ceiling"],
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
    ]
    cases: list[dict[str, Any]] = []
    for index, (turns, plan) in enumerate(no_action_specs, start=1):
        cases.append(
            _safety_case(
                case_id=f"no_action-{index:03d}",
                category="no_action",
                turns=turns,
                plan=plan,
                outcome="no_action",
            )
        )
    for index, (turns, plan, candidates, resolved) in enumerate(action_specs, start=1):
        outcome = plan["outcome"]
        expected_outcome = "valid_query" if outcome == "query" else "valid_action"
        execution_allowed = outcome == "action"
        cases.append(
            _safety_case(
                case_id=f"no_action-{len(no_action_specs) + index:03d}",
                category="no_action",
                turns=turns,
                plan=plan,
                candidates=candidates,
                resolved=resolved,
                outcome=expected_outcome,
                execution_allowed=execution_allowed,
            )
        )
    return cases


def author_safety_cases() -> list[dict[str, Any]]:
    cases = [
        *_author_ambiguity_cases(),
        *_author_pronoun_cases(),
        *_author_negation_cases(),
        *_author_exclusion_cases(),
        *_author_unsupported_cases(),
        *_author_no_action_cases(),
    ]
    if len(cases) != SAFETY_CASE_COUNT:
        msg = f"authored {len(cases)} cases, expected {SAFETY_CASE_COUNT}"
        raise RuntimeError(msg)
    counts = Counter(case["category"] for case in cases)
    if dict(counts) != SAFETY_CATEGORY_COUNTS:
        msg = f"authored category counts {dict(counts)} != {SAFETY_CATEGORY_COUNTS}"
        raise RuntimeError(msg)
    parsed = [EvalCase.model_validate(case) for case in cases]
    validate_safety_corpus(parsed)
    verify_expected_resolutions(parsed)
    return cases


def write_safety_dataset(path: Path | None = None) -> Path:
    target = path or SAFETY_DATASET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    cases = author_safety_cases()
    lines = [json.dumps(case, separators=(",", ":")) for case in cases]
    target.write_text("\n".join(lines) + "\n")
    return target


_NoiseSpec = tuple[list[str], dict[str, Any], list[str], list[str], str, bool]


def _cases_from_noise_specs(category: str, specs: list[_NoiseSpec]) -> list[dict[str, Any]]:
    return [
        _case(
            case_id=f"{category}-{index:03d}",
            category=category,
            turns=turns,
            plan=plan,
            candidates=candidates,
            resolved=resolved,
            outcome=outcome,
            execution_allowed=execution_allowed,
        )
        for index, (
            turns,
            plan,
            candidates,
            resolved,
            outcome,
            execution_allowed,
        ) in enumerate(specs, start=1)
    ]


def _author_casual_cases() -> list[dict[str, Any]]:
    living = list(_LIVING_ROOM_ENTITIES)
    lights = list(_LIVING_ROOM_LIGHTS)
    all_candidates = list(_ALL_ENTITIES)
    specs: list[_NoiseSpec] = [
        (
            ["hey can you turn off the ceiling lights"],
            {
                "outcome": "action",
                "intent": "turn off the ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "off",
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["uh please turn on the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn on the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "on",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["yo switch off the lamp"],
            {
                "outcome": "action",
                "intent": "switch off the lamp",
                "domain": "light",
                "targets": ["lamp"],
                "state": "off",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["could you turn on the reading lamp for me"],
            {
                "outcome": "action",
                "intent": "turn on the reading lamp",
                "domain": "light",
                "targets": ["reading lamp"],
                "state": "on",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["like dim the ceiling lights to fifty percent"],
            {
                "outcome": "action",
                "intent": "dim the ceiling lights to fifty percent",
                "domain": "light",
                "targets": ["ceiling lights"],
                "value": 50,
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["um set the floor lamp brightness to seventy five please"],
            {
                "outcome": "action",
                "intent": "set the floor lamp brightness to seventy five",
                "domain": "light",
                "targets": ["floor lamp"],
                "value": 75,
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["go ahead and toggle the floor lamp"],
            {
                "outcome": "action",
                "intent": "toggle the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "toggle",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["can you turn off the overhead lights"],
            {
                "outcome": "action",
                "intent": "turn off the overhead lights",
                "domain": "light",
                "targets": ["overhead lights"],
                "state": "off",
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["hey turn on living room ceiling"],
            {
                "outcome": "action",
                "intent": "turn on living room ceiling",
                "domain": "light",
                "targets": ["living room ceiling"],
                "state": "on",
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["switch off ceiling lights please"],
            {
                "outcome": "action",
                "intent": "switch off ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "off",
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["hey turn off the lights in here"],
            {
                "outcome": "action",
                "intent": "turn off the lights in here",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "off",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["uh can you turn on the lights in the living room"],
            {
                "outcome": "action",
                "intent": "turn on the lights in the living room",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "state": "on",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["please turn off lounge lights"],
            {
                "outcome": "action",
                "intent": "turn off lounge lights",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "lounge"},
                "state": "off",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["yo turn on family room lights"],
            {
                "outcome": "action",
                "intent": "turn on family room lights",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "family room"},
                "state": "on",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["turn off lights downstairs please"],
            {
                "outcome": "action",
                "intent": "turn off lights downstairs",
                "domain": "light",
                "scope": {"kind": "floor", "name": "downstairs"},
                "state": "off",
            },
            all_candidates,
            lights,
            "valid_action",
            True,
        ),
        (
            ["could you dim the lights in this room to thirty"],
            {
                "outcome": "action",
                "intent": "dim the lights in this room to thirty",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "value": 30,
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["hey activate movie time in the living room"],
            {
                "outcome": "action",
                "intent": "activate movie time in the living room",
                "domain": "scene",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "targets": ["movie time"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["run bedtime in the primary bedroom please"],
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
            "valid_action",
            True,
        ),
        (
            ["uh turn off all the lights in here"],
            {
                "outcome": "action",
                "intent": "turn off all the lights in here",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "off",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["hey turn on both living room lights"],
            {
                "outcome": "action",
                "intent": "turn on both living room lights",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "state": "on",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["please turn off lights in here except the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn off lights in here except the floor lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["floor lamp"],
                "state": "off",
            },
            living,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["yo turn on the ceiling lights and the floor lamp"],
            {
                "outcome": "action",
                "intent": "turn on the ceiling lights and the floor lamp",
                "domain": "light",
                "targets": ["ceiling lights", "floor lamp"],
                "state": "on",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["can you turn off all lights but leave the lamp on"],
            {
                "outcome": "action",
                "intent": "turn off all lights but leave the lamp on",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["lamp"],
                "state": "off",
            },
            living,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["turn on all the lights in the lounge"],
            {
                "outcome": "action",
                "intent": "turn on all the lights in the lounge",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "lounge"},
                "state": "on",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["hey shut off all lights in here except reading lamp"],
            {
                "outcome": "action",
                "intent": "shut off all lights in here except reading lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["reading lamp"],
                "state": "off",
            },
            living,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["uh start movie time"],
            {
                "outcome": "action",
                "intent": "start movie time",
                "domain": "scene",
                "targets": ["movie time"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["hey activate movie mode"],
            {
                "outcome": "action",
                "intent": "activate movie mode",
                "domain": "scene",
                "targets": ["movie mode"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["please run cinema scene"],
            {
                "outcome": "action",
                "intent": "run cinema scene",
                "domain": "scene",
                "targets": ["cinema"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["yo turn on movie time in here"],
            {
                "outcome": "action",
                "intent": "turn on movie time in here",
                "domain": "scene",
                "scope": {"kind": "current_area"},
                "targets": ["movie time"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["could you activate movie time please"],
            {
                "outcome": "action",
                "intent": "activate movie time",
                "domain": "scene",
                "targets": ["movie time"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["hey run good night"],
            {
                "outcome": "action",
                "intent": "run good night",
                "domain": "script",
                "targets": ["good night"],
                "state": "activate",
                "scope": {"kind": "all"},
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
        ),
        (
            ["uh trigger bedtime"],
            {
                "outcome": "action",
                "intent": "trigger bedtime",
                "domain": "script",
                "targets": ["bedtime"],
                "state": "activate",
                "scope": {"kind": "all"},
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
        ),
        (
            ["please run good night upstairs"],
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
            "valid_action",
            True,
        ),
        (
            ["yo start the bedtime routine"],
            {
                "outcome": "action",
                "intent": "start the bedtime routine",
                "domain": "script",
                "targets": ["bedtime"],
                "state": "activate",
                "scope": {"kind": "all"},
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
        ),
        (
            ["can you run bedtime in the master bedroom"],
            {
                "outcome": "action",
                "intent": "run bedtime in the master bedroom",
                "domain": "script",
                "scope": {"kind": "named_area", "name": "master bedroom"},
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
        ),
        (
            ["uh set the thermostat to seventy two"],
            {
                "outcome": "action",
                "intent": "set the thermostat to seventy two",
                "domain": "climate",
                "targets": ["thermostat"],
                "value": 72,
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["hey turn on heat"],
            {
                "outcome": "action",
                "intent": "turn on heat",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "heat",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["please set thermostat to sixty eight"],
            {
                "outcome": "action",
                "intent": "set thermostat to sixty eight",
                "domain": "climate",
                "targets": ["thermostat"],
                "value": 68,
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["yo set the hvac to cool"],
            {
                "outcome": "action",
                "intent": "set the hvac to cool",
                "domain": "climate",
                "targets": ["hvac"],
                "mode": "cool",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["can you turn off the thermostat"],
            {
                "outcome": "action",
                "intent": "turn off the thermostat",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "off",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["hey set heat to seventy four"],
            {
                "outcome": "action",
                "intent": "set heat to seventy four",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "heat",
                "value": 74,
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["um switch thermostat to auto"],
            {
                "outcome": "action",
                "intent": "switch thermostat to auto",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "auto",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["please set the temperature to seventy one"],
            {
                "outcome": "action",
                "intent": "set the temperature to seventy one",
                "domain": "climate",
                "targets": ["thermostat"],
                "value": 71,
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["hey is the door closed"],
            {
                "outcome": "query",
                "intent": "is the door closed",
                "domain": "binary_sensor",
                "targets": ["door"],
            },
            living,
            ["binary_sensor.front_door"],
            "valid_query",
            False,
        ),
        (
            ["uh are any lights on in here"],
            {
                "outcome": "query",
                "intent": "are any lights on in here",
                "domain": "light",
                "scope": {"kind": "current_area"},
            },
            living,
            lights,
            "valid_query",
            False,
        ),
        (
            ["what is the thermostat set to please"],
            {
                "outcome": "query",
                "intent": "what is the thermostat set to",
                "domain": "climate",
                "targets": ["thermostat"],
                "attribute": "temperature",
            },
            living,
            ["climate.downstairs"],
            "valid_query",
            False,
        ),
        (
            ["yo is the front door open"],
            {
                "outcome": "query",
                "intent": "is the front door open",
                "domain": "binary_sensor",
                "targets": ["front door"],
            },
            living,
            ["binary_sensor.front_door"],
            "valid_query",
            False,
        ),
        (
            ["can you check if any lights are on"],
            {
                "outcome": "query",
                "intent": "check if any lights are on",
                "domain": "light",
                "scope": {"kind": "current_area"},
            },
            living,
            lights,
            "valid_query",
            False,
        ),
        (
            ["hey is the floor lamp on"],
            {
                "outcome": "query",
                "intent": "is the floor lamp on",
                "domain": "light",
                "targets": ["floor lamp"],
            },
            living,
            ["light.floor_lamp"],
            "valid_query",
            False,
        ),
        (
            ["uh what is the hvac mode"],
            {
                "outcome": "query",
                "intent": "what is the hvac mode",
                "domain": "climate",
                "targets": ["hvac"],
                "attribute": "hvac_mode",
            },
            living,
            ["climate.downstairs"],
            "valid_query",
            False,
        ),
    ]
    return _cases_from_noise_specs("casual", specs)


def _author_alias_cases() -> list[dict[str, Any]]:
    living = list(_LIVING_ROOM_ENTITIES)
    lights = list(_LIVING_ROOM_LIGHTS)
    all_candidates = list(_ALL_ENTITIES)
    specs: list[_NoiseSpec] = [
        (
            ["Turn off overhead lights"],
            {
                "outcome": "action",
                "intent": "turn off overhead lights",
                "domain": "light",
                "targets": ["overhead lights"],
                "state": "off",
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["Turn on reading lamp"],
            {
                "outcome": "action",
                "intent": "turn on reading lamp",
                "domain": "light",
                "targets": ["reading lamp"],
                "state": "on",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["Switch off lamp"],
            {
                "outcome": "action",
                "intent": "switch off lamp",
                "domain": "light",
                "targets": ["lamp"],
                "state": "off",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["Dim ceiling lights to forty"],
            {
                "outcome": "action",
                "intent": "dim ceiling lights to forty",
                "domain": "light",
                "targets": ["ceiling lights"],
                "value": 40,
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["Set hvac to cool"],
            {
                "outcome": "action",
                "intent": "set hvac to cool",
                "domain": "climate",
                "targets": ["hvac"],
                "mode": "cool",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["Turn off thermostat"],
            {
                "outcome": "action",
                "intent": "turn off thermostat",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "off",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["Set thermostat to seventy"],
            {
                "outcome": "action",
                "intent": "set thermostat to seventy",
                "domain": "climate",
                "targets": ["thermostat"],
                "value": 70,
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["Is door closed"],
            {
                "outcome": "query",
                "intent": "is door closed",
                "domain": "binary_sensor",
                "targets": ["door"],
            },
            living,
            ["binary_sensor.front_door"],
            "valid_query",
            False,
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
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["Run cinema"],
            {
                "outcome": "action",
                "intent": "run cinema",
                "domain": "scene",
                "targets": ["cinema"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["Trigger bedtime"],
            {
                "outcome": "action",
                "intent": "trigger bedtime",
                "domain": "script",
                "targets": ["bedtime"],
                "state": "activate",
                "scope": {"kind": "all"},
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
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
            living,
            lights,
            "valid_action",
            True,
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
            living,
            lights,
            "valid_action",
            True,
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
            lights,
            "valid_action",
            True,
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
            "valid_action",
            True,
        ),
        (
            ["Run bedtime on the upper floor"],
            {
                "outcome": "action",
                "intent": "run bedtime on the upper floor",
                "domain": "script",
                "scope": {"kind": "floor", "name": "upper floor"},
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
        ),
        (
            ["Run good night in master bedroom"],
            {
                "outcome": "action",
                "intent": "run good night in master bedroom",
                "domain": "script",
                "scope": {"kind": "named_area", "name": "master bedroom"},
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
        ),
        (
            ["Turn on lights on first floor"],
            {
                "outcome": "action",
                "intent": "turn on lights on first floor",
                "domain": "light",
                "scope": {"kind": "floor", "name": "first floor"},
                "state": "on",
            },
            all_candidates,
            lights,
            "valid_action",
            True,
        ),
        (
            ["Run bedtime on second floor"],
            {
                "outcome": "action",
                "intent": "run bedtime on second floor",
                "domain": "script",
                "scope": {"kind": "floor", "name": "second floor"},
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
        ),
        (
            ["Brighten reading lamp to sixty"],
            {
                "outcome": "action",
                "intent": "brighten reading lamp to sixty",
                "domain": "light",
                "targets": ["reading lamp"],
                "value": 60,
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["Toggle lamp"],
            {
                "outcome": "action",
                "intent": "toggle lamp",
                "domain": "light",
                "targets": ["lamp"],
                "state": "toggle",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["Set downstairs thermostat to sixty nine"],
            {
                "outcome": "action",
                "intent": "set downstairs thermostat to sixty nine",
                "domain": "climate",
                "targets": ["downstairs thermostat"],
                "value": 69,
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["Switch hvac to auto mode"],
            {
                "outcome": "action",
                "intent": "switch hvac to auto mode",
                "domain": "climate",
                "targets": ["hvac"],
                "mode": "auto",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
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
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["Launch cinema scene"],
            {
                "outcome": "action",
                "intent": "launch cinema scene",
                "domain": "scene",
                "targets": ["cinema"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["Begin movie mode in lounge"],
            {
                "outcome": "action",
                "intent": "begin movie mode in lounge",
                "domain": "scene",
                "scope": {"kind": "named_area", "name": "lounge"},
                "targets": ["movie mode"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["Execute bedtime script"],
            {
                "outcome": "action",
                "intent": "execute bedtime script",
                "domain": "script",
                "targets": ["bedtime"],
                "state": "activate",
                "scope": {"kind": "all"},
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
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
            "valid_action",
            True,
        ),
        (
            ["Activate good night"],
            {
                "outcome": "action",
                "intent": "activate good night",
                "domain": "script",
                "targets": ["good night"],
                "state": "activate",
                "scope": {"kind": "all"},
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
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
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["Turn off every light in lounge"],
            {
                "outcome": "action",
                "intent": "turn off every light in lounge",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "lounge"},
                "state": "off",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["Dim all lights in family room to twenty"],
            {
                "outcome": "action",
                "intent": "dim all lights in family room to twenty",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "family room"},
                "value": 20,
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["Turn on lights in lounge"],
            {
                "outcome": "action",
                "intent": "turn on lights in lounge",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "lounge"},
                "state": "on",
            },
            living,
            lights,
            "valid_action",
            True,
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
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["What is hvac mode"],
            {
                "outcome": "query",
                "intent": "what is hvac mode",
                "domain": "climate",
                "targets": ["hvac"],
                "attribute": "hvac_mode",
            },
            living,
            ["climate.downstairs"],
            "valid_query",
            False,
        ),
        (
            ["Is ceiling light on"],
            {
                "outcome": "query",
                "intent": "is ceiling light on",
                "domain": "light",
                "targets": ["ceiling lights"],
            },
            living,
            ["light.living_room_ceiling"],
            "valid_query",
            False,
        ),
        (
            ["Are any lights on downstairs"],
            {
                "outcome": "query",
                "intent": "are any lights on downstairs",
                "domain": "light",
                "scope": {"kind": "floor", "name": "downstairs"},
            },
            all_candidates,
            lights,
            "valid_query",
            False,
        ),
        (
            ["What temperature is thermostat reading"],
            {
                "outcome": "query",
                "intent": "what temperature is thermostat reading",
                "domain": "climate",
                "targets": ["thermostat"],
                "attribute": "current_temperature",
            },
            living,
            ["climate.downstairs"],
            "valid_query",
            False,
        ),
        (
            ["Is door open or closed"],
            {
                "outcome": "query",
                "intent": "is door open or closed",
                "domain": "binary_sensor",
                "targets": ["door"],
            },
            living,
            ["binary_sensor.front_door"],
            "valid_query",
            False,
        ),
        (
            ["Turn off lights on ground floor"],
            {
                "outcome": "action",
                "intent": "turn off lights on ground floor",
                "domain": "light",
                "scope": {"kind": "floor", "name": "ground floor"},
                "state": "off",
            },
            all_candidates,
            lights,
            "valid_action",
            True,
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
            "valid_action",
            True,
        ),
        (
            ["Run bedtime in master bedroom"],
            {
                "outcome": "action",
                "intent": "run bedtime in master bedroom",
                "domain": "script",
                "scope": {"kind": "named_area", "name": "master bedroom"},
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
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
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
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
            living,
            lights,
            "valid_action",
            True,
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
            lights,
            "valid_action",
            True,
        ),
        (
            ["Run movie time in family room"],
            {
                "outcome": "action",
                "intent": "run movie time in family room",
                "domain": "scene",
                "scope": {"kind": "named_area", "name": "family room"},
                "targets": ["movie time"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["Start good night in primary bedroom"],
            {
                "outcome": "action",
                "intent": "start good night in primary bedroom",
                "domain": "script",
                "scope": {"kind": "named_area", "name": "Primary Bedroom"},
                "targets": ["good night"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
        ),
        (
            ["Turn on cooling via hvac"],
            {
                "outcome": "action",
                "intent": "turn on cooling via hvac",
                "domain": "climate",
                "targets": ["hvac"],
                "mode": "cool",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["Check if door is closed"],
            {
                "outcome": "query",
                "intent": "check if door is closed",
                "domain": "binary_sensor",
                "targets": ["door"],
            },
            living,
            ["binary_sensor.front_door"],
            "valid_query",
            False,
        ),
        (
            ["Are all lights off in lounge"],
            {
                "outcome": "query",
                "intent": "are all lights off in lounge",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "lounge"},
            },
            living,
            lights,
            "valid_query",
            False,
        ),
    ]
    return _cases_from_noise_specs("alias", specs)


def _author_asr_cases() -> list[dict[str, Any]]:
    living = list(_LIVING_ROOM_ENTITIES)
    lights = list(_LIVING_ROOM_LIGHTS)
    all_candidates = list(_ALL_ENTITIES)
    specs: list[_NoiseSpec] = [
        (
            ["turn off the sealing lights"],
            {
                "outcome": "action",
                "intent": "turn off the ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "off",
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["turn on the floor lam"],
            {
                "outcome": "action",
                "intent": "turn on the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "on",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["switch of the lamp"],
            {
                "outcome": "action",
                "intent": "switch off the lamp",
                "domain": "light",
                "targets": ["lamp"],
                "state": "off",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["dim the ceiling lites to fifty"],
            {
                "outcome": "action",
                "intent": "dim the ceiling lights to fifty",
                "domain": "light",
                "targets": ["ceiling lights"],
                "value": 50,
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["set floor lamp brightnes to seventy five"],
            {
                "outcome": "action",
                "intent": "set floor lamp brightness to seventy five",
                "domain": "light",
                "targets": ["floor lamp"],
                "value": 75,
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["toggl the floor lamp"],
            {
                "outcome": "action",
                "intent": "toggle the floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "toggle",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["turn of the overhead lights"],
            {
                "outcome": "action",
                "intent": "turn off the overhead lights",
                "domain": "light",
                "targets": ["overhead lights"],
                "state": "off",
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["turn on livin room ceiling"],
            {
                "outcome": "action",
                "intent": "turn on living room ceiling",
                "domain": "light",
                "targets": ["living room ceiling"],
                "state": "on",
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["switch off ceiling lites"],
            {
                "outcome": "action",
                "intent": "switch off ceiling lights",
                "domain": "light",
                "targets": ["ceiling lights"],
                "state": "off",
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["turn of the lights in hear"],
            {
                "outcome": "action",
                "intent": "turn off the lights in here",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "off",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["turn on lights in livin room"],
            {
                "outcome": "action",
                "intent": "turn on lights in living room",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "state": "on",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["turn off lounge lites"],
            {
                "outcome": "action",
                "intent": "turn off lounge lights",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "lounge"},
                "state": "off",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["turn on famly room lights"],
            {
                "outcome": "action",
                "intent": "turn on family room lights",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "family room"},
                "state": "on",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["turn off lights down stairs"],
            {
                "outcome": "action",
                "intent": "turn off lights downstairs",
                "domain": "light",
                "scope": {"kind": "floor", "name": "downstairs"},
                "state": "off",
            },
            all_candidates,
            lights,
            "valid_action",
            True,
        ),
        (
            ["dim lights in this room to therty"],
            {
                "outcome": "action",
                "intent": "dim lights in this room to thirty",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "value": 30,
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["activate movie tiem in living room"],
            {
                "outcome": "action",
                "intent": "activate movie time in living room",
                "domain": "scene",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "targets": ["movie time"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["run bed time in primary bedroom"],
            {
                "outcome": "action",
                "intent": "run bedtime in primary bedroom",
                "domain": "script",
                "scope": {"kind": "named_area", "name": "Primary Bedroom"},
                "targets": ["bedtime"],
                "state": "activate",
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
        ),
        (
            ["turn off all lites in here"],
            {
                "outcome": "action",
                "intent": "turn off all lights in here",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "state": "off",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["turn on both livin room lights"],
            {
                "outcome": "action",
                "intent": "turn on both living room lights",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "Living Room"},
                "state": "on",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["turn off lights here except floor lam"],
            {
                "outcome": "action",
                "intent": "turn off lights here except floor lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["floor lamp"],
                "state": "off",
            },
            living,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["turn on ceiling lites and floor lam"],
            {
                "outcome": "action",
                "intent": "turn on ceiling lights and floor lamp",
                "domain": "light",
                "targets": ["ceiling lights", "floor lamp"],
                "state": "on",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["turn off all lights but leave lam on"],
            {
                "outcome": "action",
                "intent": "turn off all lights but leave lamp on",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["lamp"],
                "state": "off",
            },
            living,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["turn on all lites in lounge"],
            {
                "outcome": "action",
                "intent": "turn on all lights in lounge",
                "domain": "light",
                "scope": {"kind": "named_area", "name": "lounge"},
                "state": "on",
            },
            living,
            lights,
            "valid_action",
            True,
        ),
        (
            ["shut off all lights except readin lam"],
            {
                "outcome": "action",
                "intent": "shut off all lights except reading lamp",
                "domain": "light",
                "scope": {"kind": "current_area"},
                "exclude": ["reading lamp"],
                "state": "off",
            },
            living,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
        (
            ["start movie tiem"],
            {
                "outcome": "action",
                "intent": "start movie time",
                "domain": "scene",
                "targets": ["movie time"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["activate movie mod"],
            {
                "outcome": "action",
                "intent": "activate movie mode",
                "domain": "scene",
                "targets": ["movie mode"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["run sinema scene"],
            {
                "outcome": "action",
                "intent": "run cinema scene",
                "domain": "scene",
                "targets": ["cinema"],
                "state": "activate",
            },
            living,
            ["scene.movie_time"],
            "valid_action",
            True,
        ),
        (
            ["run good nite"],
            {
                "outcome": "action",
                "intent": "run good night",
                "domain": "script",
                "targets": ["good night"],
                "state": "activate",
                "scope": {"kind": "all"},
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
        ),
        (
            ["trigger bed time"],
            {
                "outcome": "action",
                "intent": "trigger bedtime",
                "domain": "script",
                "targets": ["bedtime"],
                "state": "activate",
                "scope": {"kind": "all"},
            },
            all_candidates,
            ["script.good_night"],
            "valid_action",
            True,
        ),
        (
            ["run good nite upstairs"],
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
            "valid_action",
            True,
        ),
        (
            ["set thermo stat to seventy two"],
            {
                "outcome": "action",
                "intent": "set thermostat to seventy two",
                "domain": "climate",
                "targets": ["thermostat"],
                "value": 72,
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["turn on heet"],
            {
                "outcome": "action",
                "intent": "turn on heat",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "heat",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["set thermostate to sixty eight"],
            {
                "outcome": "action",
                "intent": "set thermostat to sixty eight",
                "domain": "climate",
                "targets": ["thermostat"],
                "value": 68,
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["set the h vac to cool"],
            {
                "outcome": "action",
                "intent": "set hvac to cool",
                "domain": "climate",
                "targets": ["hvac"],
                "mode": "cool",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["turn of the thermostate"],
            {
                "outcome": "action",
                "intent": "turn off thermostat",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "off",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["set heet to seventy for"],
            {
                "outcome": "action",
                "intent": "set heat to seventy four",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "heat",
                "value": 74,
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["switch thermostate to auto"],
            {
                "outcome": "action",
                "intent": "switch thermostat to auto",
                "domain": "climate",
                "targets": ["thermostat"],
                "mode": "auto",
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["set temprature to seventy one"],
            {
                "outcome": "action",
                "intent": "set temperature to seventy one",
                "domain": "climate",
                "targets": ["thermostat"],
                "value": 71,
            },
            living,
            ["climate.downstairs"],
            "valid_action",
            True,
        ),
        (
            ["is the dor closed"],
            {
                "outcome": "query",
                "intent": "is the door closed",
                "domain": "binary_sensor",
                "targets": ["door"],
            },
            living,
            ["binary_sensor.front_door"],
            "valid_query",
            False,
        ),
        (
            ["are any lites on in hear"],
            {
                "outcome": "query",
                "intent": "are any lights on in here",
                "domain": "light",
                "scope": {"kind": "current_area"},
            },
            living,
            lights,
            "valid_query",
            False,
        ),
        (
            ["what is thermostate set to"],
            {
                "outcome": "query",
                "intent": "what is thermostat set to",
                "domain": "climate",
                "targets": ["thermostat"],
                "attribute": "temperature",
            },
            living,
            ["climate.downstairs"],
            "valid_query",
            False,
        ),
        (
            ["is front dor open"],
            {
                "outcome": "query",
                "intent": "is front door open",
                "domain": "binary_sensor",
                "targets": ["front door"],
            },
            living,
            ["binary_sensor.front_door"],
            "valid_query",
            False,
        ),
        (
            ["check if any lites are on"],
            {
                "outcome": "query",
                "intent": "check if any lights are on",
                "domain": "light",
                "scope": {"kind": "current_area"},
            },
            living,
            lights,
            "valid_query",
            False,
        ),
        (
            ["is floor lam on"],
            {
                "outcome": "query",
                "intent": "is floor lamp on",
                "domain": "light",
                "targets": ["floor lamp"],
            },
            living,
            ["light.floor_lamp"],
            "valid_query",
            False,
        ),
        (
            ["what is h vac mode"],
            {
                "outcome": "query",
                "intent": "what is hvac mode",
                "domain": "climate",
                "targets": ["hvac"],
                "attribute": "hvac_mode",
            },
            living,
            ["climate.downstairs"],
            "valid_query",
            False,
        ),
        (
            ["turn on readin lam"],
            {
                "outcome": "action",
                "intent": "turn on reading lamp",
                "domain": "light",
                "targets": ["reading lamp"],
                "state": "on",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["brighten floor lam to forty"],
            {
                "outcome": "action",
                "intent": "brighten floor lamp to forty",
                "domain": "light",
                "targets": ["floor lamp"],
                "value": 40,
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["turn the lam on"],
            {
                "outcome": "action",
                "intent": "turn the lamp on",
                "domain": "light",
                "targets": ["lamp"],
                "state": "on",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["shut of floor lam"],
            {
                "outcome": "action",
                "intent": "shut off floor lamp",
                "domain": "light",
                "targets": ["floor lamp"],
                "state": "off",
            },
            lights,
            ["light.floor_lamp"],
            "valid_action",
            True,
        ),
        (
            ["set ceiling lites to twenty five percent"],
            {
                "outcome": "action",
                "intent": "set ceiling lights to twenty five percent",
                "domain": "light",
                "targets": ["ceiling lights"],
                "value": 25,
            },
            lights,
            ["light.living_room_ceiling"],
            "valid_action",
            True,
        ),
    ]
    return _cases_from_noise_specs("asr", specs)


def author_language_noise_cases() -> list[dict[str, Any]]:
    cases = [
        *_author_casual_cases(),
        *_author_alias_cases(),
        *_author_asr_cases(),
    ]
    if len(cases) != LANGUAGE_NOISE_CASE_COUNT:
        msg = f"authored {len(cases)} cases, expected {LANGUAGE_NOISE_CASE_COUNT}"
        raise RuntimeError(msg)
    counts = Counter(case["category"] for case in cases)
    if dict(counts) != LANGUAGE_NOISE_CATEGORY_COUNTS:
        msg = (
            f"authored category counts {dict(counts)} != "
            f"{LANGUAGE_NOISE_CATEGORY_COUNTS}"
        )
        raise RuntimeError(msg)
    parsed = [EvalCase.model_validate(case) for case in cases]
    validate_language_noise_corpus(parsed)
    verify_expected_resolutions(parsed)
    return cases


def write_language_noise_dataset(path: Path | None = None) -> Path:
    target = path or LANGUAGE_NOISE_DATASET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    cases = author_language_noise_cases()
    lines = [json.dumps(case, separators=(",", ":")) for case in cases]
    target.write_text("\n".join(lines) + "\n")
    return target


if __name__ == "__main__":
    written = write_core_dataset()
    print(f"wrote {CORE_CASE_COUNT} cases to {written}")
    safety_written = write_safety_dataset()
    print(f"wrote {SAFETY_CASE_COUNT} cases to {safety_written}")
    noise_written = write_language_noise_dataset()
    print(f"wrote {LANGUAGE_NOISE_CASE_COUNT} cases to {noise_written}")
