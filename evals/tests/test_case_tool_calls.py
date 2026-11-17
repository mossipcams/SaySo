from __future__ import annotations

from typing import Any

import pytest

from evals.cases import Case, load_all_cases, load_home
from evals.runner import production_catalog


def _fold(values: Any) -> set[str]:
    return {str(value).casefold() for value in values if value is not None}


def _entity_index(home: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for key in ("entities", "context_entities"):
        for entity in home.get(key) or []:
            names = [entity.get("name"), *(entity.get("aliases") or [])]
            for name in names:
                if name is not None:
                    index.setdefault(str(name).casefold(), []).append(entity)
    return index


def _area_names(home: dict[str, Any]) -> set[str]:
    names = set()
    for area, aliases in (home.get("area_aliases") or {}).items():
        names.update(_fold([area, *aliases]))
    return names


def _floors(home: dict[str, Any]) -> set[str]:
    return _fold(entity.get("floor") for entity in home.get("entities") or [])


def _type_ok(value: Any, spec: dict[str, Any]) -> bool:
    if "enum" in spec and value not in spec["enum"]:
        return False
    any_of = spec.get("anyOf")
    if any_of:
        return any(_type_ok(value, sub) for sub in any_of)
    kind = spec.get("type")
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "array":
        if not isinstance(value, list):
            return False
        items = spec.get("items") or {}
        return all(_type_ok(item, items) for item in value) if items else True
    return True


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        return [value]
    return list(value)


def _is_timer_tool(name: str) -> bool:
    return name.startswith("intent__") and "Timer" in name


def _call_problems(
    case_id: str,
    call: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    index: dict[str, list[dict[str, Any]]],
    areas: set[str],
    floors: set[str],
) -> list[str]:
    problems: list[str] = []
    name = call.get("name")
    arguments = call.get("arguments") or {}
    parameters = catalog.get(name) if isinstance(name, str) else None
    if parameters is None:
        return [f"{case_id} {name}: tool not in household catalog"]
    properties = parameters.get("properties") or {}
    for key, value in arguments.items():
        prop = properties.get(key)
        if prop is None:
            problems.append(f"{case_id} {name}: argument {key!r} not in schema")
        elif not _type_ok(value, prop):
            problems.append(
                f"{case_id} {name}: argument {key!r} value {value!r} "
                "does not match schema type or enum"
            )
    area = arguments.get("area")
    if area is not None and str(area).casefold() not in areas:
        problems.append(f"{case_id} {name}: area {area!r} not in household")
    floor = arguments.get("floor")
    if floor is not None and str(floor).casefold() not in floors:
        problems.append(f"{case_id} {name}: floor {floor!r} not in household")
    entity_name = arguments.get("name")
    if entity_name is None or _is_timer_tool(name):
        return problems
    resolved = index.get(str(entity_name).casefold())
    if resolved is None:
        problems.append(
            f"{case_id} {name}: name {entity_name!r} does not resolve to an "
            "entity or script in the household"
        )
        return problems
    domains = _fold(entity.get("domain") for entity in resolved)
    device_classes = _fold(entity.get("device_class") for entity in resolved)
    for value in _values(arguments.get("domain")):
        if str(value).casefold() not in domains:
            problems.append(
                f"{case_id} {name}: domain {value!r} does not match "
                f"resolved entity domain ({', '.join(sorted(domains))})"
            )
    for value in _values(arguments.get("device_class")):
        if str(value).casefold() not in device_classes:
            problems.append(
                f"{case_id} {name}: device_class {value!r} does not match "
                f"resolved entity device_class ({', '.join(sorted(device_classes))})"
            )
    return problems


@pytest.mark.parametrize("case", load_all_cases(), ids=lambda case: case.id)
def test_expected_tool_calls_exist_and_are_accurate(case: Case) -> None:
    home = load_home(case.household)
    catalog = {
        tool["function"]["name"]: tool["function"].get("parameters") or {}
        for tool in production_catalog(
            home, removed_tools=(case.unavailable or {}).get("removed_tools") or []
        )
    }
    index = _entity_index(home)
    areas = _area_names(home)
    floors = _floors(home)
    problems: list[str] = []
    for call in case.expected["calls"] or []:
        problems.extend(_call_problems(case.id, call, catalog, index, areas, floors))
    assert not problems, "\n".join(problems)
