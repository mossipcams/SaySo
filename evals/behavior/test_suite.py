"""Load and validate the independent behavioral voice-command case suite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schemas" / "sayso-tool-schema-v1.json"
DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "cases.yaml"


@dataclass(frozen=True, slots=True)
class BehaviorCase:
    """One behavioral voice-command evaluation case."""

    id: str
    category: str
    scenario: str
    description: str
    expect: dict[str, Any]
    checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BehaviorCaseSet:
    """Versioned behavioral evaluation cases independent of evals/cases/."""

    version: int
    cases: tuple[BehaviorCase, ...]


def _schema_tool_parameters() -> dict[str, set[str]]:
    payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    parameters_by_tool: dict[str, set[str]] = {}
    for entry in payload.get("tools", []):
        function = entry.get("function", {})
        name = function.get("name")
        if not isinstance(name, str):
            continue
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        if isinstance(properties, dict):
            parameters_by_tool[name] = set(properties)
        else:
            parameters_by_tool[name] = set()
    return parameters_by_tool


def _parse_checks(raw_checks: Any) -> tuple[str, ...]:
    if not isinstance(raw_checks, list):
        return ()
    return tuple(str(item) for item in raw_checks)


def _parse_case(raw: dict[str, Any]) -> BehaviorCase:
    return BehaviorCase(
        id=str(raw["id"]),
        category=str(raw["category"]),
        scenario=str(raw["scenario"]),
        description=str(raw.get("description", "")),
        expect=dict(raw.get("expect", {})),
        checks=_parse_checks(raw.get("checks")),
    )


def load_behavior_cases(path: str | Path | None = None) -> BehaviorCaseSet:
    """Load behavioral cases from YAML without touching evals/cases/."""
    source = Path(path) if path is not None else DEFAULT_CASES_PATH
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("behavior cases root must be a mapping")
    version = int(payload["version"])
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("cases must be a list")
    cases = tuple(_parse_case(entry) for entry in raw_cases if isinstance(entry, dict))
    return BehaviorCaseSet(version=version, cases=cases)


def validate_behavior_cases(case_set: BehaviorCaseSet) -> None:
    """Fail fast when the suite shape or schema grounding is invalid."""
    if case_set.version != 1:
        raise ValueError(f"unexpected behavior suite version: {case_set.version}")
    if len(case_set.cases) != 300:
        raise ValueError(f"expected 300 behavior cases, found {len(case_set.cases)}")

    ids = [case.id for case in case_set.cases]
    if len(ids) != len(set(ids)):
        raise ValueError("behavior case ids must be unique")

    utterances = [case.scenario for case in case_set.cases]
    if len(utterances) != len(set(utterances)):
        raise ValueError("behavior case utterances must be unique")

    parameters_by_tool = _schema_tool_parameters()

    for case in case_set.cases:
        if not case.id or not case.category or not case.scenario:
            raise ValueError(f"case {case.id or '<missing id>'} missing id/category/scenario")
        if not case.checks:
            raise ValueError(f"case {case.id} must include checks")

        tool_calls = case.expect.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            raise ValueError(f"case {case.id} must include expect.tool_calls")

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise ValueError(f"case {case.id} has invalid tool call entry")
            tool_name = tool_call.get("name")
            if tool_name not in parameters_by_tool:
                raise ValueError(f"case {case.id} references unknown tool {tool_name}")
            arguments = tool_call.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError(f"case {case.id} tool arguments must be a mapping")
            allowed = parameters_by_tool[tool_name]
            unknown = set(arguments) - allowed
            if unknown:
                raise ValueError(
                    f"case {case.id} tool {tool_name} has unknown arguments: {sorted(unknown)}"
                )

        _validate_pause_media_case(case)


def _validate_pause_media_case(case: BehaviorCase) -> None:
    """Pause-media utterances use the same HassTurnOff tv contract as pause TV."""
    utterance = case.scenario.lower()
    if "pause" not in utterance or "media" not in utterance:
        return
    tool_calls = case.expect.get("tool_calls", [])
    if len(tool_calls) != 1:
        raise ValueError(f"case {case.id} pause-media must have exactly one tool call")
    tool_call = tool_calls[0]
    if tool_call.get("name") != "HassTurnOff":
        raise ValueError(f"case {case.id} pause-media must use HassTurnOff")
    arguments = tool_call.get("arguments", {})
    if arguments.get("device_class") != ["tv"]:
        raise ValueError(
            f"case {case.id} pause-media must use device_class ['tv'], not domain media_player"
        )
    if "domain" in arguments and arguments["domain"] == ["media_player"]:
        raise ValueError(f"case {case.id} pause-media must not target domain media_player")
