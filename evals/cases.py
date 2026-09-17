"""Case loading, suite manifests, and schema validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVALS_ROOT = Path(__file__).resolve().parent
CASES_DIR = EVALS_ROOT / "cases"
HOMES_DIR = EVALS_ROOT / "homes"
SUITES_DIR = EVALS_ROOT / "suites"
GATES_PATH = EVALS_ROOT / "config" / "gates.yaml"
ARCHIVE_DIR = EVALS_ROOT / "archive"

_CASE_FILES = ("realistic_v3.jsonl", "regressions.jsonl")
REQUIRED_CASE_KEYS = frozenset(
    {"id", "category", "tags", "household", "utterance", "unavailable", "expected", "provenance"}
)
REQUIRED_EXPECTED_KEYS = frozenset(
    {"response_type", "calls", "clarification", "forbidden_entities", "spoken_aliases"}
)
RESPONSE_TYPES = frozenset({"action", "status", "clarification", "refusal"})


class CaseError(ValueError):
    """A case file, suite, or home fixture is invalid."""


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    category: str
    tags: tuple[str, ...]
    household: str
    utterance: str
    unavailable: dict[str, Any] | None
    expected: dict[str, Any]
    provenance: dict[str, Any]
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "tags": list(self.tags),
            "household": self.household,
            "utterance": self.utterance,
            "unavailable": self.unavailable,
            "expected": self.expected,
            "provenance": self.provenance,
        }


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as err:  # pragma: no cover - HA and training hosts ship PyYAML
        raise CaseError(f"PyYAML is required to load {path}") from err
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _require_keys(payload: dict[str, Any], required: frozenset[str], label: str) -> None:
    missing = required - payload.keys()
    if missing:
        raise CaseError(f"{label} missing keys: {sorted(missing)}")


def _parse_case(payload: dict[str, Any], *, source: str, line: int) -> Case:
    label = f"{source}:{line}"
    _require_keys(payload, REQUIRED_CASE_KEYS, label)
    expected = payload["expected"]
    if not isinstance(expected, dict):
        raise CaseError(f"{label} expected must be an object")
    _require_keys(expected, REQUIRED_EXPECTED_KEYS, label)
    if expected["response_type"] not in RESPONSE_TYPES:
        raise CaseError(f"{label} unknown response_type {expected['response_type']!r}")
    if not isinstance(expected["calls"], list):
        raise CaseError(f"{label} expected.calls must be a list")
    tags = payload["tags"]
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise CaseError(f"{label} tags must be a list of strings")
    if not isinstance(payload["id"], str) or not payload["id"]:
        raise CaseError(f"{label} id must be a non-empty string")
    if not isinstance(payload["utterance"], str) or not payload["utterance"]:
        raise CaseError(f"{label} utterance must be a non-empty string")
    if not isinstance(payload["household"], str) or not payload["household"]:
        raise CaseError(f"{label} household must be a non-empty string")
    provenance = payload["provenance"]
    if not isinstance(provenance, dict) or "source" not in provenance:
        raise CaseError(f"{label} provenance.source is required")
    return Case(
        id=payload["id"],
        category=str(payload["category"]),
        tags=tuple(tags),
        household=payload["household"],
        utterance=payload["utterance"],
        unavailable=payload["unavailable"],
        expected=expected,
        provenance=provenance,
        source=source,
    )


def _read_jsonl(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise CaseError(f"{path.name}:{line_number} must be a JSON object")
        cases.append(_parse_case(payload, source=path.name, line=line_number))
    return cases


@lru_cache(maxsize=1)
def load_all_cases() -> tuple[Case, ...]:
    """Every active case. Archive/ is never scanned."""
    cases: list[Case] = []
    seen: dict[str, str] = {}
    for name in _CASE_FILES:
        path = CASES_DIR / name
        if not path.is_file():
            raise CaseError(f"missing case file {path}")
        for case in _read_jsonl(path):
            if case.id in seen:
                raise CaseError(f"duplicate case id {case.id!r} in {seen[case.id]} and {name}")
            seen[case.id] = name
            cases.append(case)
    return tuple(cases)


def case_index() -> dict[str, Case]:
    return {case.id: case for case in load_all_cases()}


def load_home(home_id: str) -> dict[str, Any]:
    path = HOMES_DIR / f"{home_id}.json"
    if not path.is_file():
        raise CaseError(f"missing home fixture {path}")
    home = json.loads(path.read_text(encoding="utf-8"))
    if home.get("home_id") != home_id:
        raise CaseError(f"{path.name} home_id {home.get('home_id')!r} != {home_id!r}")
    home.setdefault("sayso_entity_area", home.get("satellite_area"))
    home.setdefault("area_aliases", home.get("areas") or {})
    return home


def load_all_homes() -> dict[str, dict[str, Any]]:
    homes = {}
    for path in sorted(HOMES_DIR.glob("*.json")):
        home = json.loads(path.read_text(encoding="utf-8"))
        homes[path.stem] = home
    return homes


def load_suite(name: str) -> list[str]:
    path = SUITES_DIR / f"{name}.yaml"
    if not path.is_file():
        raise CaseError(f"unknown suite {name!r}")
    payload = _load_yaml(path)
    ids = payload.get("cases")
    if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids):
        raise CaseError(f"{path.name} cases must be a non-empty list of ids")
    if len(ids) != len(set(ids)):
        raise CaseError(f"{path.name} has duplicate case ids")
    index = case_index()
    missing = [case_id for case_id in ids if case_id not in index]
    if missing:
        raise CaseError(f"{path.name} unknown case ids: {missing[:5]}")
    return list(ids)


def load_gates(suite: str) -> dict[str, Any]:
    payload = _load_yaml(GATES_PATH)
    suites = payload.get("suites") or {}
    if suite not in suites:
        raise CaseError(f"gates.yaml has no suite {suite!r}")
    gates = dict(suites[suite])
    gates.setdefault("version", payload.get("version"))
    return gates


def select_cases(
    *,
    suite: str | None = None,
    category: str | None = None,
    tag: str | None = None,
    case_id: str | None = None,
) -> list[Case]:
    """Diagnostic evaluation is filtering, not a third runner."""
    if case_id:
        try:
            return [case_index()[case_id]]
        except KeyError as err:
            raise CaseError(f"unknown case id {case_id!r}") from err
    if suite:
        index = case_index()
        cases = [index[item] for item in load_suite(suite)]
    else:
        cases = list(load_all_cases())
    if category:
        cases = [case for case in cases if case.category == category]
        if not cases:
            raise CaseError(f"no cases in category {category!r}")
    if tag:
        cases = [case for case in cases if tag in case.tags]
        if not cases:
            raise CaseError(f"no cases with tag {tag!r}")
    return cases


def fingerprint_cases(cases: list[Case]) -> str:
    payload = json.dumps([case.as_dict() for case in cases], sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def production_contract_fingerprint() -> str:
    files = [
        ROOT / "sayso_contract.py",
        ROOT / "custom_components" / "sayso" / "area_context.py",
        ROOT / "custom_components" / "sayso" / "completion.py",
        ROOT / "custom_components" / "sayso" / "tool_contract.py",
        ROOT / "custom_components" / "sayso" / "tool_schema.py",
        ROOT / "custom_components" / "sayso" / "lfm_parse.py",
    ]
    return fingerprint_paths(files)


def normalize_utterance(text: str) -> str:
    prompt = " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("’", "'")))
    prompt = re.sub(r"^(?:(?:please|can you|could you|tell me) )+", "", prompt)
    return re.sub(r"(?: for me)+$", "", prompt)


def excluded_train_utterances() -> set[str]:
    """Every active eval utterance. Do not train on these."""
    return {case.utterance for case in load_all_cases()}


def cases_with_tag(tag: str) -> list[Case]:
    return [case for case in load_all_cases() if tag in case.tags]


def entity_names_for_tag(tag: str, *, include_aliases: bool = True) -> set[str]:
    names: set[str] = set()
    for case in cases_with_tag(tag):
        home = load_home(case.household)
        for entity in home.get("entities") or []:
            names.add(entity["name"])
            if include_aliases:
                names.update(entity.get("aliases") or [])
    return names
