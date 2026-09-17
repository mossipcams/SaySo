"""Rows that teach area grounding in the production area-context format.

Most real requests come from a satellite assigned to an area, so the model has
to read ``satellite_area``/``target_area``/``target_area_source`` correctly:
generic commands land in the satellite's area, a named area overrides it, a
name that exists in two areas needs a question, and so on. Each scenario below
is one of those situations with a label written from the home, not inferred.

Rows render through ``labels.render_example``, so the prompt, area block and
tool schema come from the integration's own code. The mix is set by a versioned
distribution config and checked after generation by ``validate_distribution``.
"""

from __future__ import annotations

import json
import random
import zlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from generators.grounding import entity
from generators.rendering import render_example

DEFAULT_DISTRIBUTION = Path(__file__).resolve().parents[2] / "configs" / "area_distribution_v1.json"

# Area, alias. Aliases are what an area registry would hold.
_AREAS: tuple[tuple[str, str], ...] = (
    ("Kitchen", "cookhouse"),
    ("Bedroom", "main bedroom"),
    ("Living Room", "lounge"),
    ("Office", "study"),
    ("Den", "tv room"),
    ("Guest Room", "spare room"),
    ("Dining Room", "dining area"),
    ("Nursery", "baby room"),
)
# Named in requests, absent from every home.
_UNKNOWN_AREAS = ("Garage", "Attic", "Basement", "Patio", "Laundry Room")

_LEGACY_AREA_MARKER = "You are in area"
_AREA_BLOCK_MARKER = "\ntarget_area_source: "


def load_distribution(path: Path | str | None = None) -> dict[str, Any]:
    plan = json.loads(Path(path or DEFAULT_DISTRIBUTION).read_text(encoding="utf-8"))
    unknown = set(plan["scenarios"]) - set(SCENARIOS)
    missing = set(SCENARIOS) - set(plan["scenarios"])
    if unknown or missing:
        raise ValueError(f"area distribution scenarios mismatch: unknown={sorted(unknown)} missing={sorted(missing)}")
    return plan


def required_counts(plan: dict[str, Any], count: int) -> dict[str, int]:
    """Rows each scenario must deliver in a corpus of ``count`` rows."""
    return {
        name: max(int(rule.get("min", 0)), round(count * float(rule["per_1000"]) / 1000))
        for name, rule in plan["scenarios"].items()
    }


def assert_required_counts_feasible(plan: dict[str, Any], count: int) -> dict[str, int]:
    """Fail closed when a listed scenario rounds to zero rows."""
    counts = required_counts(plan, count)
    zero = sorted(name for name, need in counts.items() if need <= 0)
    if zero:
        raise ValueError(
            f"area distribution resolves to zero rows at count={count} for scenarios: {zero}"
        )
    return counts


def _light(name: str, area: str, entity_id: str) -> dict[str, Any]:
    return entity(name, "lights", area, entity_id=entity_id, state="on")


def _slug(text: str) -> str:
    return "_".join(text.casefold().split())


def _home(
    index: int,
    satellite: str | None,
    areas: list[tuple[str, str]],
    lamp_areas: tuple[str, str],
    desk_area: str,
) -> dict[str, Any]:
    """Every area gets a ceiling light; the first area also an accent light.

    "Lamp" exists in exactly two areas, so it is only resolvable with an area.
    "Desk Lamp" exists once, so it is resolvable anywhere.
    """
    first = areas[0][0]
    entities = [_light(f"{area} Ceiling Light", area, f"light.{_slug(area)}_ceiling") for area, _ in areas]
    entities.append(_light(f"{first} Accent Light", first, f"light.{_slug(first)}_accent"))
    entities += [_light("Lamp", area, f"light.{_slug(area)}_lamp") for area in lamp_areas]
    entities.append(_light("Desk Lamp", desk_area, "light.desk_lamp"))
    return {
        "home_id": f"area_grounding_{index:06d}",
        "size": len(entities),
        "sayso_entity_area": satellite,
        "entities": entities,
        "active_timers": [],
        "areas": [area for area, _ in areas],
        "area_aliases": {area: [alias] for area, alias in areas},
        "area_floors": {area: "Main Floor" for area, _ in areas},
        "owners": (),
        "exposure_source": "synthetic",
    }


def _call(tool: str, **arguments: Any) -> dict[str, Any]:
    return {"name": tool, "arguments": arguments}


def _lights(**arguments: Any) -> dict[str, Any]:
    return {**arguments, "domain": ["light"]}


# Each builder takes (rng, sat, other, extra) as (area, alias) pairs and returns
# (utterance, satellite_area, lamp_areas, desk_area, expected, extra spec fields,
# clarification text or None).
_Built = tuple[str, str | None, tuple[str, str], str, dict[str, Any], dict[str, Any], str | None]


def _pick(rng: random.Random, *options: str) -> str:
    return rng.choice(options)


def _implicit(rng, sat, other, extra) -> _Built:
    utterance = _pick(rng, "turn off the lights", "switch the lights off", "turn the lights off, please")
    call = _call("HassTurnOff", **_lights(area=sat[0]))
    return utterance, sat[0], (sat[0], other[0]), extra[0], {"kind": "action", "calls": [call]}, {}, None


def _explicit(rng, sat, other, extra) -> _Built:
    utterance = _pick(rng, "turn off the {a} lights", "switch off the lights in the {a}").format(a=other[0].lower())
    call = _call("HassTurnOff", **_lights(area=other[0]))
    return utterance, sat[0], (sat[0], other[0]), extra[0], {"kind": "action", "calls": [call]}, {}, None


def _exact_in(rng, sat, other, extra) -> _Built:
    name = f"{sat[0]} Accent Light"
    utterance = _pick(rng, "turn on the {n}", "switch on {n}").format(n=name)
    call = _call("HassTurnOn", **_lights(name=name))
    return utterance, sat[0], (sat[0], other[0]), extra[0], {"kind": "action", "calls": [call]}, {}, None


def _exact_out(rng, sat, other, extra) -> _Built:
    name = f"{other[0]} Ceiling Light"
    utterance = _pick(rng, "turn on the {n}", "switch {n} on").format(n=name)
    call = _call("HassTurnOn", **_lights(name=name))
    return utterance, sat[0], (sat[0], other[0]), extra[0], {"kind": "action", "calls": [call]}, {}, None


def _unique(rng, sat, other, extra) -> _Built:
    utterance = _pick(rng, "turn on the Desk Lamp", "switch the Desk Lamp on")
    call = _call("HassTurnOn", **_lights(name="Desk Lamp"))
    return utterance, sat[0], (sat[0], other[0]), extra[0], {"kind": "action", "calls": [call]}, {}, None


def _duplicate(rng, sat, other, extra) -> _Built:
    # The satellite's own area has no Lamp, so it cannot break the tie.
    utterance = _pick(rng, "turn on the lamp", "switch the lamp on")
    question = f"Which Lamp do you mean, the one in the {other[0]} or the {extra[0]}?"
    expected = {"kind": "no_action", "response": "clarify", "calls": []}
    return utterance, sat[0], (other[0], extra[0]), sat[0], expected, {}, question


def _area_wide(rng, sat, other, extra) -> _Built:
    utterance = _pick(rng, "turn off everything in the {a}", "switch off everything in the {a}").format(a=other[0].lower())
    call = _call("HassTurnOff", area=other[0])
    return utterance, sat[0], (sat[0], other[0]), extra[0], {"kind": "action", "calls": [call]}, {}, None


def _alias(rng, sat, other, extra) -> _Built:
    utterance = _pick(rng, "turn off the lights in the {a}", "switch the {a} lights off").format(a=other[1])
    call = _call("HassTurnOff", **_lights(area=other[0]))
    return utterance, sat[0], (sat[0], other[0]), extra[0], {"kind": "action", "calls": [call]}, {}, None


def _missing_satellite(rng, sat, other, extra) -> _Built:
    utterance = _pick(rng, "turn off the lights", "switch the lights off")
    question = "Which area do you mean?"
    expected = {"kind": "no_action", "response": "clarify", "calls": []}
    return utterance, None, (sat[0], other[0]), extra[0], expected, {}, question


def _unresolved(rng, sat, other, extra) -> _Built:
    unknown = rng.choice(_UNKNOWN_AREAS)
    utterance = _pick(rng, "turn off the {a} lights", "switch off the lights in the {a}").format(a=unknown.lower())
    expected = {
        "kind": "no_action",
        "response": "area_unavailable",
        "calls": [],
        "unavailable": {"area": unknown, "type": "lights"},
    }
    return utterance, sat[0], (sat[0], other[0]), extra[0], expected, {}, None


def _conflict(rng, sat, other, extra) -> _Built:
    # Both areas have a Lamp; the named one wins over the satellite's.
    utterance = _pick(rng, "turn on the lamp in the {a}", "switch on the {a} lamp").format(a=other[0].lower())
    call = _call("HassTurnOn", **_lights(name="Lamp", area=other[0]))
    return utterance, sat[0], (sat[0], other[0]), extra[0], {"kind": "action", "calls": [call]}, {}, None


def _multi(rng, sat, other, extra) -> _Built:
    first, second = f"{sat[0]} Ceiling Light", f"{other[0]} Ceiling Light"
    utterance = _pick(
        rng, "turn off the {a} and turn on the {b}", "switch the {a} off and the {b} on"
    ).format(a=first, b=second)
    calls = [_call("HassTurnOff", **_lights(name=first)), _call("HassTurnOn", **_lights(name=second))]
    return utterance, sat[0], (sat[0], other[0]), extra[0], {"kind": "action", "calls": calls}, {"category": "multi_action"}, None


def _exclusion(rng, sat, other, extra) -> _Built:
    kept = f"{sat[0]} Accent Light"
    utterance = _pick(
        rng, "turn off the {a} lights but leave the {k} on", "leave the {k} on and switch off the other {a} lights"
    ).format(a=sat[0].lower(), k=kept)
    calls = [
        _call("HassTurnOff", **_lights(name=f"{sat[0]} Ceiling Light")),
        _call("HassTurnOff", **_lights(name="Lamp", area=sat[0])),
    ]
    fields = {"category": "exclusion", "excluded_names": [kept]}
    return utterance, sat[0], (sat[0], other[0]), extra[0], {"kind": "action", "calls": calls}, fields, None


SCENARIOS: dict[str, Callable[..., _Built]] = {
    "implicit_satellite_area": _implicit,
    "explicit_area": _explicit,
    "exact_name_in_satellite_area": _exact_in,
    "exact_name_outside_satellite_area": _exact_out,
    "unique_entity_across_home": _unique,
    "duplicate_name_clarification": _duplicate,
    "area_wide_action": _area_wide,
    "area_alias": _alias,
    "missing_satellite_area": _missing_satellite,
    "unresolved_target_area": _unresolved,
    "explicit_area_conflicts_with_satellite": _conflict,
    "multi_action_across_areas": _multi,
    "exclusion_within_area": _exclusion,
}


def build_area_spec(scenario: str, index: int, seed: int) -> dict[str, Any]:
    """Structured spec for one area scenario, before pipeline validation."""
    rng = random.Random(zlib.crc32(f"{seed}:{scenario}:{index}".encode()))
    sat, other, extra = rng.sample(_AREAS, 3)
    utterance, satellite, lamp_areas, desk_area, expected, fields, question = SCENARIOS[scenario](
        rng, sat, other, extra
    )
    home = _home(index, satellite, [sat, other, extra], lamp_areas, desk_area)
    candidate_id = f"area_{scenario}_{seed}_{index:05d}"
    names = [call["arguments"]["name"] for call in expected["calls"] if "name" in call["arguments"]]
    return {
        "candidate_id": candidate_id,
        "semantic_id": candidate_id,
        "seed": seed,
        "category": "area_grounding",
        "subcategory": scenario,
        "capability": "lights",
        "operation": "turn_off" if "off" in utterance.lower() else "turn_on",
        "tier": 1,
        "family": "area",
        "area_scenario": scenario,
        "home": home,
        "expected": expected,
        "target_names": names,
        "spoken_targets": {},
        "excluded_names": fields.get("excluded_names", []),
        "contrastive_group": None,
        "request_hint": "",
        "stt_corruption": None,
        "utterance": utterance,
        "clarify_question": question,
        **{key: value for key, value in fields.items() if key not in {"excluded_names", "category"}},
        "behavior_category": fields.get("category"),
    }


def build_row(scenario: str, index: int, seed: int) -> dict[str, Any]:
    """One deterministic row for ``scenario``."""
    spec = build_area_spec(scenario, index, seed)
    row = render_example(spec)
    if spec.get("clarify_question"):
        row["messages"][-1]["content"] = spec["clarify_question"]
    row["metadata"]["area_scenario"] = scenario
    return row


def build_rows(
    required: dict[str, int],
    *,
    seed: int,
    reject: Callable[[str], bool] = lambda _utterance: False,
) -> list[dict[str, Any]]:
    """``required[scenario]`` rows per scenario, minus any whose request ``reject`` refuses."""
    rows = []
    for scenario, count in required.items():
        for index in range(count):
            row = build_row(scenario, index, seed)
            user = next(m["content"] for m in row["messages"] if m["role"] == "user")
            if not reject(user):
                rows.append(row)
    return rows


def validate_distribution(
    rows: list[dict[str, Any]],
    plan: dict[str, Any] | None,
    count: int,
) -> dict[str, Any]:
    """Fail the build on a short area scenario, too few satellite-area rows, or
    any row whose prompt is not in the production area format."""
    legacy = [
        row["metadata"].get("candidate_id")
        for row in rows
        if _LEGACY_AREA_MARKER in row["messages"][0]["content"]
        or _AREA_BLOCK_MARKER not in row["messages"][0]["content"]
    ]
    if legacy:
        raise RuntimeError(f"{len(legacy)} rows are not in the production area format, e.g. {legacy[:3]}")

    actionable = [row for row in rows if any(m.get("tool_calls") for m in row["messages"])]
    with_satellite = sum(1 for row in actionable if row["metadata"]["area_context"]["satellite_area"])
    share = with_satellite / max(len(actionable), 1)
    report: dict[str, Any] = {
        "satellite_area_share_of_actionable_rows": round(share, 4),
        "source_counts": dict(
            sorted(Counter(row["metadata"]["area_context"]["target_area_source"] for row in rows).items())
        ),
    }
    if plan is None:
        return report

    floor = float(plan["min_satellite_area_share_of_actionable_rows"])
    if actionable and share < floor:
        raise RuntimeError(f"satellite-area share of actionable rows {share:.3f} is below {floor}")
    required = required_counts(plan, count)
    achieved = Counter(row["metadata"].get("area_scenario") for row in rows)
    short = {name: (achieved[name], need) for name, need in required.items() if achieved[name] < need}
    if short:
        raise RuntimeError(f"area scenarios below their minimum (achieved, required): {short}")
    report.update(
        {
            "version": plan["version"],
            "required": required,
            "achieved": {name: achieved[name] for name in required},
            "rows": sum(achieved[name] for name in required),
            "share": round(sum(achieved[name] for name in required) / max(count, 1), 4),
        }
    )
    return report

