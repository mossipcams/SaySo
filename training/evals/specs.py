"""The vocabulary every locked eval set is written in.

A gold row is a home, an utterance, and the tool calls that home makes correct.
Both locked sets — the 38 recipe-lock rows and the v3 gold/shadow rows — used
to carry their own copy of these builders, which let the two drift on exactly
the details that decide a label (which argument targets an entity, what a
status call looks like). There is one copy now.
"""

from __future__ import annotations

import json
import re
from typing import Any

from evals.metrics import (
    extract_assistant_tool_calls,
    parse_tool_arguments,
    score_expected_vs_actual,
)

# A rendered eval row must never contain integration-eval ids or ChatML
# tool-call markers: either means the row leaked in from another corpus.
BANNED = re.compile(r"evals/cases/|<tool_call>|tool_call_start", re.I)

# Domain, device class, capabilities. One table so a kind means the same thing
# in every eval set.
_KINDS: dict[str, tuple[str, str | None, tuple[str, ...]]] = {
    "light": ("light", None, ("on", "off", "brightness")),
    "fan": ("fan", None, ("on", "off", "percentage")),
    "switch": ("switch", None, ("on", "off")),
    "blinds": ("cover", "blind", ("open", "close")),
    "garage_door": ("cover", "garage", ("open", "close")),
    "lock": ("lock", "door", ("lock", "unlock")),
    "climate": ("climate", None, ("heat", "cool", "off")),
    "media_player": ("media_player", "tv", ("on", "off")),
    "vacuum": ("vacuum", None, ("start", "stop")),
    "scene": ("scene", None, ("activate",)),
    "script": ("script", None, ("run",)),
}

# Home Assistant targets these by domain name. Covers and locks are missing on
# purpose: several device classes share those domains, so a call names the
# device class instead.
_DOMAIN_TARGETED = frozenset(
    {
        "light", "fan", "switch", "climate",
        "media_player", "vacuum", "scene", "script",
    }
)


def slug(name: str) -> str:
    """The entity-id fragment Home Assistant would derive from a friendly name."""
    return "".join(c.casefold() if c.isalnum() else "_" for c in name).strip("_")


def normalized(text: str) -> str:
    """Casefolded word tokens, with curly apostrophes folded to straight ones."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("’", "'")))


def entity(
    *,
    name: str,
    kind: str,
    area: str,
    aliases: list[str] | None = None,
    state: str = "off",
    floor: str = "Main Floor",
) -> dict[str, Any]:
    """One entity in an eval home."""
    domain, device_class, capabilities = _KINDS[kind]
    return {
        "entity_id": f"{domain}.{slug(name)}",
        "name": name,
        "aliases": aliases or [name],
        "domain": domain,
        "kind": kind,
        "device_class": device_class,
        "area": area,
        "floor": floor,
        "state": state,
        "capabilities": list(capabilities),
    }


def home(
    *entities: dict[str, Any],
    sayso_entity_area: str,
    home_id: str,
) -> dict[str, Any]:
    """The entity graph one row is answered against."""
    return {
        "home_id": home_id,
        "sayso_entity_area": sayso_entity_area,
        "entities": list(entities),
    }


def target_args(item: dict[str, Any], *, by_device_class: bool) -> dict[str, Any]:
    """How a call names one entity.

    Actions may fall back to ``device_class`` for the domains Home Assistant
    does not target by name; a status query never does.
    """
    args: dict[str, Any] = {"name": item["name"]}
    if item["domain"] in _DOMAIN_TARGETED:
        args["domain"] = [item["domain"]]
    elif by_device_class and item["device_class"]:
        args["device_class"] = [item["device_class"]]
    return args


def action(*calls: dict[str, Any]) -> dict[str, Any]:
    """An expectation that the model calls these tools, in this order."""
    return {"kind": "action", "calls": list(calls)}


def no_action(response: str, **extra: Any) -> dict[str, Any]:
    """An expectation that the model answers in words and calls nothing."""
    return {"kind": "no_action", "response": response, "calls": [], **extra}


def status(item: dict[str, Any]) -> dict[str, Any]:
    """An expectation that the model reads state instead of changing it."""
    return {
        "kind": "status",
        "calls": [
            {
                "name": "GetLiveContext",
                "arguments": target_args(item, by_device_class=False),
            }
        ],
        "state": item["state"],
    }


def turn_on(item: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassTurnOn", "arguments": target_args(item, by_device_class=True)}


def turn_off(item: dict[str, Any]) -> dict[str, Any]:
    return {"name": "HassTurnOff", "arguments": target_args(item, by_device_class=True)}


def light_set(item: dict[str, Any], brightness: int) -> dict[str, Any]:
    """Brightness, not temperature — the argument Run 008 got wrong most often."""
    return {
        "name": "HassLightSet",
        "arguments": {
            "name": item["name"],
            "domain": ["light"],
            "brightness": brightness,
        },
    }


def fan_speed(item: dict[str, Any], percentage: int) -> dict[str, Any]:
    """Percentage, not brightness — the other half of the same confusion."""
    return {
        "name": "HassFanSetSpeed",
        "arguments": {
            "name": item["name"],
            "domain": ["fan"],
            "percentage": percentage,
        },
    }


def _target_names(expected: dict[str, Any]) -> list[str]:
    """Entities the expectation names, for rows that do not state them."""
    return [
        call["arguments"]["name"]
        for call in expected.get("calls") or []
        if isinstance(call.get("arguments"), dict) and call["arguments"].get("name")
    ]


def spec(
    *,
    candidate_id: str,
    category: str,
    subcategory: str,
    utterance: str | None,
    home: dict[str, Any],
    expected: dict[str, Any],
    target_names: list[str] | None = None,
    request_hint: str = "",
    seed: int = 0,
    quality_eval: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """One authoritative eval row, before rendering.

    ``quality_eval`` marks a row as part of a promotion gate. Shadow rows are
    the same shape but are an overfitting check, so they carry no such claim.
    ``target_names=None`` derives the names from the expected calls; an
    explicit empty list is kept as given.
    """
    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "seed": seed,
        "category": category,
        "subcategory": subcategory,
        "home": home,
        "expected": expected,
        "target_names": (
            target_names if target_names is not None else _target_names(expected)
        ),
        "spoken_targets": {},
        "excluded_names": [],
        "contrastive_group": None,
        "request_hint": request_hint,
        "stt_corruption": None,
        "utterance": utterance,
    }
    if quality_eval:
        row["quality_eval"] = True
    return {**row, **extra}


def expected_tool_calls(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the assistant tool calls a rendered row expects."""
    batches = extract_assistant_tool_calls(example.get("messages") or [])
    return [call for batch in batches for call in batch]


def score_quality_gold(
    example: dict[str, Any], actual_messages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score exact tool name/args and no-call-when-expected for one gold row."""
    expected_calls = expected_tool_calls(example)
    actual_calls = expected_tool_calls({"messages": actual_messages})
    name_ok, args_ok, _multi_ok, category = score_expected_vs_actual(
        example.get("messages") or [], actual_messages
    )
    no_call_expected = not expected_calls
    if no_call_expected and actual_calls:
        category = category or "unexpected_tool_call"
    if not no_call_expected and not actual_calls:
        category = category or "missing_tool_call"
    no_call_ok = no_call_expected == (not actual_calls)
    return {
        "tool_name_exact": name_ok,
        "args_exact": args_ok,
        "no_call_when_expected": no_call_ok,
        "failure_category": category,
        "pass": name_ok and args_ok and no_call_ok,
    }


def assert_row_contract(example: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """Check what every locked set requires, and return its expected calls.

    A row must not carry another corpus's markers or ids, and every expected
    call must keep ``function.arguments`` as the JSON string the wire format
    uses. Set-specific rules are layered on top by the caller.
    """
    if BANNED.search(json.dumps(example, ensure_ascii=False)):
        raise ValueError(f"{label} contains banned eval or ChatML tool-call markers")
    metadata = example.get("metadata") or {}
    if str(metadata.get("candidate_id", "")).startswith("evals/cases"):
        raise ValueError(f"{label} must not use evals/cases IDs")

    calls = expected_tool_calls(example)
    for call in calls:
        args = call.get("function", {}).get("arguments")
        if not isinstance(args, str):
            raise ValueError(f"{label} keeps function.arguments as JSON strings")
        if parse_tool_arguments(args) is None:
            raise ValueError(f"{label} arguments must parse as JSON objects")
    return calls
