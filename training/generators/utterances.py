"""Deterministic utterance templates from authoritative labels."""

from __future__ import annotations

import re
from typing import Any

_CONVERSATIONAL = (
    "Hey, could you {action}.",
    "When you get a chance, {action}.",
    "Please {action}.",
    "Can you {action} for me?",
)


def request_seed_from_spec(spec: dict[str, Any]) -> str:
    """Derive compact semantic seed from expected behavior."""
    expected = spec.get("expected") or {}
    if expected.get("kind") == "no_action":
        return spec.get("request_hint") or _no_action_hint(expected)
    targets = [
        spec.get("spoken_targets", {}).get(name, name)
        for name in spec.get("target_names", [])
    ]
    if expected.get("kind") == "status":
        if targets:
            return f"what is the status of {targets[0]}"
        call = (expected.get("calls") or [{}])[0]
        domain = (call.get("arguments") or {}).get("domain")
        if isinstance(domain, list) and domain:
            return f"what is the status of the {domain[0]}"
        return "what is the device status"
    phrases: list[str] = []
    for target, call in zip(targets, expected.get("calls") or []):
        phrases.append(_phrase_for_call(target, call))
    seed = " and ".join(phrases)
    excluded = spec.get("excluded_names") or []
    if excluded:
        seed += ", but leave " + " and ".join(excluded) + " alone"
    return seed


def _no_action_hint(expected: dict[str, Any]) -> str:
    response = expected.get("response", "unsupported")
    hints = {
        "unsupported": "play music in the garage",
        "refuse": "disable the smoke alarm safety system",
        "clarify": "turn on the light",
        "area_unavailable": "turn on the light",
    }
    return hints.get(response, "do something unsupported")


def _phrase_for_call(target: str, call: dict[str, Any]) -> str:
    name, arguments = call["name"], call.get("arguments") or {}
    device_class = set(arguments.get("device_class") or [])
    if name == "HassTurnOn":
        if "door" in device_class:
            return f"lock {target}"
        if device_class & {"blind", "garage", "curtain", "shade"}:
            return f"open {target}"
        return f"turn on {target}"
    if name == "HassTurnOff":
        if "door" in device_class:
            return f"unlock {target}"
        if "garage" in device_class:
            return f"close {target}"
        if device_class:
            return f"close {target}"
        return f"turn off {target}"
    if name == "HassLightSet":
        if "brightness" in arguments:
            return f"set {target} brightness to {arguments['brightness']} percent"
        if "color" in arguments:
            return f"set {target} color to {arguments['color']}"
        if "temperature" in arguments:
            return f"set {target} color temperature to {arguments['temperature']}"
    if name == "HassFanSetSpeed":
        return f"set {target} speed to {arguments['percentage']} percent"
    if name == "HassCancelAllTimers":
        area = arguments.get("area")
        return f"cancel all timers in {area}" if area else "cancel all timers"
    if name == "GetLiveContext":
        return f"what is the status of {target}"
    area = arguments.get("area")
    if area and not target:
        domain = arguments.get("domain") or ["device"]
        domain_label = domain[0] if isinstance(domain, list) else domain
        floor = arguments.get("floor")
        if name == "HassTurnOn":
            op = "turn on"
        elif name == "HassTurnOff":
            op = "turn off"
        elif name == "HassLightSet":
            op = "set"
        else:
            op = "control"
        if floor:
            return f"{op} the {domain_label}s on {floor} in {area}"
        return f"{op} the {domain_label}s in {area}"
    return f"control {target}"


def expand_utterance(spec: dict[str, Any]) -> str:
    """Render deterministic utterance from labels."""
    category = spec.get("category", "clean_direct")
    expected = spec.get("expected") or {}
    if expected.get("kind") == "no_action":
        hint = spec.get("request_hint") or _no_action_hint(expected)
        return hint
    if category == "conversational":
        action = request_seed_from_spec(spec)
        template = _CONVERSATIONAL[hash(spec.get("candidate_id", "")) % len(_CONVERSATIONAL)]
        return template.format(action=action)
    seed = request_seed_from_spec(spec)
    if category == "clean_direct" and expected.get("calls"):
        call = expected["calls"][0]
        target_names = spec.get("target_names") or []
        target = ""
        if target_names:
            target = spec.get("spoken_targets", {}).get(target_names[0], target_names[0])
        return _direct_utterance(target, call) if target or call["name"] == "HassCancelAllTimers" else seed
    return seed


def _direct_utterance(target: str, call: dict[str, Any]) -> str:
    phrase = _phrase_for_call(target, call)
    if not phrase:
        return "Cancel all timers"
    return phrase[0].upper() + phrase[1:] if phrase else phrase


def protected_slots(spec: dict[str, Any]) -> list[tuple[str, str]]:
    slots: list[tuple[str, str]] = []
    for index, name in enumerate(spec.get("target_names") or [], start=1):
        slots.append((f"<TARGET_{index}>", spec.get("spoken_targets", {}).get(name, name)))
    for index, name in enumerate(spec.get("excluded_names") or [], start=1):
        slots.append((f"<EXCLUDED_{index}>", name))
    values: list[str] = []
    for call in (spec.get("expected") or {}).get("calls") or []:
        for value in (call.get("arguments") or {}).values():
            if isinstance(value, (int, float)) and str(value) not in values:
                values.append(str(value))
    for index, value in enumerate(values, start=1):
        slots.append((f"<VALUE_{index}>", value))
    return slots
