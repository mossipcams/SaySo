"""Deterministic utterance templates from authoritative labels."""

from __future__ import annotations

import re
import random
import zlib
from typing import Any

from generators.ohf import render_call

_CONVERSATIONAL = (
    "Hey, could you {action}.",
    "When you get a chance, {action}.",
    "Please {action}.",
    "Can you {action} for me?",
)


def vary_training_utterance(text: str, rng: random.Random) -> str:
    """Vary request style independently of the expected call/no-call decision."""
    text = text[:1].lower() + text[1:]
    if re.match(r"^(how|is|are|does|do|which)\b", text):
        return text
    for technical, spoken in (("climates", "thermostats"), ("switchs", "outlets"), ("covers", "blinds"), ("media players", "TVs")):
        text = text.replace(f"the {technical} ", f"the {spoken} ")
    # Preserve multi-action/exclusion scope; vary single-clause sentence structure.
    if " and " not in text and ", but leave " not in text:
        patterns = (
            (r"turn on (.+)", ("turn on {0}", "turn {0} on", "switch {0} on", "switch on {0}")),
            (r"turn off (.+)", ("turn off {0}", "turn {0} off", "switch {0} off", "switch off {0}")),
            (r"set (.+) brightness to (.+) percent", ("set {0} brightness to {1} percent", "dim {0} to {1} percent", "set {0} to {1} percent brightness")),
            (r"set (.+) color to (.+)", ("set {0} color to {1}", "make {0} {1}", "change {0} to {1}")),
            (r"set (.+) color temperature to (\d+)(?: kelvin)?", ("set {0} color temperature to {1} kelvin", "set {0} to {1} kelvin")),
            (r"set (.+) temperature to (.+) degrees", ("set {0} temperature to {1} degrees", "set {0} to {1} degrees", "adjust {0} to {1} degrees")),
            (r"what is the status of (.+)", ("what is the status of {0}", "what's the status of {0}", "check the status of {0}", "tell me the status of {0}")),
        )
        for pattern, alternatives in patterns:
            match = re.fullmatch(pattern, text)
            if match:
                if "brightness" in pattern and match[2].isdigit() and int(match[2]) > 50:
                    alternatives = tuple(choice for choice in alternatives if not choice.startswith("dim "))
                text = rng.choice(alternatives).format(*match.groups())
                break
    variants = {
        "turn on ": ("turn on ", "switch on "),
        "turn off ": ("turn off ", "switch off "),
        "run ": ("run ", "start ", "activate "),
        "play ": ("play ", "resume "),
        "what is the status of ": ("what is the status of ", "what is the current state of "),
    }
    for prefix, alternatives in variants.items():
        if text.startswith(prefix):
            text = rng.choice(alternatives) + text[len(prefix):]
            break
    template = rng.choice(("{action}", "{action}", "{action}", "please {action}", "could you {action}?", "can you {action} for me?"))
    if template != "{action}" and text.startswith(("what ", "what's ")):
        text = re.sub(r"^what(?: is|'s) ", "tell me ", text)
    return template.format(action=text)


def request_seed_from_spec(spec: dict[str, Any]) -> str:
    """Derive compact semantic seed from expected behavior."""
    provenance: list[dict[str, Any]] = []
    spec["linguistics"] = provenance
    expected = spec.get("expected") or {}
    if expected.get("kind") == "no_action":
        return spec.get("request_hint") or _no_action_hint(expected)
    targets = [
        spec.get("spoken_targets", {}).get(name, name)
        for name in spec.get("target_names", [])
    ]
    if expected.get("kind") == "status":
        provenance.append({"source": "sayso_fallback", "intent": "GetLiveContext"})
        if targets:
            return f"what is the status of {targets[0]}"
        call = (expected.get("calls") or [{}])[0]
        domain = (call.get("arguments") or {}).get("domain")
        if isinstance(domain, list) and domain:
            return f"what is the status of the {domain[0]}"
        return "what is the device status"
    phrases: list[str] = []
    for index, call in enumerate(expected.get("calls") or []):
        target = targets[index] if index < len(targets) else ""
        canonical = (call.get("arguments") or {}).get("name")
        if canonical in spec.get("spoken_targets", {}):
            target = spec["spoken_targets"][canonical]
        elif canonical in spec.get("target_names", []):
            target = canonical
        phrases.append(_phrase_for_call(target, call, f"{spec.get('candidate_id', '')}:{index}", provenance))
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


def _phrase_for_call(target: str, call: dict[str, Any], seed: str = "", provenance=None) -> str:
    rendered = render_call(call, target, seed, provenance=provenance)
    if rendered is not None:
        return rendered
    if provenance is not None:
        provenance.append({"source": "sayso_fallback", "intent": call["name"]})
    name, arguments = call["name"], call.get("arguments") or {}
    if not target and (arguments.get("area") or arguments.get("floor")):
        domain = arguments.get("domain") or arguments.get("device_class") or ["device"]
        noun = domain[0] if isinstance(domain, list) else domain
        target = f"the {noun.replace('_', ' ')}s"
    if arguments.get("area"):
        target += f" in {arguments['area']}"
    if arguments.get("floor"):
        target += f" on {arguments['floor']}"
    # Per-script tools are named after the script itself, not Hass*/Get*.
    if not name.startswith(("Hass", "Get")):
        return f"run {target}"
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
        settings = []
        if "brightness" in arguments:
            settings.append(f"brightness to {arguments['brightness']} percent")
        if "color" in arguments:
            settings.append(f"color to {arguments['color']}")
        if "temperature" in arguments:
            settings.append(f"color temperature to {arguments['temperature']} kelvin")
        if settings:
            return f"set {target} " + " and ".join(settings)
    if name == "HassFanSetSpeed":
        return f"set {target} speed to {arguments['percentage']} percent"
    if name == "HassClimateSetTemperature":
        return f"set {target} temperature to {arguments['temperature']} degrees"
    if name == "HassMediaPause":
        return f"pause {target}"
    if name == "HassMediaUnpause":
        return f"play {target}"
    if name == "HassSetVolume":
        return f"set {target} volume to {arguments['volume_level']} percent"
    if name == "HassSetVolumeRelative":
        step = arguments.get("volume_step")
        if step == "up":
            return f"turn up {target} volume"
        if step == "down":
            return f"turn down {target} volume"
        return f"adjust {target} volume"
    if name == "HassMediaPlayerMute":
        return f"mute {target}"
    if name == "HassStartTimer":
        duration = " and ".join(f"{arguments[unit]} {unit}" for unit in ("hours", "minutes", "seconds") if arguments.get(unit))
        suffix = f" called {arguments['name']}" if arguments.get("name") else ""
        suffix += "".join(f" {preposition} {arguments[key]}" for key, preposition in (("area", "in"), ("floor", "on")) if arguments.get(key))
        if duration:
            return f"start a timer for {duration}{suffix}"
        return f"start a timer{suffix}"
    if name == "HassPauseTimer":
        scope = "".join(f" {preposition} {arguments[key]}" for key, preposition in (("area", "in"), ("floor", "on")) if arguments.get(key))
        return f"pause the {arguments.get('name', '')} timer{scope}".replace("  ", " ")
    if name == "HassTimerStatus":
        scope = "".join(f" {preposition} {arguments[key]}" for key, preposition in (("area", "in"), ("floor", "on")) if arguments.get(key))
        return f"what is the {arguments.get('name', '')} timer status{scope}".replace("  ", " ")
    if name == "HassVacuumStart":
        return f"start {target}"
    if name == "HassVacuumReturnToBase":
        return f"send {target} home"
    if name == "HassVacuumCleanArea":
        area = arguments.get("area")
        return f"vacuum {area}" if area else f"vacuum with {target}"
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
    if category == "ambiguity" and spec.get("request_hint"):
        return spec["request_hint"]
    if category == "conversational":
        action = request_seed_from_spec(spec)
        # crc32, not builtin hash(): randomized str hashing would pick a different
        # template per process for the same spec.
        index = zlib.crc32(str(spec.get("candidate_id", "")).encode()) % len(_CONVERSATIONAL)
        template = _CONVERSATIONAL[index]
        return template.format(action=action)
    seed = request_seed_from_spec(spec)
    if category == "clean_direct" and seed:
        return seed[0].upper() + seed[1:]
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
