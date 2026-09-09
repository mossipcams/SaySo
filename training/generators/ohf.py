"""Seeded English phrasing from pinned OHF intents, bound to exact call slots.

Unsupported slot combinations return None for the existing explicit renderer.
Hassil parses the upstream syntax; this adapter samples one AST path without
enumerating the grammar's Cartesian product or parsing user/device names.
"""
from functools import lru_cache
import json
from pathlib import Path
import random
import re

from hassil.expression import Alternative, Group, ListReference, Permutation, RuleReference, TextChunk
from hassil.parse_expression import parse_sentence


@lru_cache(maxsize=1)
def _grammar():
    data = json.loads(Path(__file__).with_name("ohf_en.json").read_text())
    rules = {name: parse_sentence(text).expression for name, text in data["rules"].items()}
    entries = []
    for block in data["blocks"]:
        for sentence in block["sentences"]:
            # Bare names are ambiguous training requests; suffixed light names
            # require upstream entity-name stripping, which we do not perform.
            if sentence == "[<the>] {name}" or "{name} <light>" in sentence:
                continue
            expression = parse_sentence(sentence).expression
            refs = tuple(_references(expression, rules))
            entries.append((block, sentence, expression, refs))
    return rules, entries, data["revision"]


def _references(node, rules):
    if isinstance(node, ListReference):
        yield node
    elif isinstance(node, RuleReference):
        yield from _references(rules[node.rule_name], rules)
    elif isinstance(node, Group):
        for item in node.items:
            yield from _references(item, rules)


def _sample(node, values, rules, rng, used):
    if isinstance(node, TextChunk):
        return node.original_text
    if isinstance(node, ListReference):
        used.add(node.slot_name)
        return values[node.list_name]
    if isinstance(node, RuleReference):
        return _sample(rules[node.rule_name], values, rules, rng, used)
    if isinstance(node, Alternative):
        return _sample(rng.choice(node.items), values, rules, rng, used)
    if isinstance(node, Group):
        items = list(node.items)
        if isinstance(node, Permutation):
            rng.shuffle(items)
        return "".join(_sample(item, values, rules, rng, used) for item in items)
    raise ValueError(f"Unsupported Hassil expression: {type(node).__name__}")


def _one(value):
    if isinstance(value, list):
        return value[0] if len(value) == 1 else None
    return value


def render_call(call: dict, target: str, seed="", provenance: list | None = None) -> str | None:
    """Render only when every supplied semantic argument is represented."""
    intent = call["name"]
    args = dict(call.get("arguments") or {})
    domain = _one(args.pop("domain", None))
    if "domain" in (call.get("arguments") or {}) and domain is None:
        return None
    device_class = _one(args.get("device_class"))
    if "device_class" in args and device_class is None:
        return None
    if device_class:
        if device_class == "door" and domain == "lock":
            args.pop("device_class")
        elif domain == "cover" and args.get("name"):
            args.pop("device_class")
        else:
            args["device_class"] = device_class
    if args.get("name") and target and "Timer" not in intent:
        args["name"] = target
    # Named-only call specs may supply the resolved name separately.
    if target and not args.get("name") and not (args.get("area") or args.get("floor")) and "Timer" not in intent:
        args["name"] = target
    args = {key: value for key, value in args.items() if value is not None}
    rules, entries, revision = _grammar()
    candidates = []
    for block, sentence, expression, refs in entries:
        if block["intent"] != intent:
            continue
        inferred = block.get("inferred_domain")
        domains = block.get("name_domains")
        if inferred and inferred != domain:
            continue
        if domain and isinstance(domains, list) and domain not in domains:
            continue
        if domains == "default" and domain in {"cover", "lock", "valve", "scene", "script"}:
            continue
        if not domain and isinstance(domains, list) and intent in {"HassTurnOn", "HassTurnOff"} and not set(domains) & {"light", "switch", "fan"}:
            continue
        if set(ref.slot_name for ref in refs) != set(args):
            continue
        values = {}
        for ref in refs:
            value = args[ref.slot_name]
            # These lists map linguistic labels to numbers; numeric binding
            # would produce invalid wording (e.g. '35 brightness level').
            if ref.list_name in {"brightness_level", "color_temperature_names", "timer_half"}:
                break
            if ref.list_name in {"volume_step_up", "volume_step_down"}:
                if not isinstance(value, (int, float)) or value == 0:
                    break
                if (value > 0) != (ref.list_name == "volume_step_up"):
                    break
                value = abs(value)
            elif ref.list_name == "volume_step" and value not in {"up", "down"}:
                break
            elif ref.list_name == "cover_classes":
                value = {"blind": "blinds", "curtain": "curtains", "shade": "shades", "garage": "garage doors", "door": "doors", "window": "windows"}.get(value)
                if value is None:
                    break
            elif ref.list_name not in {ref.slot_name, "timer_name", "timer_hours", "timer_minutes", "timer_seconds", "color_temperature", "volume"}:
                break
            values[ref.list_name] = str(value)
        else:
            candidates.append((expression, values, block, sentence))
    if not candidates:
        return None
    rng = random.Random(str(seed))
    for _ in range(128):
        expression, values, block, sentence = rng.choice(candidates)
        used = set()
        literals = {f"SAYSOSLOT{index}TOKEN": value for index, (key, value) in enumerate(values.items())
                    if key in {"name", "timer_name", "area", "floor"}}
        protected = dict(values)
        for index, (key, value) in enumerate(values.items()):
            if key in {"name", "timer_name", "area", "floor"}:
                protected[key] = f"SAYSOSLOT{index}TOKEN"
        text = " ".join(_sample(expression, protected, rules, rng, used).split())
        if used == set(args) and text:
            # Recognition also permits fragments such as "lamp on". Keep
            # complete imperatives so polite framing remains grammatical.
            starts = {"HassTurnOn": r"^(turn|switch|open|lock)\b",
                      "HassTurnOff": r"^(turn|switch|close|unlock)\b"}
            if intent in starts and domain not in {"scene", "script"} and not re.search(starts[intent], text, re.I):
                continue
            if intent == "HassLightSet" and not args.get("name"):
                if "brightness" in args and "brightness" not in text:
                    continue
                if "light" not in text:
                    if "brightness" not in text:
                        continue
                    text = text.replace("brightness", "light brightness", 1)
            if intent == "HassStartTimer" and not re.match(r"^(start|set|create|begin)\b", text):
                continue
            if intent in {"HassLightSet", "HassClimateSetTemperature", "HassSetVolume", "HassFanSetSpeed"}:
                if not re.match(r"^(set|change|turn|adjust|make|dim|brighten|increase|decrease)\b", text):
                    continue
                if any(key in args for key in ("brightness", "temperature", "volume_level", "percentage")) and " to " not in text:
                    continue
                if "brightness" in args or "percentage" in args:
                    if not re.match(r"^(set|turn)\b", text):
                        continue
                    if not any(unit in text for unit in ("%", "percent", "brightness", "speed")):
                        continue
            if intent == "HassTimerStatus" and not re.match(r"^(what|how|is|are)\b", text):
                continue
            if intent == "HassSetVolumeRelative" and not re.match(r"^(turn|increase|decrease|raise|lower|change|adjust)\b", text):
                continue
            # Upstream recognition accepts optional plural suffixes; generation
            # chooses the grammatical singular/plural for explicit durations.
            def duration(match):
                adjective = re.match(r"\s+timer\b", text[match.end():])
                return match[1] + ("-" if adjective else " ") + match[3] + ("" if adjective or match[1] == "1" else "s")

            text = re.sub(r"\b(\d+)([ -])(hour|minute|second)s?\b", duration, text)
            for marker, literal in literals.items():
                if re.match(r"^(the|my|our)\b", literal, re.I) or re.search(r"['’]s\b", literal):
                    text = re.sub(r"\b(?:the|my|our) " + marker, marker, text)
                text = text.replace(marker, literal)
            if provenance is not None:
                provenance.append({"source": block["source"], "block": block["block"],
                                   "template": sentence, "revision": revision})
            return text
    return None
