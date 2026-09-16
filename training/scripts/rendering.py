#!/usr/bin/env python3
"""Turn one spec into the row that is actually trained on.

Two jobs, in order: render the canonical OpenAI envelope for a spec, and
supply the deterministic phrasing seeds language generation must not wander
from. Nothing here talks to a model.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.context import system_prompt  # noqa: E402
from generators.tools import offered_tools, script_tool_name, script_tools  # noqa: E402
from generators.utterances import (  # noqa: E402
    expand_utterance as render_utterance,
    request_seed_from_spec,
)
from v2_scenarios import _UNAVAILABLE_TYPE, validate_spec  # noqa: E402

_BANNED_UTTERANCE = re.compile(
    r"<tool_call>|evals/cases/|tool_call_start",
    re.I,
)
_CLEAN_DIRECT_START = re.compile(
    r"^(turn|set|open|close|lock|unlock|switch)\b",
    re.I,
)

_FRAMING_PREFIXES = ("", "Please", "Could you please", "Hey, could you please", "Uh", "Okay, can you", "Would you please")
_FRAMING_SUFFIXES = ("", "please.", "for me?", "right now?", "thanks.")
_UNNATURAL_FRAMING = re.compile(
    r"\bplease\s+what\b|\b(?:could|can|would)\s+you\s+what\b",
    re.I,
)


def _system_prompt(home: dict[str, Any]) -> str:
    """Same Home Assistant prompt the v3 rows use, so eval sets stay in distribution."""
    return system_prompt(home)


def _call_id(candidate_id: str, index: int, call: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{candidate_id}:{index}:{json.dumps(call, sort_keys=True)}".encode()
    ).hexdigest()[:12]
    return f"call_{digest}"


def _final_text(spec: dict[str, Any]) -> str:
    expected = spec["expected"]
    if expected["kind"] == "status":
        return f"{spec['target_names'][0]} is {expected['state']}."
    if expected["kind"] == "action":
        return "Done."
    if expected.get("response") == "area_unavailable":
        unavailable = expected.get("unavailable") or {}
        area = unavailable.get("area", "this area")
        device_type = unavailable.get("type", "devices")
        return f"The {area} has no {device_type} available."
    return {
        "clarify": "Which device did you mean?",
        "unsupported": "I can't do that with the available Home Assistant tools.",
        "refuse": "I can't help with that request.",
    }[expected["response"]]


def render_example(spec: dict[str, Any]) -> dict[str, Any]:
    """Render a validated, verbalized specification as canonical SaySo JSONL data."""
    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    calls = spec["expected"].get("calls") or []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(spec["home"]), "train_on_turn": False},
        {"role": "user", "content": utterance.strip(), "train_on_turn": False},
    ]
    if calls:
        rendered_calls = []
        call_ids = []
        for index, call in enumerate(calls):
            call_id = _call_id(spec["candidate_id"], index, call)
            call_ids.append(call_id)
            rendered_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False, sort_keys=True),
                    },
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "train_on_turn": True,
                "tool_calls": rendered_calls,
            }
        )
        for call_id, call in zip(call_ids, calls):
            result: dict[str, Any] = {"result": "Success"}
            if call["name"] == "GetLiveContext":
                result = {
                    "entities": [
                        {"name": spec["target_names"][0], "state": spec["expected"]["state"]}
                    ]
                }
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                    "train_on_turn": False,
                    "tool_call_id": call_id,
                }
            )
    messages.append(
        {"role": "assistant", "content": _final_text(spec), "train_on_turn": True}
    )
    family = spec["contrastive_group"] or spec["candidate_id"]
    metadata = {
        "candidate_id": spec["candidate_id"],
        "template_family": spec["contrastive_group"] or spec["category"],
        "phrasing_family": spec["contrastive_group"] or spec["subcategory"],
        "seed": family,
        "generation_seed": spec["seed"],
        "category": spec["category"],
        "subcategory": spec["subcategory"],
        "home_id": spec["home"]["home_id"],
        "contrastive_group": spec["contrastive_group"],
    }
    if "quality" in spec:
        metadata["quality"] = spec["quality"]
    # Same candidate-set shape as the v3 train rows, so the eval sets rendered through
    # here are not out of distribution against a model trained on 8 offered tools.
    offered = offered_tools(
        [call["name"] for call in calls],
        spec["candidate_id"],
        extra_tools=script_tools(spec["home"]),
    )
    return {"messages": messages, "tools": offered, "metadata": metadata}


def request_seed(spec: dict[str, Any]) -> str:
    """Use the same semantic renderer as the generation pipeline."""
    return request_seed_from_spec(spec)


def expand_utterance(spec: dict[str, Any]) -> str:
    """Use the shared deterministic OHF renderer."""
    return render_utterance(spec)


def _protected_slots(spec: dict[str, Any]) -> list[tuple[str, str]]:
    slots: list[tuple[str, str]] = []
    for index, name in enumerate(spec["target_names"], start=1):
        slots.append((f"<TARGET_{index}>", spec["spoken_targets"].get(name, name)))
    for index, name in enumerate(spec["excluded_names"], start=1):
        slots.append((f"<EXCLUDED_{index}>", name))
    values: list[str] = []
    for call in spec["expected"].get("calls", []):
        for value in call["arguments"].values():
            if isinstance(value, (int, float)) and str(value) not in values:
                values.append(str(value))
    if spec["expected"]["kind"] == "no_action":
        values.extend(value for value in re.findall(r"\d+(?:\.\d+)?", spec["request_hint"]) if value not in values)
    for index, value in enumerate(values, start=1):
        slots.append((f"<VALUE_{index}>", value))
    return slots


def template_seed(spec: dict[str, Any]) -> str:
    """Replace authoritative identity/value slots before LLM verbalization."""
    seed = request_seed(spec)
    for placeholder, value in sorted(_protected_slots(spec), key=lambda item: -len(item[1])):
        seed = seed.replace(value, placeholder)
    return seed


def _expand_template(spec: dict[str, Any], utterance: str) -> str | None:
    expanded = utterance.strip()
    slots = _protected_slots(spec)
    known = {placeholder for placeholder, _value in slots}
    if set(re.findall(r"<(?:TARGET|EXCLUDED|VALUE)_\d+>", expanded)) - known:
        return None
    for placeholder, value in slots:
        if expanded.count(placeholder) != 1:
            return None
        expanded = expanded.replace(placeholder, value)
    if re.search(r"<(?:TARGET|EXCLUDED|VALUE)_\d+>", expanded):
        return None
    return expanded
