"""Render validated specs to canonical SaySo JSONL training rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from generators.context import AreaContext, resolve_area_context, system_prompt
from generators.tools import namespaced_tool_name, offered_tools, script_tools
from generators.gold import target_names_from_expected
from generators.validate import validate_spec


def _call_id(candidate_id: str, index: int, call: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{candidate_id}:{index}:{json.dumps(call, sort_keys=True)}".encode()
    ).hexdigest()[:12]
    return f"call_{digest}"


def _final_text(spec: dict[str, Any]) -> str:
    expected = spec["expected"]
    if expected["kind"] == "status":
        names = spec.get("target_names") or target_names_from_expected(expected)
        return f"{names[0]} is {expected.get('state', 'unknown')}."
    if expected["kind"] == "action":
        return "Done."
    if expected.get("response") == "clarify" and spec.get("capability") == "scripts":
        return "Which routine did you mean?"
    if expected.get("response") == "device_unsupported":
        names = expected.get("unsupported_names") or []
        return f"{names[0]} does not support that." if names else "That device does not support that."
    if expected.get("response") == "device_absent":
        unavailable = expected.get("unavailable") or {}
        area = unavailable.get("area", "this area")
        device = unavailable.get("type", "device")
        return f"There is no {device} in the {area}."
    if expected.get("response") == "area_unavailable":
        unavailable = expected.get("unavailable") or {}
        area = unavailable.get("area", "this area")
        device_type = unavailable.get("type", "devices")
        return f"The {area} has no {device_type} available."
    return {
        "clarify": "Which device did you mean?",
        "unsupported": "I can't do that with the available Home Assistant tools.",
        "refuse": "I can't help with that request.",
    }.get(expected.get("response", ""), "I can't help with that request.")


def scenario_to_spec(scenario: dict[str, Any]) -> dict[str, Any]:
    """Convert a scenario to the renderer's validated spec shape."""
    expected = scenario["expected"]
    return {
        "candidate_id": scenario.get("semantic_id", f"candidate_{scenario.get('scenario_index', 0):06d}"),
        "seed": scenario.get("seed", 0),
        "category": scenario.get("robustness", "ordinary"),
        "subcategory": scenario.get("operation", ""),
        "home": scenario["home"],
        "expected": expected,
        "target_names": target_names_from_expected(expected),
        "spoken_targets": scenario.get("spoken_targets", {}),
        "excluded_names": scenario.get("excluded_names", []),
        "contrastive_group": scenario.get("contrastive_group"),
        "phrasing_seed": scenario.get("phrasing_seed"),
        "request_hint": scenario.get("request_hint", ""),
        "stt_corruption": scenario.get("stt_corruption"),
        "utterance": scenario.get("utterance"),
        "capability": scenario.get("capability"),
        "operation": scenario.get("operation"),
        "targeting": scenario.get("targeting"),
        "tier": scenario.get("tier"),
        "semantic_id": scenario.get("semantic_id"),
        "namespaced_tools": scenario.get("namespaced_tools", False),
        "full_tool_catalog": scenario.get("full_tool_catalog", False),
        "satellite_area": scenario.get("satellite_area"),
        "target_area": scenario.get("target_area"),
        "target_area_source": scenario.get("target_area_source"),
        "area_context": scenario.get("area_context", False),
    }


def render_example(spec: dict[str, Any]) -> dict[str, Any]:
    """Render a validated spec as canonical SaySo JSONL."""
    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    area_context = resolve_area_context(
        utterance,
        satellite_area=spec["home"].get("satellite_area"),
        areas=spec["home"].get("areas", ()),
        entities=spec["home"].get("entities", ()),
    )
    if (
        spec["expected"].get("response") == "clarify"
        and spec.get("category") == "ambiguity"
        and area_context.target_area_source != "missing_area"
    ):
        area_context = AreaContext(
            area_context.satellite_area, area_context.target_area, "ambiguous"
        )
    spec["satellite_area"] = area_context.satellite_area
    spec["target_area"] = area_context.target_area
    spec["target_area_source"] = area_context.target_area_source
    calls = spec["expected"].get("calls") or []
    # The label, the offered schema and the prompt must all name a tool the same
    # way, so one flag drives all three.
    namespaced = bool(spec.get("namespaced_tools"))
    tool_name = namespaced_tool_name if namespaced else (lambda name: name)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(
                spec["home"], utterance=utterance.strip(), namespaced=namespaced
            ),
            "train_on_turn": False,
        },
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
                        "name": tool_name(call["name"]),
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
                names = spec.get("target_names") or []
                result = {
                    "entities": [
                        {"name": names[0] if names else "device", "state": spec["expected"].get("state", "unknown")}
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
    messages.append({"role": "assistant", "content": _final_text(spec), "train_on_turn": True})
    metadata = {
        "candidate_id": spec["candidate_id"],
        "semantic_id": spec.get("semantic_id", spec["candidate_id"]),
        "template_family": spec.get("contrastive_group") or spec.get("category", "ordinary"),
        "phrasing_family": spec.get("subcategory", ""),
        "seed": spec.get("semantic_id", spec["candidate_id"]),
        "generation_seed": spec.get("seed"),
        "category": spec.get("category"),
        "subcategory": spec.get("subcategory"),
        "capability": spec.get("capability"),
        "operation": spec.get("operation"),
        "tier": spec.get("tier"),
        "home_id": spec["home"]["home_id"],
        "home_size": spec["home"].get("size"),
        "expected_target_names": spec.get("target_names", []),
        "satellite_area": spec["home"].get("satellite_area"),
        "target_area": spec.get("target_area"),
        "target_area_source": spec.get("target_area_source"),
        "area_context": bool(spec.get("area_context")),
        "area_context_category": spec.get("target_area_source"),
        "targeting": spec.get("targeting"),
        "contrastive_group": spec.get("contrastive_group"),
        "stt_corruption": spec.get("stt_corruption"),
        "excluded_names": spec.get("excluded_names", []),
        "spoken_targets": spec.get("spoken_targets", {}),
        "no_action_reason": spec["expected"].get("response"),
        "unavailable": spec["expected"].get("unavailable"),
        "unavailable_tools": spec["expected"].get("unavailable_tools", []),
        "linguistics": spec.get("linguistics", []),
        # Recorded so coverage is countable on accepted rows, not on config.
        "namespaced_tools": namespaced,
        "full_tool_catalog": bool(spec.get("full_tool_catalog")),
        "offered_tool_count": 0,
    }
    offered = offered_tools(
        [call["name"] for call in calls],
        spec.get("semantic_id") or spec["candidate_id"],
        extra_tools=script_tools(spec["home"]),
        excluded_names=spec["expected"].get("unavailable_tools", []),
        full_catalog=bool(spec.get("full_tool_catalog")),
        namespaced=namespaced,
    )
    metadata["offered_tool_count"] = len(offered)
    return {"messages": messages, "tools": offered, "metadata": metadata}


def render_for_trl(example: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical row with native tool arguments for TRL only."""
    from adapters.schema import extract_text_content, normalize_tool_arguments

    rendered = dict(example)
    rendered["messages"] = []
    for original in example.get("messages", []):
        message = dict(original)
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = extract_text_content(content)
        elif content is None and message.get("role") == "assistant" and message.get("tool_calls"):
            message["content"] = ""
        if message.get("role") == "assistant" and message.get("tool_calls"):
            calls = []
            for original_call in message["tool_calls"]:
                call = dict(original_call)
                function = dict(call.get("function") or {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    parsed = normalize_tool_arguments(arguments)
                    if parsed is not None:
                        function["arguments"] = parsed
                call["function"] = function
                calls.append(call)
            message["tool_calls"] = calls
        rendered["messages"].append(message)
    return rendered
