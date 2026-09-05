"""Render validated specs to canonical SaySo JSONL training rows."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from adapters.schema import v1_openai_tools
from generators.context import system_prompt
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
    """Convert v3 scenario to legacy spec shape for rendering."""
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
        "request_hint": scenario.get("request_hint", ""),
        "stt_corruption": scenario.get("stt_corruption"),
        "utterance": scenario.get("utterance"),
        "capability": scenario.get("capability"),
        "operation": scenario.get("operation"),
        "tier": scenario.get("tier"),
        "semantic_id": scenario.get("semantic_id"),
    }


def render_example(spec: dict[str, Any]) -> dict[str, Any]:
    """Render a validated spec as canonical SaySo JSONL."""
    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    calls = spec["expected"].get("calls") or []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(spec["home"]), "train_on_turn": False},
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
        "contrastive_group": spec.get("contrastive_group"),
        "stt_corruption": spec.get("stt_corruption"),
        "paraphrase_source": spec.get("paraphrase_source"),
    }
    return {"messages": messages, "tools": v1_openai_tools(), "metadata": metadata}
