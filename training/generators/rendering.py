"""Production prompt, tool catalog, and canonical training row rendering."""

from __future__ import annotations

import hashlib
import json
import random
import re
import zlib
from typing import Any

from generators.capability_registry import operation_spec
from generators.context import area_context_for, system_prompt
from generators.tools import namespaced_tool_name, production_catalog, script_tools


def scenario_to_spec(scenario: dict[str, Any]) -> dict[str, Any]:
    """Convert structured scenario facts to a renderable spec."""
    from generators.gold import target_names_from_expected

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
        "exclusion_scope": scenario.get("exclusion_scope"),
        "contrastive_group": scenario.get("contrastive_group"),
        "phrasing_seed": scenario.get("phrasing_seed"),
        "request_hint": scenario.get("request_hint", ""),
        "stt_corruption": scenario.get("stt_corruption"),
        "utterance": scenario.get("utterance"),
        "capability": scenario.get("capability"),
        "operation": scenario.get("operation"),
        "tier": scenario.get("tier"),
        "semantic_id": scenario.get("semantic_id"),
        "family": scenario.get("family"),
        "removed_tools": scenario.get("removed_tools", []),
    }


# Share of rows (outside deliberate removals) that withhold tools they do not need.
# Without it, only `unavailable` rows had a short catalog, and the model learned
# "a tool is missing -> refuse" from catalog shape instead of checking relevance.
DECOY_WITHHOLD_RATE = 0.3


def _offered_catalog(spec: dict[str, Any], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    removed = list(spec["expected"].get("unavailable_tools") or spec.get("removed_tools") or [])
    removed_ns = [namespaced_tool_name(t) if "__" not in t else t for t in removed]
    if not removed_ns:
        removed_ns = _decoy_removals(spec, messages)
    return production_catalog(spec["home"], removed_tools=removed_ns)


def _decoy_removals(spec: dict[str, Any], messages: list[dict[str, Any]]) -> list[str]:
    """Withhold 1-2 tools the row never needs, drawn from every tool like `unavailable` does."""
    rng = random.Random(zlib.crc32(f"decoy:{spec['candidate_id']}".encode()))
    if rng.random() >= DECOY_WITHHOLD_RATE:
        return []
    needed = {call["function"]["name"] for m in messages for call in m.get("tool_calls") or []}
    op = operation_spec(spec.get("capability") or "", spec.get("operation") or "")
    if op and op.tool_name:
        needed.add(namespaced_tool_name(op.tool_name))
    for call in (spec["expected"].get("requested") or {}).get("calls") or []:
        needed.add(namespaced_tool_name(call["name"]))
    candidates = [t["function"]["name"] for t in production_catalog(spec["home"]) if t["function"]["name"] not in needed]
    return rng.sample(candidates, min(len(candidates), rng.choice((1, 1, 2))))


def _call_id(candidate_id: str, index: int, call: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{candidate_id}:{index}:{json.dumps(call, sort_keys=True)}".encode()
    ).hexdigest()[:12]
    return f"call_{digest}"


def _did_you_mean(names: list[str]) -> str:
    """Name the ambiguous devices instead of a generic 'which device'."""
    labels = [
        name if name.casefold().startswith("the ") else f"the {name}"
        for name in names[:3]
        if name
    ]
    if len(labels) == 1:
        return f"Did you mean {labels[0]}?"
    if len(labels) == 2:
        return f"Did you mean {labels[0]} or {labels[1]}?"
    if len(labels) >= 3:
        return f"Did you mean {labels[0]}, {labels[1]}, or {labels[2]}?"
    return ""


def _final_text(spec: dict[str, Any]) -> str:
    expected = spec["expected"]
    if expected["kind"] == "status":
        names = spec.get("target_names") or []
        from generators.gold import target_names_from_expected

        names = names or target_names_from_expected(expected)
        state = str(expected.get("state", "unknown"))
        # Real-home states are raw HA values: scenes/buttons hold a timestamp, to-do lists a count.
        if re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d.*", state):
            return f"{names[0]} was last activated at {state[11:16]}."
        if state.isdigit():
            return f"{names[0]} has {state} item{'' if state == '1' else 's'}."
        return f"{names[0]} is {state}."
    if expected["kind"] == "action":
        return "Done."
    if expected.get("response") == "clarify":
        named = _did_you_mean(list(expected.get("candidates") or []))
        if named:
            return named
        if spec.get("capability") == "scripts":
            return "Which routine did you mean?"
        return "Which device did you mean?"
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
        area = unavailable.get("area", "this area").casefold()  # one spelling across families
        device_type = unavailable.get("type", "devices")
        return f"The {area} has no {device_type} available."
    return {
        "not_understood": "Sorry, I didn't catch that.",
        "clarify": "Which device did you mean?",
        "unsupported": "I can't do that with the available Home Assistant tools.",
        "refuse": "I can't help with that request.",
    }.get(expected.get("response", ""), "I can't help with that request.")


def _production_metadata(spec: dict[str, Any], offered: list[dict[str, Any]], utterance: str) -> dict[str, Any]:
    return {
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
        "family": spec.get("family"),
        "home_id": spec["home"]["home_id"],
        "home_size": spec["home"].get("size"),
        "expected_target_names": spec.get("target_names", []),
        "contrastive_group": spec.get("contrastive_group"),
        "stt_corruption": spec.get("stt_corruption"),
        "paraphrase_source": spec.get("paraphrase_source"),
        "excluded_names": spec.get("excluded_names", []),
        "spoken_targets": spec.get("spoken_targets", {}),
        "no_action_reason": spec["expected"].get("response"),
        "unavailable": spec["expected"].get("unavailable"),
        "unavailable_tools": spec["expected"].get("unavailable_tools", []),
        "linguistics": spec.get("linguistics", []),
        "contract": "production_v3",
        "area_context": area_context_for(spec["home"], utterance.strip()).as_dict(),
        "offered_tool_count": len(offered),
        "catalog_source": "production_exposure",
        "full_tool_catalog": spec.get("full_tool_catalog", True),
    }


def _append_action_turn(
    messages: list[dict[str, Any]],
    spec: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    train_tool_turn: bool,
) -> None:
    tool_name = namespaced_tool_name
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
            "train_on_turn": train_tool_turn,
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


def _render_follow_up(spec: dict[str, Any]) -> dict[str, Any]:
    from generators.validation import validate_spec

    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    follow_up_reply = spec.get("follow_up_reply")
    candidates = spec.get("follow_up_clarify_candidates") or []
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    if not isinstance(follow_up_reply, str) or not follow_up_reply.strip():
        raise ValueError("missing_follow_up_reply")
    if len(candidates) < 2:
        raise ValueError("missing_follow_up_candidates")
    calls = spec["expected"].get("calls") or []
    if not calls:
        raise ValueError("follow_up_requires_action_gold")
    clarify_text = _did_you_mean(list(candidates))
    if not clarify_text:
        raise ValueError("missing_clarify_text")
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(spec["home"], utterance.strip()),
            "train_on_turn": False,
        },
        {"role": "user", "content": utterance.strip(), "train_on_turn": False},
        {"role": "assistant", "content": clarify_text, "train_on_turn": True},
        {"role": "user", "content": follow_up_reply.strip(), "train_on_turn": False},
    ]
    _append_action_turn(messages, spec, calls, train_tool_turn=True)
    offered = _offered_catalog(spec, messages)
    return {"messages": messages, "tools": offered, "metadata": _production_metadata(spec, offered, utterance)}


def _render_correction(spec: dict[str, Any]) -> dict[str, Any]:
    from generators.correction import (
        format_synthetic_validation_error,
        validation_error_message,
        wrong_name_tool_call,
    )
    from generators.validation import validate_spec

    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    wrong_name = spec.get("correction_wrong_name")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    if not isinstance(wrong_name, str) or not wrong_name.strip():
        raise ValueError("missing_correction_wrong_name")
    calls = spec["expected"].get("calls") or []
    if not calls:
        raise ValueError("correction_requires_action_gold")
    correct_call = calls[0]
    bad_call = wrong_name_tool_call(correct_call, wrong_name)
    tool_name = namespaced_tool_name
    # The messages are built below from `allowed_tools`, so pass the calls directly.
    planned = [{"tool_calls": [{"function": {"name": tool_name(call["name"])}} for call in calls]}]
    offered = _offered_catalog(spec, planned)
    allowed_tools = sorted({entry["function"]["name"] for entry in offered})
    wrong_id = _call_id(spec["candidate_id"], 0, bad_call)
    correct_id = _call_id(spec["candidate_id"], 1, correct_call)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(spec["home"], utterance.strip()),
            "train_on_turn": False,
        },
        {"role": "user", "content": utterance.strip(), "train_on_turn": False},
        {
            "role": "assistant",
            "content": "",
            "train_on_turn": False,
            "tool_calls": [
                {
                    "id": wrong_id,
                    "type": "function",
                    "function": {
                        "name": tool_name(bad_call["name"]),
                        "arguments": json.dumps(bad_call["arguments"], ensure_ascii=False, sort_keys=True),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": json.dumps(
                format_synthetic_validation_error(
                    code="schema_mismatch",
                    message=validation_error_message(wrong_name),
                    allowed_tools=allowed_tools,
                ),
                ensure_ascii=False,
            ),
            "train_on_turn": False,
            "tool_call_id": wrong_id,
        },
    ]
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "train_on_turn": True,
            "tool_calls": [
                {
                    "id": correct_id,
                    "type": "function",
                    "function": {
                        "name": tool_name(correct_call["name"]),
                        "arguments": json.dumps(correct_call["arguments"], ensure_ascii=False, sort_keys=True),
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "content": json.dumps({"result": "Success"}, ensure_ascii=False),
            "train_on_turn": False,
            "tool_call_id": correct_id,
        }
    )
    messages.append({"role": "assistant", "content": _final_text(spec), "train_on_turn": True})
    return {"messages": messages, "tools": offered, "metadata": _production_metadata(spec, offered, utterance)}


def render_example(spec: dict[str, Any]) -> dict[str, Any]:
    """Render a validated spec as canonical SaySo JSONL using production catalog."""
    family = spec.get("family")
    if family == "follow_up":
        return _render_follow_up(spec)
    if family == "correction":
        return _render_correction(spec)
    from generators.validation import validate_spec

    reason = validate_spec(spec)
    if reason:
        raise ValueError(reason)
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        raise ValueError("missing_utterance")
    calls = spec["expected"].get("calls") or []
    tool_name = namespaced_tool_name
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt(spec["home"], utterance.strip()),
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

    offered = _offered_catalog(spec, messages)

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
        "family": spec.get("family"),
        "home_id": spec["home"]["home_id"],
        "home_size": spec["home"].get("size"),
        "expected_target_names": spec.get("target_names", []),
        "contrastive_group": spec.get("contrastive_group"),
        "stt_corruption": spec.get("stt_corruption"),
        "paraphrase_source": spec.get("paraphrase_source"),
        "excluded_names": spec.get("excluded_names", []),
        "spoken_targets": spec.get("spoken_targets", {}),
        "no_action_reason": spec["expected"].get("response"),
        "unavailable": spec["expected"].get("unavailable"),
        "unavailable_tools": spec["expected"].get("unavailable_tools", []),
        "linguistics": spec.get("linguistics", []),
        "contract": "production_v3",
        "area_context": area_context_for(spec["home"], utterance.strip()).as_dict(),
        "offered_tool_count": len(offered),
        "catalog_source": "production_exposure",
        "full_tool_catalog": spec.get("full_tool_catalog", True),
    }
    return {"messages": messages, "tools": offered, "metadata": metadata}
