"""Semantic, serialized-format, and token-length validation."""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from typing import Any
from pathlib import Path

from adapters.schema import tool_schema_map, validate_tool_arguments, v2_openai_tools
from generators.tools import script_tool_name
from generators.capability_registry import CAPABILITIES, SupportLevel, entity_supports
from generators.config import DEFAULT_TOKEN_BUDGET
from generators.gold import _type_label
from generators.stt_noise import _int_to_words, utterance_contains_target

_BANNED = re.compile(r"<tool_call>|evals/cases/|tool_call_start", re.I)


def _normalize_prompt(text: str) -> str:
    prompt = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    prompt = re.sub(r"^(?:(?:please|can you|could you|tell me) )+", "", prompt)
    return re.sub(r"(?: for me)+$", "", prompt)


@lru_cache(maxsize=1)
def quality_eval_prompts() -> frozenset[str]:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.append(str(repo))
    from evals.cases import excluded_train_utterances
    return frozenset(_normalize_prompt(prompt) for prompt in excluded_train_utterances())


def check_quality_eval_overlap(utterance: str) -> bool:
    return _normalize_prompt(utterance) in quality_eval_prompts()


def corrupt_spec(spec: dict[str, Any], field: str) -> dict[str, Any]:
    corrupted = dict(spec)
    expected = dict(corrupted.get("expected", {}))
    calls = list(expected.get("calls") or [])
    if calls and field in {"wrong_tool", "wrong_entity"}:
        calls[0] = dict(calls[0])
        if field == "wrong_tool":
            calls[0]["name"] = "NonexistentTool"
        else:
            calls[0]["arguments"] = {**calls[0].get("arguments", {}), "name": "Invented Device"}
        expected["calls"] = calls
        corrupted["expected"] = expected
    return corrupted

_TIMER_NAME_TOOLS = frozenset({
    "HassStartTimer", "HassPauseTimer", "HassUnpauseTimer", "HassCancelTimer",
    "HassIncreaseTimer", "HassDecreaseTimer", "HassTimerStatus",
})


def _entity_can_run(entity: dict[str, Any], tool: str) -> bool:
    capability = entity.get("capability")
    cap = CAPABILITIES.get(capability)
    if cap is None:
        return True
    candidates = [op for op in cap.operations if op.tool_name == tool]
    if not candidates:
        return True
    return any(entity_supports(entity, capability, op.name) for op in candidates)


def validate_spec(spec: dict[str, Any]) -> str | None:
    """Validate authoritative behavior against home and pinned v2 schema."""
    expected = spec.get("expected") or {}
    calls = expected.get("calls") or []
    if expected.get("kind") == "no_action" and calls:
        return "no_action_has_calls"
    cap = CAPABILITIES.get(spec.get("capability"))
    operation = next((op for op in cap.operations if op.name == spec.get("operation")), None) if cap else None
    if expected.get("kind") == "no_action":
        if spec.get("capability") == "timers" and expected.get("response") != "unsupported":
            return "timer_inventory_refusal"
        if expected.get("response") == "unsupported" and operation and operation.support != SupportLevel.UNAVAILABLE:
            requested = expected.get("requested", {}).get("calls", [])
            withheld = set(expected.get("unavailable_tools", []))
            if not requested or any(call["name"] not in withheld for call in requested):
                return "unsupported_without_withheld_tool"
        if expected.get("response") == "device_unsupported":
            named = expected.get("unsupported_names") or []
            by_name = {e["name"]: e for e in spec.get("home", {}).get("entities", [])}
            if not named:
                return "device_unsupported_without_names"
            for name in named:
                if name not in by_name:
                    return "unknown_canonical_entity"
                if entity_supports(by_name[name], spec.get("capability"), spec.get("operation")):
                    return "contradictory_device_support"
    if any(call.get("name") == "GetLiveContext" for call in calls) and expected.get("kind") != "status":
        return "state_query_requires_status_label"
    entities = {entity["name"]: entity for entity in spec.get("home", {}).get("entities", [])}
    if expected.get("response") == "area_unavailable":
        unavailable = expected.get("unavailable") or {}
        area = unavailable.get("area", "").casefold()
        domains = {cap.domain for cap in CAPABILITIES.values() if _type_label(cap.name) == unavailable.get("type")}
        if any(
            e.get("area", "").casefold() == area
            and (e.get("domain") in domains or unavailable.get("type") == "devices")
            for e in entities.values()
        ):
            return "contradictory_absence"
    if spec.get("category") in {"multi_action", "exclusion"} and len(calls) < 2:
        return "missing_multiple_actions"
    if spec.get("category") == "exclusion" and not spec.get("excluded_names"):
        return "missing_excluded_target"
    schemas = tool_schema_map(v2_openai_tools())
    excluded = set(spec.get("excluded_names") or [])
    script_names = {
        script_tool_name(entity)
        for entity in spec.get("home", {}).get("entities", [])
        if entity.get("domain") == "script"
    }
    for call in calls:
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return "invalid_call_shape"
        if name in script_names:
            if arguments:
                return "script_tool_takes_no_arguments"
            if any(
                script_tool_name(e) == name and e["name"] in excluded
                for e in entities.values()
                if e.get("domain") == "script"
            ):
                return "excluded_entity_called"
            continue
        reason = validate_tool_arguments(name, arguments, schemas)
        if reason:
            return reason
        target = arguments.get("name")
        if target is not None and target not in entities and name not in _TIMER_NAME_TOOLS:
            return "unknown_canonical_entity"
        if target in excluded:
            return "excluded_entity_called"
        if target in entities and not _entity_can_run(entities[target], name):
            return "entity_lacks_capability"
    for canonical, spoken in (spec.get("spoken_targets") or {}).items():
        if canonical not in entities or not str(spoken).strip():
            return "invalid_spoken_target"
    return None


_STATUS_QUERY = re.compile(r"status|\bwhat|^\W*(?:please\s+)?(?:is|are|did|does|do|how)\b|\?\W*$")


def _names_one_clarify_candidate(spec: dict[str, Any], lowered: str) -> bool:
    """A clarify label is wrong when the request names exactly one candidate.

    Home Assistant resolves an exact multi-word name or alias ("the kitchen tv"
    when "Kitchen TV" exists), so asking "did you mean" there teaches a needless
    question. A shared alias matches every candidate and stays ambiguous.
    """
    expected = spec.get("expected") or {}
    if expected.get("response") == "clarify":
        candidates = list(expected.get("candidates") or [])
    else:
        candidates = list(spec.get("follow_up_clarify_candidates") or [])
    if len(candidates) < 2:
        return False
    entities = {e["name"]: e for e in (spec.get("home") or {}).get("entities", [])}

    def spoken(name: str) -> bool:
        phrases = [name, *(entities.get(name) or {}).get("aliases", [])]
        return any(
            len(p.split()) >= 2 and re.search(r"(?<!\w)" + re.escape(p.casefold()) + r"(?!\w)", lowered)
            for p in phrases
        )

    return sum(spoken(name) for name in candidates) == 1


def validate_utterance(spec: dict[str, Any]) -> str | None:
    """Check utterance aligns with expected behavior."""
    utterance = spec.get("utterance")
    if not isinstance(utterance, str) or not utterance.strip():
        return "missing_utterance"
    if _BANNED.search(utterance):
        return "banned_marker"
    expected = spec.get("expected") or {}
    lowered = utterance.casefold()
    if _names_one_clarify_candidate(spec, lowered):
        return "clarify_target_named"
    if spec.get("area_scenario") or spec.get("category") == "area_grounding":
        for name in spec.get("excluded_names") or []:
            if "leave" not in lowered or name.casefold() not in lowered:
                return "missing_exclusion"
        return None
    if expected.get("kind") == "action":
        calls = expected.get("calls") or []
        for call in calls:
            arguments = call.get("arguments") or {}
            for scope in ("area", "floor"):
                if arguments.get(scope) and str(arguments[scope]).casefold() not in lowered:
                    return "missing_expected_scope"
            for key, value in arguments.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if key in {"hours", "minutes", "seconds"} and value == 0:
                    continue
                spellings = [str(value)]
                if float(value).is_integer():
                    spellings.append(str(int(value)))
                    words = _int_to_words(int(value))
                    if words:
                        spellings.append(words)
                if not any(
                    re.search(r"(?<![\w.])" + re.escape(word) + r"(?!\w|\.\d)", lowered)
                    for word in spellings
                ):
                    return "missing_expected_value"
        has_area_target = any(
            isinstance(c.get("arguments"), dict) and c["arguments"].get("area") and not c["arguments"].get("name")
            for c in calls
        )
        if (
            calls
            and not has_area_target
            and calls[0]["name"]
            not in {
                "HassCancelAllTimers",
                "HassStartTimer",
                "HassPauseTimer",
                "HassTimerStatus",
            }
        ):
            for name in spec.get("target_names") or []:
                spoken = spec.get("spoken_targets", {}).get(name, name).casefold()
                if (
                    spoken not in lowered
                    and not utterance_contains_target(lowered, name)
                    and not utterance_contains_target(lowered, spoken)
                ):
                    area = (calls[0].get("arguments") or {}).get("area")
                    if not area or str(area).casefold() not in lowered:
                        if spec.get("discrimination"):
                            continue
                        return "missing_expected_target"
        for name in spec.get("excluded_names") or []:
            if "leave" not in lowered or name.casefold() not in lowered:
                return "missing_exclusion"
    if expected.get("kind") == "status" and not _STATUS_QUERY.search(lowered):
        return "status_not_query"
    return None


@lru_cache(maxsize=2)
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _call_with_parsed_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``call`` with its JSON-string ``arguments`` parsed into a dict."""
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
        return call
    try:
        arguments = json.loads(function["arguments"])
    except json.JSONDecodeError:
        return call
    return {**call, "function": {**function, "arguments": arguments}}


def count_row_tokens(row: dict[str, Any], *, model_name: str) -> int:
    """Token count via pinned tokenizer and chat template (tools + supervision)."""
    tokenizer = _tokenizer(model_name)
    tools = []
    for tool in row.get("tools") or []:
        fn = tool.get("function", tool)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                },
            }
        )
    messages = []
    for message in row.get("messages") or []:
        entry = {"role": message["role"], "content": message.get("content", "")}
        if message.get("tool_calls"):
            # Canonical rows keep ``arguments`` as a JSON string, which is what
            # the OpenAI wire format uses and what we write to disk. The LFM2
            # chat template calls ``.items()`` on it, so templating a canonical
            # row raises and every caller silently fell back to an estimate.
            entry["tool_calls"] = [_call_with_parsed_arguments(c) for c in message["tool_calls"]]
        if message.get("tool_call_id"):
            entry["tool_call_id"] = message["tool_call_id"]
        messages.append(entry)
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=False,
    )
    # Not ``tokenize=True``: transformers >= 5 returns a BatchEncoding there, so
    # ``len()`` counted its two keys and reported every row as 2 tokens.
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def row_characters(row: dict[str, Any]) -> int:
    """Characters the chat template will render: tools, content, and tool calls."""
    total = len(json.dumps(row.get("tools") or []))
    for message in row.get("messages") or []:
        total += len(str(message.get("content") or ""))
        total += len(json.dumps(message.get("tool_calls") or []))
    return total


# Lowest chars-per-token seen across a rendered corpus is 3.27 (p50 3.32, max
# 3.40): the catalog and static context are ordinary English and JSON, which this
# tokenizer packs very consistently. Dividing by 3.0 is therefore a safe upper
# bound on the token count -- if that bound fits the budget, the real count does.
_MIN_CHARS_PER_TOKEN = 3.0


def validate_token_budget(
    row: dict[str, Any],
    budget: int,
    *,
    tokenizer_model: str,
) -> str | None:
    """Reject oversized rows; never truncate supervision.

    Tokenizing costs ~9 ms. Generation calls this once per *candidate*, and a 40k
    corpus burns several hundred thousand candidates, so tokenizing every one of
    them added two hours to a build in order to reject almost nothing. Rows the
    cheap character bound proves are under budget skip the tokenizer and are
    measured once, on the accepted corpus, by ``manifest.build_manifest``.
    """
    if row_characters(row) / _MIN_CHARS_PER_TOKEN <= budget:
        return None
    estimated = False
    try:
        tokens = count_row_tokens(row, model_name=tokenizer_model)
    except Exception:  # noqa: BLE001 — offline fallback when HF hub unavailable
        from generators.context import serialize_context

        spec_len = len(serialize_context(row.get("metadata", {}))) + sum(
            len(str(m.get("content", ""))) for m in row.get("messages", [])
        )
        tokens = spec_len // 4
        estimated = True
    # Flagged, not silent: this fallback hid a dead tokenizer path for three
    # shipped corpora. The manifest must be able to say the counts are guesses.
    row.setdefault("metadata", {})["_token_length"] = tokens
    row["metadata"]["_token_length_estimated"] = estimated
    if tokens > budget:
        return "token_budget_exceeded"
    return None


def validate_row(
    spec: dict[str, Any],
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    tokenizer_model: str = "LiquidAI/LFM2.5-230M-Base",
    rendered: dict[str, Any] | None = None,
) -> str | None:
    reason = validate_spec(spec)
    if reason:
        return reason
    reason = validate_utterance(spec)
    if reason:
        return reason
    if rendered is not None:
        return validate_token_budget(rendered, token_budget, tokenizer_model=tokenizer_model)
    return None


def validate_serialized_row(
    row: dict[str, Any],
    *,
    token_budget: int,
    tokenizer_model: str,
) -> str | None:
    """Validate the final JSONL row envelope."""
    if not row.get("tools"):
        return "missing_tools"
    for message in row.get("messages", []):
        if message.get("role") == "assistant" and message.get("train_on_turn"):
            if message.get("tool_calls"):
                for call in message["tool_calls"]:
                    args = call.get("function", {}).get("arguments")
                    if isinstance(args, str) and len(args) > 4096:
                        return "truncated_supervision"
    return validate_token_budget(row, token_budget, tokenizer_model=tokenizer_model)
