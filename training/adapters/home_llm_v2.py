"""Convert Home-LLM V2 dataset entries to SaySo OpenAI tool-call JSONL."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterator, TextIO

from .schema import (
    ALLOWED_HASS_TOOLS,
    RejectionStats,
    TrainingExample,
    extract_text_content,
    is_legacy_tool_name,
    normalize_tool_arguments,
    shorten_response,
    tool_schema_map,
    v1_openai_tools,
    validate_tool_arguments,
)

_CALL_ID_PATTERN = re.compile(r"^call_\d+$")


def _reject(stats: RejectionStats | None, reason: str) -> None:
    if stats is not None:
        stats.record(reason)


def _stable_call_id(seed: int, index: int, tool_name: str, args: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{seed}:{index}:{tool_name}:{json.dumps(args, sort_keys=True)}".encode()
    ).hexdigest()[:12]
    return f"call_{digest}"


def _dedupe_key(messages: list[dict[str, Any]]) -> str:
    """Build a deduplication key from user utterance and tool targets."""
    user_text = ""
    targets: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            user_text = str(message.get("content", ""))
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                targets.append(f"{name}:{args}")
    payload = f"{user_text}|{'|'.join(sorted(targets))}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _extract_metadata(entry: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Preserve leakage-resistant split fields from source metadata."""
    meta = entry.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    template = meta.get("template_family") or meta.get("template") or ""
    phrasing = meta.get("phrasing_family") or meta.get("phrasing") or ""
    gen_seed = meta.get("seed", seed)
    return {
        "template_family": str(template),
        "phrasing_family": str(phrasing),
        "seed": gen_seed,
    }


def _message(role: str, content: str, *, train: bool = False, **extra: Any) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": role, "content": content, "train_on_turn": train}
    msg.update(extra)
    return msg


def _convert_tool_results(
    message: dict[str, Any],
    call_ids: list[str],
    call_names: list[str],
) -> list[dict[str, Any]] | None:
    """Convert Home-LLM tool result messages to SaySo format."""
    content = message.get("content")
    results: list[dict[str, Any]] = []

    if isinstance(content, list):
        for idx, item in enumerate(content):
            if isinstance(item, dict):
                if "name" in item and "response" in item:
                    response = item["response"]
                    results.append(
                        _message(
                            "tool",
                            json.dumps(response, ensure_ascii=False),
                            train=False,
                            tool_call_id=call_ids[min(idx, len(call_ids) - 1)],
                        )
                    )
                elif item.get("type") == "text":
                    text = item.get("text", "")
                    try:
                        parsed = json.loads(text) if isinstance(text, str) else text
                    except json.JSONDecodeError:
                        parsed = {"result": text}
                    if isinstance(parsed, list):
                        for sub_idx, sub in enumerate(parsed):
                            if isinstance(sub, dict):
                                tc_id = sub.get("tool_call_id") or call_ids[
                                    min(sub_idx, len(call_ids) - 1)
                                ]
                                tr = sub.get("tool_result", sub)
                                results.append(
                                    _message(
                                        "tool",
                                        json.dumps(
                                            {"result": tr}
                                            if not isinstance(tr, dict)
                                            else tr,
                                            ensure_ascii=False,
                                        ),
                                        train=False,
                                        tool_call_id=tc_id,
                                    )
                                )
                    elif isinstance(parsed, dict):
                        tc_id = parsed.get("tool_call_id") or call_ids[
                            min(idx, len(call_ids) - 1)
                        ]
                        tr = parsed.get("tool_result", parsed)
                        results.append(
                            _message(
                                "tool",
                                json.dumps(
                                    {"result": tr} if not isinstance(tr, dict) else tr,
                                    ensure_ascii=False,
                                ),
                                train=False,
                                tool_call_id=tc_id,
                            )
                        )
                else:
                    return None
            else:
                return None
        return results

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"result": content}
        if isinstance(parsed, list):
            for idx, item in enumerate(parsed):
                if not isinstance(item, dict):
                    return None
                tc_id = item.get("tool_call_id") or call_ids[min(idx, len(call_ids) - 1)]
                tr = item.get("tool_result", item)
                results.append(
                    _message(
                        "tool",
                        json.dumps(
                            {"result": tr} if not isinstance(tr, dict) else tr,
                            ensure_ascii=False,
                        ),
                        train=False,
                        tool_call_id=tc_id,
                    )
                )
            return results
        if isinstance(parsed, dict):
            tc_id = parsed.get("tool_call_id") or call_ids[0]
            tr = parsed.get("tool_result", parsed)
            return [
                _message(
                    "tool",
                    json.dumps(
                        {"result": tr} if not isinstance(tr, dict) else tr,
                        ensure_ascii=False,
                    ),
                    train=False,
                    tool_call_id=tc_id,
                )
            ]
    return None


def convert_entry(
    entry: dict[str, Any],
    *,
    seed: int = 0,
    stats: RejectionStats | None = None,
) -> TrainingExample | None:
    """Convert one Home-LLM V2 entry to SaySo training format."""
    raw_messages = entry.get("messages")
    raw_tools = entry.get("tools")
    if not isinstance(raw_messages, list) or not isinstance(raw_tools, list):
        _reject(stats, "invalid_entry_shape")
        return None

    schemas = tool_schema_map(v1_openai_tools())
    declared_names: set[str] = set()
    for tool in raw_tools:
        fn = tool.get("function")
        if not isinstance(fn, dict):
            _reject(stats, "invalid_tool_shape")
            return None
        name = fn.get("name")
        if not isinstance(name, str):
            _reject(stats, "invalid_tool_name")
            return None
        if is_legacy_tool_name(name):
            _reject(stats, "legacy_tool_name")
            return None
        if name not in ALLOWED_HASS_TOOLS:
            _reject(stats, "unknown_tool")
            return None
        declared_names.add(name)

    if not declared_names:
        _reject(stats, "empty_tool_catalog")
        return None

    # Runtime llama.cpp always receives the full locked v1 catalog.
    sayso_tools = v1_openai_tools()

    sayso_messages: list[dict[str, Any]] = []
    call_counter = 0

    idx = 0
    while idx < len(raw_messages):
        message = raw_messages[idx]
        if not isinstance(message, dict):
            _reject(stats, "invalid_message")
            return None

        role = message.get("role")
        if role in {"system", "user"}:
            text = extract_text_content(message.get("content"))
            sayso_messages.append(_message(role, text, train=False))
            idx += 1
            continue

        if role == "assistant":
            tool_calls_raw = message.get("tool_calls") or []
            answer = extract_text_content(message.get("content"))

            if tool_calls_raw:
                formatted_calls: list[dict[str, Any]] = []
                call_ids: list[str] = []
                call_names: list[str] = []

                for tc in tool_calls_raw:
                    fn = tc.get("function") if isinstance(tc, dict) else None
                    if not isinstance(fn, dict):
                        _reject(stats, "invalid_tool_call")
                        return None
                    name = fn.get("name")
                    if not isinstance(name, str):
                        _reject(stats, "invalid_tool_call_name")
                        return None
                    if is_legacy_tool_name(name):
                        _reject(stats, "legacy_tool_name")
                        return None
                    if name not in ALLOWED_HASS_TOOLS:
                        _reject(stats, "unknown_tool")
                        return None

                    args = normalize_tool_arguments(fn.get("arguments"))
                    if args is None:
                        _reject(stats, "invalid_arguments")
                        return None

                    arg_error = validate_tool_arguments(name, args, schemas)
                    if arg_error:
                        _reject(stats, arg_error)
                        return None

                    call_counter += 1
                    call_id = _stable_call_id(seed, call_counter, name, args)
                    call_ids.append(call_id)
                    call_names.append(name)
                    formatted_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, sort_keys=True, ensure_ascii=False),
                            },
                        }
                    )

                sayso_messages.append(
                    _message("assistant", "", train=True, tool_calls=formatted_calls)
                )

                idx += 1
                if idx < len(raw_messages) and raw_messages[idx].get("role") == "tool":
                    tool_results = _convert_tool_results(
                        raw_messages[idx], call_ids, call_names
                    )
                    if tool_results is None:
                        _reject(stats, "invalid_tool_result")
                        return None
                    sayso_messages.extend(tool_results)
                    idx += 1

                if answer.strip():
                    sayso_messages.append(
                        _message("assistant", shorten_response(answer), train=True)
                    )
                continue

            if answer.strip():
                sayso_messages.append(
                    _message("assistant", shorten_response(answer), train=True)
                )
            idx += 1
            continue

        if role == "tool":
            _reject(stats, "orphan_tool_message")
            return None

        _reject(stats, "unsupported_role")
        return None

    if not sayso_messages:
        _reject(stats, "empty_messages")
        return None

    return TrainingExample(
        messages=sayso_messages,
        tools=sayso_tools,
        metadata=_extract_metadata(entry, seed=seed),
    )


def convert_jsonl_stream(
    lines: Iterator[str],
    *,
    seed: int = 0,
    output: TextIO | None = None,
    view: str = "axolotl",
) -> tuple[RejectionStats, int]:
    """Convert a JSONL stream, deduplicating and optionally writing line-by-line."""
    stats = RejectionStats()
    seen: set[str] = set()
    written = 0

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            stats.record("json_decode_error")
            continue
        if not isinstance(entry, dict):
            stats.record("invalid_entry_shape")
            continue

        converted = convert_entry(entry, seed=seed + line_no, stats=stats)
        if converted is None:
            continue

        key = _dedupe_key(converted.messages)
        if key in seen:
            stats.record("duplicate")
            continue
        seen.add(key)
        written += 1
        if output is not None:
            output.write(converted.to_jsonl_line(view=view) + "\n")

    return stats, written
