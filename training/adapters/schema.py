
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_SCHEMA_ARTIFACT = REPO_ROOT / "schemas" / "sayso-tool-schema-v1.json"
V2_SCHEMA_ARTIFACT = REPO_ROOT / "schemas" / "sayso-tool-schema-v2.json"
TRAINING_V1_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sayso_tool_schema_v1.json"
TRAINING_V2_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sayso_tool_schema_v2.json"

TRAINING_TOOL_DEVICE_TYPE_TIERS: dict[str, frozenset[str]] = {
    "query": frozenset({"GetDateTime", "GetLiveContext"}),
    "generic": frozenset({"HassTurnOff", "HassTurnOn"}),
    "light": frozenset({"HassLightSet"}),
    "fan": frozenset({"HassFanSetSpeed"}),
    "climate": frozenset({"HassClimateSetTemperature"}),
    "media_player": frozenset(
        {
            "HassMediaNext",
            "HassMediaPause",
            "HassMediaPlayerMute",
            "HassMediaPlayerUnmute",
            "HassMediaPrevious",
            "HassMediaSearchAndPlay",
            "HassMediaUnpause",
            "HassSetVolume",
            "HassSetVolumeRelative",
        }
    ),
    "vacuum": frozenset(
        {
            "HassVacuumCleanArea",
            "HassVacuumReturnToBase",
            "HassVacuumStart",
        }
    ),
    "timer": frozenset(
        {
            "HassCancelAllTimers",
            "HassCancelTimer",
            "HassDecreaseTimer",
            "HassIncreaseTimer",
            "HassPauseTimer",
            "HassStartTimer",
            "HassTimerStatus",
            "HassUnpauseTimer",
        }
    ),
}

V1_TOOL_DEVICE_TYPE_TIERS = TRAINING_TOOL_DEVICE_TYPE_TIERS

LEGACY_ARGUMENT_KEYS: frozenset[str] = frozenset(
    {
        "entity_id",
        "target_device",
        "service",
        "service_name",
    }
)

LEGACY_TOOL_PREFIXES: tuple[str, ...] = (
    "light.",
    "switch.",
    "fan.",
    "lock.",
    "cover.",
    "climate.",
    "media_player.",
    "vacuum.",
    "timer.",
    "todo.",
)

CHATML_TOOL_CALL_MARKERS: frozenset[str] = frozenset(
    {
        "<tool_call>",
        "</tool_call>",
    }
)


def _load_artifact(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} schema artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_tools(schema: dict[str, Any], label: str) -> tuple[dict[str, Any], ...]:
    tools = schema.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{label} schema must contain a non-empty tools list")
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"Tool at index {index} must be type:function")
        fn = tool.get("function")
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            raise ValueError(f"Tool at index {index} must declare function.name")
    return tuple(tools)


def _assert_tiers_cover(allowed: frozenset[str]) -> None:
    tiered: set[str] = set()
    for names in TRAINING_TOOL_DEVICE_TYPE_TIERS.values():
        if overlap := tiered & set(names):
            raise ValueError(f"tool tier overlap: {sorted(overlap)}")
        tiered.update(names)
    if tiered != set(allowed):
        missing = sorted(set(allowed) - tiered)
        extra = sorted(tiered - set(allowed))
        raise ValueError(f"tier/catalog mismatch: missing={missing!r} extra={extra!r}")


@lru_cache(maxsize=1)
def load_v2_schema() -> dict[str, Any]:
    return _load_artifact(V2_SCHEMA_ARTIFACT, "tiered")


@lru_cache(maxsize=1)
def load_v1_schema() -> dict[str, Any]:
    return _load_artifact(V1_SCHEMA_ARTIFACT, "locked")


@lru_cache(maxsize=1)
def load_v2_tools() -> tuple[dict[str, Any], ...]:
    return _validated_tools(load_v2_schema(), "Pinned v2")


@lru_cache(maxsize=1)
def load_v1_tools() -> tuple[dict[str, Any], ...]:
    return _validated_tools(load_v1_schema(), "Locked v1")


def v2_tool_catalog_by_device_type() -> dict[str, list[dict[str, Any]]]:
    catalog = load_v2_schema().get("tool_catalog_by_device_type")
    if not isinstance(catalog, dict):
        raise ValueError("v2 schema must contain tool_catalog_by_device_type")
    return {str(key): list(value) for key, value in catalog.items()}


def v2_tool_names() -> frozenset[str]:
    return frozenset(tool["function"]["name"] for tool in load_v2_tools())


def v1_tool_names() -> frozenset[str]:
    return frozenset(tool["function"]["name"] for tool in load_v1_tools())


ALLOWED_HASS_TOOLS: frozenset[str] = v2_tool_names()


def v2_tool_device_type_tiers() -> dict[str, frozenset[str]]:
    return TRAINING_TOOL_DEVICE_TYPE_TIERS


v1_tool_device_type_tiers = v2_tool_device_type_tiers


def assert_v2_tiers_cover_catalog() -> None:
    _assert_tiers_cover(ALLOWED_HASS_TOOLS)


def assert_v1_tiers_cover_catalog() -> None:
    _assert_tiers_cover(v1_tool_names())


def v2_openai_tools() -> list[dict[str, Any]]:
    return [dict(tool) for tool in load_v2_tools()]


def v1_openai_tools() -> list[dict[str, Any]]:
    return [dict(tool) for tool in load_v1_tools()]


def assert_openai_tool_envelope(tool: dict[str, Any]) -> None:
    if tool.get("type") != "function":
        raise ValueError("tool entry must have type 'function'")
    fn = tool.get("function")
    if not isinstance(fn, dict):
        raise ValueError("tool entry must include function object")
    if not isinstance(fn.get("name"), str):
        raise ValueError("tool.function.name must be a string")
    params = fn.get("parameters")
    if not isinstance(params, dict):
        raise ValueError("tool.function.parameters must be an object")


def assert_tools_subset_of_v1(tools: list[dict[str, Any]]) -> None:
    assert_tools_subset_of_v2(tools)


def assert_tools_subset_of_v2(tools: list[dict[str, Any]]) -> None:
    allowed = ALLOWED_HASS_TOOLS
    for tool in tools:
        assert_openai_tool_envelope(tool)
        name = tool["function"]["name"]
        if name not in allowed:
            raise ValueError(f"tool {name!r} is not in pinned v2 catalog")


def contains_chatml_tool_call_markers(text: str) -> bool:
    return any(marker in text for marker in CHATML_TOOL_CALL_MARKERS)


@dataclass(frozen=True, slots=True)
class RejectionStats:

    counts: dict[str, int] = field(default_factory=dict)

    def record(self, reason: str) -> None:
        object.__setattr__(
            self,
            "counts",
            {**self.counts, reason: self.counts.get(reason, 0) + 1},
        )

    def merge(self, other: RejectionStats) -> RejectionStats:
        merged = dict(self.counts)
        for reason, count in other.counts.items():
            merged[reason] = merged.get(reason, 0) + count
        return RejectionStats(counts=merged)


@dataclass(frozen=True, slots=True)
class TrainingExample:

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl_line(self, *, view: str = "sayso") -> str:
        if view == "axolotl":
            messages = _axolotl_messages(self.messages)
        elif view in {"sayso", "lfm"}:
            messages = self.messages
        else:
            raise ValueError(f"unknown view: {view}")
        payload = {"messages": messages, "tools": self.tools}
        if self.metadata:
            payload["metadata"] = self.metadata
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _axolotl_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        msg = dict(message)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls: list[dict[str, Any]] = []
            for tc in msg["tool_calls"]:
                call = dict(tc)
                fn = dict(call.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    parsed = normalize_tool_arguments(args)
                    if parsed is not None:
                        fn["arguments"] = parsed
                call["function"] = fn
                calls.append(call)
            msg["tool_calls"] = calls
        out.append(msg)
    return out


def extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def tool_schema_map(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for tool in tools:
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        params = fn.get("parameters")
        if isinstance(name, str) and isinstance(params, dict):
            result[name] = params
    return result


def allowed_properties(schema: dict[str, Any]) -> frozenset[str]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return frozenset()
    return frozenset(str(key) for key in properties)


def validate_tool_arguments(
    tool_name: str,
    args: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
) -> str | None:
    for key in args:
        if key in LEGACY_ARGUMENT_KEYS:
            return "legacy_argument_key"

    schema = schemas.get(tool_name)
    if schema is None:
        return "unknown_tool_schema"

    allowed = allowed_properties(schema)
    for key in args:
        if key not in allowed:
            return "extra_argument"

    required = schema.get("required") or []
    if isinstance(required, list):
        for key in required:
            if key not in args:
                return "missing_required_argument"

    try:
        validator = Draft202012Validator(schema)
        validator.validate(args)
    except jsonschema.ValidationError:
        return "schema_validation_failed"
    return None


def normalize_tool_arguments(args: Any) -> dict[str, Any] | None:
    if isinstance(args, dict):
        return dict(args)
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


