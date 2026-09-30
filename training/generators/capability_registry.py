
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

TIER_PROPORTIONS: dict[int, float] = {1: 0.80, 2: 0.15, 3: 0.05}

TIER1_CAPABILITY_WEIGHTS: dict[str, int] = {
    "lights": 22,
    "media_players": 14,
    "timers": 10,
    "climate": 10,
    "switches": 9,
    "fans": 7,
    "covers": 6,
    "locks": 5,
}

TIER2_CAPABILITIES: tuple[str, ...] = ("vacuums", "scenes", "scripts")
TIER3_CAPABILITIES: tuple[str, ...] = ("lawn_mowers", "todo_lists", "buttons")

MIN_OPERATION_COVERAGE: int = 3
MIN_OPERATION_FRACTION: float = 0.08

DEFAULT_NEGATIVE_RATE: float = 0.12

TRAINING_COVERAGE_EXCLUDED: frozenset[str] = frozenset()


class SupportLevel(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class OperationSpec:

    name: str
    support: SupportLevel
    tool_name: str | None = None
    blocker: str | None = None
    requires_features: tuple[str, ...] = ()
    min_coverage: int = MIN_OPERATION_COVERAGE


@dataclass(frozen=True, slots=True)
class CapabilitySpec:

    name: str
    tier: int
    domain: str
    device_class: str | None
    sampling_weight: int
    support: SupportLevel
    operations: tuple[OperationSpec, ...]
    targeting_modes: tuple[str, ...] = (
        "individual",
        "area",
        "floor",
        "multiple",
        "exclusion",
    )
    supports_queries: bool = True
    supports_ambiguity: bool = True
    blocker: str | None = None


def _op(name: str, tool: str, *features: str) -> OperationSpec:
    return OperationSpec(name, SupportLevel.SUPPORTED, tool, requires_features=features)


def _blocked(name: str, blocker: str) -> OperationSpec:
    return OperationSpec(name, SupportLevel.UNAVAILABLE, blocker=blocker)


_QUERY = _op("query_state", "GetLiveContext")
_ON_OFF = (_op("turn_on", "HassTurnOn"), _op("turn_off", "HassTurnOff"))

SCRIPT_ACTION_TOOL = "__script__"


def _capability(
    name: str,
    tier: int,
    domain: str,
    device_class: str | None,
    operations: tuple[OperationSpec, ...],
    **extra: Any,
) -> CapabilitySpec:
    return CapabilitySpec(
        name=name,
        tier=tier,
        domain=domain,
        device_class=device_class,
        sampling_weight=TIER1_CAPABILITY_WEIGHTS.get(name, 1),
        support=SupportLevel.UNAVAILABLE if extra.get("blocker") else SupportLevel.SUPPORTED,
        operations=operations,
        **extra,
    )


def _unavailable(name: str, domain: str, blocker: str, operation_blocker: str) -> CapabilitySpec:
    return _capability(
        name, 3, domain, None,
        (_blocked("control", operation_blocker), _QUERY),
        blocker=blocker,
    )


CAPABILITIES: dict[str, CapabilitySpec] = {
    spec.name: spec
    for spec in (
        _capability("lights", 1, "light", None, (
            *_ON_OFF,
            _op("set_brightness", "HassLightSet", "brightness"),
            _op("set_color", "HassLightSet", "color"),
            _op("set_color_temperature", "HassLightSet", "color_temp"),
            _QUERY,
        )),
        _capability("media_players", 1, "media_player", "tv", (
            _op("turn_on", "HassTurnOn", "on"),
            _op("turn_off", "HassTurnOff", "off"),
            _op("play", "HassMediaUnpause", "play"),
            _op("pause", "HassMediaPause", "pause"),
            _op("next_track", "HassMediaNext", "next"),
            _op("previous_track", "HassMediaPrevious", "previous"),
            _op("volume_set", "HassSetVolume", "volume"),
            _op("volume_up", "HassSetVolumeRelative", "volume_step"),
            _op("volume_down", "HassSetVolumeRelative", "volume_step"),
            _op("mute", "HassMediaPlayerMute", "mute"),
            _op("unmute", "HassMediaPlayerUnmute", "mute"),
            _op("search_and_play", "HassMediaSearchAndPlay", "search"),
            _QUERY,
        )),
        _capability("timers", 1, "timer", None, (
            _op("cancel_all", "HassCancelAllTimers"),
            _op("start", "HassStartTimer"),
            _op("pause", "HassPauseTimer"),
            _op("unpause", "HassUnpauseTimer"),
            _op("cancel", "HassCancelTimer"),
            _op("increase", "HassIncreaseTimer"),
            _op("decrease", "HassDecreaseTimer"),
            _op("status", "HassTimerStatus"),
        ), targeting_modes=("context",)),
        _capability("climate", 1, "climate", None, (
            _op("set_temperature", "HassClimateSetTemperature"), *_ON_OFF, _QUERY,
        )),
        _capability("switches", 1, "switch", "outlet", (*_ON_OFF, _QUERY)),
        _capability("fans", 1, "fan", None, (
            *_ON_OFF, _op("set_speed", "HassFanSetSpeed", "percentage"), _QUERY,
        )),
        _capability("covers", 1, "cover", "blind", (
            _op("open", "HassTurnOn"),
            _op("close", "HassTurnOff"),
            _blocked("set_position", "no position tool in Assist schema"),
            _QUERY,
        )),
        _capability("locks", 1, "lock", "door", (
            _op("lock", "HassTurnOn"), _op("unlock", "HassTurnOff"), _QUERY,
        )),
        _capability("vacuums", 2, "vacuum", None, (
            _op("start", "HassVacuumStart"),
            _op("return_home", "HassVacuumReturnToBase"),
            _op("clean_area", "HassVacuumCleanArea"),
            _QUERY,
        )),
        _capability("scenes", 2, "scene", None, (_op("activate", "HassTurnOn"), _QUERY)),
        _capability("scripts", 2, "script", None, (
            _op("run", SCRIPT_ACTION_TOOL),
            _blocked("query_state", "GetLiveContext excludes the script domain"),
        ), targeting_modes=("individual", "multiple", "exclusion")),
        _unavailable("lawn_mowers", "lawn_mower",
                     "no lawn mower tool in Assist schema", "no lawn mower tool in Assist schema"),
        _unavailable("todo_lists", "todo",
                     "no todo tool in Assist schema", "no todo list tool in Assist schema"),
        _unavailable("buttons", "button",
                     "no button tool in Assist schema", "no button press tool in Assist schema"),
    )
}

HOME_SIZE_WEIGHTS: dict[int, int] = {8: 10, 16: 35, 32: 35, 64: 20}

ORDINARY_DIFFICULTY_RATE: float = 0.75

DIFFICULTY_TAGS: tuple[str, ...] = (
    "ordinary",
    "alias_distractor",
    "similar_name",
    "large_home",
    "multi_action",
    "exclusion",
    "ambiguity",
    "unsupported",
    "stt_noise",
)


def operation_spec(capability: str, operation: str) -> OperationSpec | None:
    cap = CAPABILITIES.get(capability)
    if cap is None:
        return None
    return next((op for op in cap.operations if op.name == operation), None)


def entity_supports(entity: dict[str, Any] | None, capability: str, operation: str) -> bool:
    op = operation_spec(capability, operation)
    if op is None:
        return False
    if not op.requires_features:
        return True
    if entity is None:
        return False
    return set(op.requires_features) <= set(entity.get("capabilities") or ())


def entities_supporting(
    entities: list[dict[str, Any]], capability: str, operation: str
) -> list[dict[str, Any]]:
    return [e for e in entities if entity_supports(e, capability, operation)]


def required_features(capability: str, operation: str) -> tuple[str, ...]:
    op = operation_spec(capability, operation)
    return op.requires_features if op else ()


def covered_tool_names() -> frozenset[str]:
    names = {
        op.tool_name
        for cap in CAPABILITIES.values()
        for op in cap.operations
        if op.tool_name and op.support is not SupportLevel.UNAVAILABLE
    }
    names.add("GetDateTime")
    return frozenset(names - TRAINING_COVERAGE_EXCLUDED)


def trainable_operations(cap: CapabilitySpec) -> list[OperationSpec]:
    return [
        op
        for op in cap.operations
        if op.support in {SupportLevel.SUPPORTED, SupportLevel.PARTIAL} and op.tool_name
        and op.tool_name not in {"GetLiveContext", "GetDateTime"}
    ]


def registry_summary() -> dict[str, Any]:
    return {
        name: {
            "tier": cap.tier,
            "support": cap.support.value,
            "blocker": cap.blocker,
            "operations": [
                {"name": op.name, "support": op.support.value, "tool": op.tool_name, "blocker": op.blocker}
                for op in cap.operations
            ],
        }
        for name, cap in CAPABILITIES.items()
    }
