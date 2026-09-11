"""SaySo capability registry: tier weights, operations, tool mappings, and blockers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Accepted-row tier proportions
TIER_PROPORTIONS: dict[int, float] = {1: 0.80, 2: 0.15, 3: 0.05}

# Tier 1 relative weights (normalized within the 80% tier-1 bucket)
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

# Minimum accepted rows per supported operation (prevents on/off from consuming a capability)
MIN_OPERATION_COVERAGE: int = 3
# Minimum fraction of a capability quota reserved per supported operation at scale
MIN_OPERATION_FRACTION: float = 0.08

# Share of accepted rows reserved for supervision that is not a successful action:
# refusals, clarifications and absence answers. Positive operation quotas are
# allocated over the remaining rows, so a refusal can never fill one.
DEFAULT_NEGATIVE_RATE: float = 0.12

# Pinned-contract tools this corpus deliberately does not teach. GetDateTime
# answers "what time is it" from no entity, no area and no home state, so an
# entity-graph generator has no scenario that produces it. Home Assistant still
# supplies it at runtime -- this set bounds only what the dataset claims to
# cover, and generators.tools keeps these tools out of distractor sampling so a
# row never trains "this tool is never the answer".
TRAINING_COVERAGE_EXCLUDED: frozenset[str] = frozenset({"GetDateTime"})


class SupportLevel(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """One trainable or documentable operation within a capability."""

    name: str
    support: SupportLevel
    tool_name: str | None = None
    blocker: str | None = None
    requires_features: tuple[str, ...] = ()
    min_coverage: int = MIN_OPERATION_COVERAGE


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """Runtime-aligned capability definition."""

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


def _light_ops() -> tuple[OperationSpec, ...]:
    return (
        OperationSpec("turn_on", SupportLevel.SUPPORTED, "HassTurnOn"),
        OperationSpec("turn_off", SupportLevel.SUPPORTED, "HassTurnOff"),
        OperationSpec("set_brightness", SupportLevel.SUPPORTED, "HassLightSet", requires_features=("brightness",)),
        OperationSpec("set_color", SupportLevel.SUPPORTED, "HassLightSet", requires_features=("color",)),
        OperationSpec(
            "set_color_temperature",
            SupportLevel.SUPPORTED,
            "HassLightSet",
            requires_features=("color_temp",),
        ),
        OperationSpec("query_state", SupportLevel.SUPPORTED, "GetLiveContext"),
    )


def _fan_ops() -> tuple[OperationSpec, ...]:
    return (
        OperationSpec("turn_on", SupportLevel.SUPPORTED, "HassTurnOn"),
        OperationSpec("turn_off", SupportLevel.SUPPORTED, "HassTurnOff"),
        OperationSpec("set_speed", SupportLevel.SUPPORTED, "HassFanSetSpeed", requires_features=("percentage",)),
        OperationSpec("query_state", SupportLevel.SUPPORTED, "GetLiveContext"),
    )


def _switch_ops() -> tuple[OperationSpec, ...]:
    return (
        OperationSpec("turn_on", SupportLevel.SUPPORTED, "HassTurnOn"),
        OperationSpec("turn_off", SupportLevel.SUPPORTED, "HassTurnOff"),
        OperationSpec("query_state", SupportLevel.SUPPORTED, "GetLiveContext"),
    )


def _cover_ops() -> tuple[OperationSpec, ...]:
    return (
        OperationSpec("open", SupportLevel.SUPPORTED, "HassTurnOn"),
        OperationSpec("close", SupportLevel.SUPPORTED, "HassTurnOff"),
        OperationSpec("set_position", SupportLevel.UNAVAILABLE, blocker="no position tool in Assist schema"),
        OperationSpec("query_state", SupportLevel.SUPPORTED, "GetLiveContext"),
    )


def _lock_ops() -> tuple[OperationSpec, ...]:
    return (
        OperationSpec("lock", SupportLevel.SUPPORTED, "HassTurnOn"),
        OperationSpec("unlock", SupportLevel.SUPPORTED, "HassTurnOff"),
        OperationSpec("query_state", SupportLevel.SUPPORTED, "GetLiveContext"),
    )


def _media_ops() -> tuple[OperationSpec, ...]:
    """Media players differ far more than lights: an Echo Dot has no power control
    and a dumb TV cannot search. Every operation names the entity feature it needs
    so generation never labels an action the device cannot perform.
    """
    return (
        OperationSpec("turn_on", SupportLevel.SUPPORTED, "HassTurnOn", requires_features=("on",)),
        OperationSpec("turn_off", SupportLevel.SUPPORTED, "HassTurnOff", requires_features=("off",)),
        OperationSpec("play", SupportLevel.SUPPORTED, "HassMediaUnpause", requires_features=("play",)),
        OperationSpec("pause", SupportLevel.SUPPORTED, "HassMediaPause", requires_features=("pause",)),
        OperationSpec("next_track", SupportLevel.SUPPORTED, "HassMediaNext", requires_features=("next",)),
        OperationSpec(
            "previous_track", SupportLevel.SUPPORTED, "HassMediaPrevious", requires_features=("previous",)
        ),
        OperationSpec("volume_set", SupportLevel.SUPPORTED, "HassSetVolume", requires_features=("volume",)),
        OperationSpec(
            "volume_up", SupportLevel.SUPPORTED, "HassSetVolumeRelative", requires_features=("volume_step",)
        ),
        OperationSpec(
            "volume_down", SupportLevel.SUPPORTED, "HassSetVolumeRelative", requires_features=("volume_step",)
        ),
        OperationSpec("mute", SupportLevel.SUPPORTED, "HassMediaPlayerMute", requires_features=("mute",)),
        OperationSpec("unmute", SupportLevel.SUPPORTED, "HassMediaPlayerUnmute", requires_features=("mute",)),
        OperationSpec(
            "search_and_play",
            SupportLevel.SUPPORTED,
            "HassMediaSearchAndPlay",
            requires_features=("search",),
        ),
        OperationSpec("query_state", SupportLevel.SUPPORTED, "GetLiveContext"),
    )


def _timer_ops() -> tuple[OperationSpec, ...]:
    return (
        OperationSpec("cancel_all", SupportLevel.SUPPORTED, "HassCancelAllTimers"),
        OperationSpec("start", SupportLevel.SUPPORTED, "HassStartTimer"),
        OperationSpec("pause", SupportLevel.SUPPORTED, "HassPauseTimer"),
        OperationSpec("unpause", SupportLevel.SUPPORTED, "HassUnpauseTimer"),
        OperationSpec("cancel", SupportLevel.SUPPORTED, "HassCancelTimer"),
        OperationSpec("increase", SupportLevel.SUPPORTED, "HassIncreaseTimer"),
        OperationSpec("decrease", SupportLevel.SUPPORTED, "HassDecreaseTimer"),
        OperationSpec("status", SupportLevel.SUPPORTED, "HassTimerStatus"),
    )


def _climate_ops() -> tuple[OperationSpec, ...]:
    return (
        OperationSpec("set_temperature", SupportLevel.SUPPORTED, "HassClimateSetTemperature"),
        OperationSpec("turn_on", SupportLevel.SUPPORTED, "HassTurnOn"),
        OperationSpec("turn_off", SupportLevel.SUPPORTED, "HassTurnOff"),
        OperationSpec("query_state", SupportLevel.SUPPORTED, "GetLiveContext"),
    )


def _vacuum_ops() -> tuple[OperationSpec, ...]:
    return (
        OperationSpec("start", SupportLevel.SUPPORTED, "HassVacuumStart"),
        OperationSpec("return_home", SupportLevel.SUPPORTED, "HassVacuumReturnToBase"),
        OperationSpec("clean_area", SupportLevel.SUPPORTED, "HassVacuumCleanArea"),
        OperationSpec("query_state", SupportLevel.SUPPORTED, "GetLiveContext"),
    )


def _scene_ops() -> tuple[OperationSpec, ...]:
    return (
        OperationSpec("activate", SupportLevel.SUPPORTED, "HassTurnOn"),
        OperationSpec("query_state", SupportLevel.SUPPORTED, "GetLiveContext"),
    )


# Placeholder tool name for scripts: the real name is the script's own object id, so it
# is only known per home. See generators.tools.script_tool_name.
SCRIPT_ACTION_TOOL = "__script__"


def _script_ops() -> tuple[OperationSpec, ...]:
    """Scripts are their own tools in HA 2026.8.3, and their state is not readable.

    ``async_get_exposed_entities`` buckets the script domain out of both the static
    overview and GetLiveContext, so a script's state cannot be queried at all.
    """
    return (
        OperationSpec("run", SupportLevel.SUPPORTED, SCRIPT_ACTION_TOOL),
        OperationSpec(
            "query_state",
            SupportLevel.UNAVAILABLE,
            blocker="GetLiveContext excludes the script domain",
        ),
    )


def _unavailable_ops(capability: str, blocker: str) -> tuple[OperationSpec, ...]:
    return (
        OperationSpec("control", SupportLevel.UNAVAILABLE, blocker=blocker),
        OperationSpec("query_state", SupportLevel.SUPPORTED, "GetLiveContext"),
    )


CAPABILITIES: dict[str, CapabilitySpec] = {
    "lights": CapabilitySpec(
        name="lights",
        tier=1,
        domain="light",
        device_class=None,
        sampling_weight=TIER1_CAPABILITY_WEIGHTS["lights"],
        support=SupportLevel.SUPPORTED,
        operations=_light_ops(),
    ),
    "media_players": CapabilitySpec(
        name="media_players",
        tier=1,
        domain="media_player",
        device_class="tv",
        sampling_weight=TIER1_CAPABILITY_WEIGHTS["media_players"],
        support=SupportLevel.SUPPORTED,
        operations=_media_ops(),
    ),
    "timers": CapabilitySpec(
        name="timers",
        tier=1,
        domain="timer",
        device_class=None,
        sampling_weight=TIER1_CAPABILITY_WEIGHTS["timers"],
        support=SupportLevel.SUPPORTED,
        operations=_timer_ops(),
        targeting_modes=("context",),
    ),
    "climate": CapabilitySpec(
        name="climate",
        tier=1,
        domain="climate",
        device_class=None,
        sampling_weight=TIER1_CAPABILITY_WEIGHTS["climate"],
        support=SupportLevel.SUPPORTED,
        operations=_climate_ops(),
    ),
    "switches": CapabilitySpec(
        name="switches",
        tier=1,
        domain="switch",
        device_class="outlet",
        sampling_weight=TIER1_CAPABILITY_WEIGHTS["switches"],
        support=SupportLevel.SUPPORTED,
        operations=_switch_ops(),
    ),
    "fans": CapabilitySpec(
        name="fans",
        tier=1,
        domain="fan",
        device_class=None,
        sampling_weight=TIER1_CAPABILITY_WEIGHTS["fans"],
        support=SupportLevel.SUPPORTED,
        operations=_fan_ops(),
    ),
    "covers": CapabilitySpec(
        name="covers",
        tier=1,
        domain="cover",
        device_class="blind",
        sampling_weight=TIER1_CAPABILITY_WEIGHTS["covers"],
        support=SupportLevel.SUPPORTED,
        operations=_cover_ops(),
    ),
    "locks": CapabilitySpec(
        name="locks",
        tier=1,
        domain="lock",
        device_class="door",
        sampling_weight=TIER1_CAPABILITY_WEIGHTS["locks"],
        support=SupportLevel.SUPPORTED,
        operations=_lock_ops(),
    ),
    "vacuums": CapabilitySpec(
        name="vacuums",
        tier=2,
        domain="vacuum",
        device_class=None,
        sampling_weight=1,
        support=SupportLevel.SUPPORTED,
        operations=_vacuum_ops(),
    ),
    "scenes": CapabilitySpec(
        name="scenes",
        tier=2,
        domain="scene",
        device_class=None,
        sampling_weight=1,
        support=SupportLevel.SUPPORTED,
        operations=_scene_ops(),
    ),
    "scripts": CapabilitySpec(
        name="scripts",
        tier=2,
        domain="script",
        device_class=None,
        sampling_weight=1,
        support=SupportLevel.SUPPORTED,
        operations=_script_ops(),
        targeting_modes=("individual", "multiple", "exclusion"),
    ),
    "lawn_mowers": CapabilitySpec(
        name="lawn_mowers",
        tier=3,
        domain="lawn_mower",
        device_class=None,
        sampling_weight=1,
        support=SupportLevel.UNAVAILABLE,
        operations=_unavailable_ops("lawn_mowers", "no lawn mower tool in Assist schema"),
        blocker="no lawn mower tool in Assist schema",
    ),
    "todo_lists": CapabilitySpec(
        name="todo_lists",
        tier=3,
        domain="todo",
        device_class=None,
        sampling_weight=1,
        support=SupportLevel.UNAVAILABLE,
        operations=_unavailable_ops("todo_lists", "no todo list tool in Assist schema"),
        blocker="no todo tool in Assist schema",
    ),
    "buttons": CapabilitySpec(
        name="buttons",
        tier=3,
        domain="button",
        device_class=None,
        sampling_weight=1,
        support=SupportLevel.UNAVAILABLE,
        operations=_unavailable_ops("buttons", "no button press tool in Assist schema"),
        blocker="no button tool in Assist schema",
    ),
}

# Home size distribution defaults. The 128-entity bucket is omitted: those rows render
# to ~5.6k tokens and OOM a GTX 1070 (8 GiB), and TRL drops over-length rows silently,
# so generating them would shrink the train set without saying so. Its weight moved to
# 64, which is the largest size that trains and still covers large-home behaviour.
HOME_SIZE_WEIGHTS: dict[int, int] = {8: 10, 16: 35, 32: 35, 64: 20}

# Difficulty tag sampling (~70-80% ordinary)
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


def capabilities_for_tier(tier: int) -> list[CapabilitySpec]:
    return [cap for cap in CAPABILITIES.values() if cap.tier == tier]


def operation_spec(capability: str, operation: str) -> OperationSpec | None:
    cap = CAPABILITIES.get(capability)
    if cap is None:
        return None
    return next((op for op in cap.operations if op.name == operation), None)


def entity_supports(entity: dict[str, Any] | None, capability: str, operation: str) -> bool:
    """Whether this entity can actually perform the operation.

    Home Assistant refuses an action the entity's ``supported_features`` does not
    carry, so a label that assumes every media player has power, pause, volume and
    mute teaches the model to emit calls the runtime rejects.
    """
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
    """Pinned-contract tools this corpus promises to produce positive rows for.

    ``SCRIPT_ACTION_TOOL`` stands in for the per-home script tools, whose real
    names are the scripts' object ids (see generators.tools.script_tool_name).
    """
    names = {
        op.tool_name
        for cap in CAPABILITIES.values()
        for op in cap.operations
        if op.tool_name and op.support is not SupportLevel.UNAVAILABLE
    }
    return frozenset(names - TRAINING_COVERAGE_EXCLUDED)


def unavailable_operations() -> list[tuple[str, str]]:
    """(capability, operation) pairs whose only correct answer is a refusal."""
    return [
        (cap.name, op.name)
        for cap in CAPABILITIES.values()
        for op in cap.operations
        if op.support is SupportLevel.UNAVAILABLE
    ]


def supported_operations(cap: CapabilitySpec) -> list[OperationSpec]:
    return [op for op in cap.operations if op.support in {SupportLevel.SUPPORTED, SupportLevel.PARTIAL}]


def trainable_operations(cap: CapabilitySpec) -> list[OperationSpec]:
    """Operations that produce a tool call (not query-only unsupported)."""
    return [
        op
        for op in cap.operations
        if op.support in {SupportLevel.SUPPORTED, SupportLevel.PARTIAL} and op.tool_name
        and op.tool_name not in {"GetLiveContext", "GetDateTime"}
    ]


def registry_summary() -> dict[str, Any]:
    """Human-readable registry snapshot for manifests."""
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
