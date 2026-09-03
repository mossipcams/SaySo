"""Map Home-LLM service names onto the locked SaySo v1 tool catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

NON_V1_DOMAINS: frozenset[str] = frozenset(
    {
        "climate",
        "vacuum",
        "media_player",
        "timer",
        "todo",
        "humidifier",
    }
)

NON_V1_ACTIONS: frozenset[str] = frozenset(
    {
        "toggle",
        "stop_cover",
        "set_cover_position",
        "media_play",
        "media_pause",
        "media_next_track",
        "media_previous_track",
        "media_stop",
        "media_play_pause",
        "volume_up",
        "volume_down",
        "volume_mute",
        "start",
        "return_to_base",
        "add_item",
        "cancel",
        "pause",
        "unpause",
        "increase_timer",
        "decrease_timer",
    }
)

V1_STATUS_DEVICE_TYPES: frozenset[str] = frozenset(
    {
        "light",
        "fan",
        "garage_door",
        "blinds",
    }
)

COLOR_NAMES: tuple[str, ...] = (
    "red",
    "blue",
    "green",
    "amber",
    "purple",
    "orange",
    "yellow",
    "pink",
    "cyan",
    "white",
    "warm white",
    "cool white",
)


@dataclass(frozen=True, slots=True)
class V1ToolCall:
    """One mapped v1 tool invocation."""

    name: str
    arguments: dict[str, Any]


def parse_service_name(service_name: str) -> tuple[str, str]:
    domain, action = service_name.split(".", 1)
    return domain, action


def drop_reason_for_service(service_name: str) -> str | None:
    """Return a drop-reason label when a service cannot map to v1."""
    if "|" in service_name:
        parts = service_name.split("|")
        reasons = [drop_reason_for_service(part) for part in parts]
        if any(reason is not None for reason in reasons):
            return "non_v1_multi_action"
        return None
    domain, action = parse_service_name(service_name)
    if domain in NON_V1_DOMAINS:
        return f"non_v1_domain:{domain}"
    if action in NON_V1_ACTIONS:
        return f"non_v1_action:{action}"
    if _map_service_to_v1_core(service_name) is None:
        return "unmapped_service"
    return None


def is_mappable_service(service_name: str) -> bool:
    return drop_reason_for_service(service_name) is None


def is_mappable_status_device(device_type: str) -> bool:
    return device_type in V1_STATUS_DEVICE_TYPES


def _domain_defaults(domain: str, action: str) -> dict[str, Any]:
    if domain == "light":
        return {"domain": ["light"]}
    if domain == "fan":
        return {"domain": ["fan"]}
    if domain == "switch":
        return {"domain": ["switch"]}
    if domain == "blinds":
        device_class = ["blind"]
        if action in {"open_cover", "close_cover"}:
            return {"device_class": device_class}
        return {"device_class": device_class}
    if domain == "garage_door":
        return {"device_class": ["garage"]}
    if domain == "lock":
        return {"device_class": ["door"]}
    return {}


def map_service_to_v1(
    service_name: str,
    *,
    has_brightness: bool = False,
    has_color: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    """Map one Home-LLM service to (v1_tool_name, default_argument_fields)."""
    if drop_reason_for_service(service_name) is not None:
        return None
    return _map_service_to_v1_core(service_name, has_brightness=has_brightness, has_color=has_color)


def _map_service_to_v1_core(
    service_name: str,
    *,
    has_brightness: bool = False,
    has_color: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    """Map after domain/action filters (no recursive drop checks)."""
    if "|" in service_name:
        return None

    domain, action = parse_service_name(service_name)
    defaults = _domain_defaults(domain, action)

    if domain == "light":
        if has_brightness or has_color:
            return "HassLightSet", defaults
        if action == "turn_on":
            return "HassTurnOn", defaults
        if action == "turn_off":
            return "HassTurnOff", defaults
        return None

    if domain == "fan":
        if action in {"increase_speed", "decrease_speed"}:
            return "HassFanSetSpeed", defaults
        if action == "turn_on":
            return "HassTurnOn", defaults
        if action == "turn_off":
            return "HassTurnOff", defaults
        return None

    if domain == "switch":
        if action == "turn_on":
            return "HassTurnOn", defaults
        if action == "turn_off":
            return "HassTurnOff", defaults
        return None

    if domain == "blinds":
        if action == "open_cover":
            return "HassTurnOn", defaults
        if action == "close_cover":
            return "HassTurnOff", defaults
        return None

    if domain == "garage_door":
        if action == "open_cover":
            return "HassTurnOn", defaults
        if action == "close_cover":
            return "HassTurnOff", defaults
        return None

    if domain == "lock":
        if action == "lock":
            return "HassTurnOn", defaults
        if action == "unlock":
            return "HassTurnOff", defaults
        return None

    return None


def build_tool_call(
    service_name: str,
    friendly_name: str,
    *,
    has_brightness: bool = False,
    has_color: bool = False,
    brightness: int | None = None,
    color: str | None = None,
    percentage: int | None = None,
) -> V1ToolCall | None:
    mapped = map_service_to_v1(
        service_name,
        has_brightness=has_brightness,
        has_color=has_color,
    )
    if mapped is None:
        return None
    tool_name, defaults = mapped
    args: dict[str, Any] = {"name": friendly_name, **defaults}
    if tool_name == "HassLightSet":
        if brightness is not None:
            args["brightness"] = brightness
        if color is not None:
            args["color"] = color
    if tool_name == "HassFanSetSpeed":
        if percentage is None:
            _, action = parse_service_name(service_name)
            percentage = 75 if action == "increase_speed" else 25
        args["percentage"] = percentage
    return V1ToolCall(name=tool_name, arguments=args)


def build_live_context_call(friendly_name: str, device_type: str) -> V1ToolCall:
    args: dict[str, Any] = {"name": friendly_name}
    if device_type == "light":
        args["domain"] = ["light"]
    elif device_type == "fan":
        args["domain"] = ["fan"]
    return V1ToolCall(name="GetLiveContext", arguments=args)


def build_get_datetime_call() -> V1ToolCall:
    return V1ToolCall(name="GetDateTime", arguments={})


def build_cancel_all_timers_call(*, area: str | None = None) -> V1ToolCall:
    args: dict[str, Any] = {}
    if area:
        args["area"] = area
    return V1ToolCall(name="HassCancelAllTimers", arguments=args)


def slug_to_friendly(slug: str) -> str:
    return slug.replace("_", " ").strip().title()
