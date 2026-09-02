"""Map Home-LLM tool calls to SaySo ControlPlan objects."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from sayso_server.control_plan import ControlPlan
from sayso_server.models import ENTITY_ID_PATTERN, ClimateMode, is_entity_id

from train.home_llm import HomeLlmRow, ParsedDevice, ToolCallRecord

UNSUPPORTED_TOOL_NAMES = frozenset(
    {
        "HassVacuumStart",
        "HassVacuumReturnToBase",
        "HassStartTimer",
        "HassCancelTimer",
        "HassPauseTimer",
        "HassUnpauseTimer",
        "HassIncreaseTimer",
        "HassDecreaseTimer",
        "HassTimerStatus",
        "HassListAddItem",
        "HassSetVolume",
        "HassMediaPause",
        "HassMediaUnpause",
        "HassMediaNext",
        "HassMediaPrevious",
        "HassSetPosition",
        "HassHumidifierSetpoint",
        "HassHumidifierMode",
        "HassClimateGetTemperature",
    }
)

UNSUPPORTED_SERVICE_PREFIXES = (
    "timer.",
    "todo.",
    "vacuum.",
    "media_player.",
)

SERVICE_DOMAIN_ALIASES = {
    "garage_door": "cover",
    "blinds": "cover",
}


def map_row_to_control_plan(row: HomeLlmRow) -> dict[str, Any] | None:
    """Return a ControlPlan dict for an expressible row, or None to drop."""
    if not row.tool_calls:
        return None

    if len(row.tool_calls) != 1:
        return None

    call = row.tool_calls[0]
    if row.declared_tool_names and call.name not in row.declared_tool_names:
        return None
    if call.name in UNSUPPORTED_TOOL_NAMES:
        return None

    devices_by_id = {device.entity_id: device for device in row.devices}
    payload = _map_tool_call(call, row.user_text, devices_by_id)
    if payload is None:
        return None

    try:
        plan = ControlPlan.model_validate(payload)
    except ValidationError:
        return None

    return plan.model_dump(mode="json", exclude_none=True)


def _map_tool_call(
    call: ToolCallRecord,
    user_text: str,
    devices_by_id: dict[str, ParsedDevice],
) -> dict[str, Any] | None:
    if "." in call.name and not call.name.startswith("Hass"):
        return _map_service_call(call, user_text, devices_by_id)
    return _map_hass_tool(call, user_text, devices_by_id)


def _map_service_call(
    call: ToolCallRecord,
    user_text: str,
    devices_by_id: dict[str, ParsedDevice],
) -> dict[str, Any] | None:
    domain, _, action = call.name.partition(".")
    if any(call.name.startswith(prefix) for prefix in UNSUPPORTED_SERVICE_PREFIXES):
        return None
    domain = SERVICE_DOMAIN_ALIASES.get(domain, domain)
    return _map_domain_action(
        domain=domain,
        action=action,
        arguments=call.arguments,
        user_text=user_text,
        devices_by_id=devices_by_id,
    )


def _map_hass_tool(
    call: ToolCallRecord,
    user_text: str,
    devices_by_id: dict[str, ParsedDevice],
) -> dict[str, Any] | None:
    args = call.arguments
    domain = _argument_domain(args, devices_by_id)
    if domain is None:
        return None

    if call.name == "HassTurnOn":
        if domain == "lock":
            return _action(user_text, domain, devices_by_id, args, state="lock")
        if domain == "cover":
            return _action(user_text, domain, devices_by_id, args, state="open")
        if "brightness" in args or "color" in args:
            if "color" in args:
                return None
            return _action(
                user_text,
                domain,
                devices_by_id,
                args,
                value=int(args["brightness"]),
            )
        return _action(user_text, domain, devices_by_id, args, state="on")

    if call.name == "HassTurnOff":
        if domain == "lock":
            return _action(user_text, domain, devices_by_id, args, state="unlock")
        if domain == "cover":
            return _action(user_text, domain, devices_by_id, args, state="close")
        return _action(user_text, domain, devices_by_id, args, state="off")

    if call.name == "HassToggle":
        return _action(user_text, domain, devices_by_id, args, state="toggle")

    if call.name == "HassLightSet":
        if "color" in args or "rgb_color" in args:
            return None
        brightness = args.get("brightness")
        if brightness is None:
            return None
        return _action(
            user_text,
            "light",
            devices_by_id,
            args,
            value=int(brightness),
        )

    if call.name == "HassClimateSetTemperature":
        payload: dict[str, Any] = {
            "outcome": "action",
            "intent": user_text,
            "domain": "climate",
        }
        targets = _semantic_targets(args, devices_by_id)
        if targets:
            payload["targets"] = targets
        temperature = args.get("temperature")
        if temperature is not None:
            payload["value"] = float(temperature)
        hvac_mode = args.get("hvac_mode")
        if hvac_mode is not None:
            mode = _climate_mode(str(hvac_mode))
            if mode is None:
                return None
            payload["mode"] = mode
        if "value" not in payload and "mode" not in payload:
            return None
        return payload

    return None


def _map_domain_action(
    *,
    domain: str,
    action: str,
    arguments: dict[str, Any],
    user_text: str,
    devices_by_id: dict[str, ParsedDevice],
) -> dict[str, Any] | None:
    if action in {"turn_on", "open_cover", "lock", "unlock"}:
        state = {
            "turn_on": "on",
            "open_cover": "open",
            "lock": "lock",
            "unlock": "unlock",
        }[action]
        if domain == "lock" and action == "turn_on":
            state = "lock"
        if domain == "lock" and action == "turn_off":
            state = "unlock"
        if domain == "cover" and action == "turn_on":
            state = "open"
        if domain == "cover" and action == "turn_off":
            state = "close"
        if action == "turn_on" and "brightness" in arguments:
            if "color" in arguments:
                return None
            return _action(
                user_text,
                domain,
                devices_by_id,
                arguments,
                value=int(arguments["brightness"]),
            )
        return _action(user_text, domain, devices_by_id, arguments, state=state)

    if action in {"turn_off", "close_cover"}:
        state = "off" if action == "turn_off" else "close"
        return _action(user_text, domain, devices_by_id, arguments, state=state)

    if action == "toggle":
        return _action(user_text, domain, devices_by_id, arguments, state="toggle")

    if action in {"set_temperature", "set_hvac_mode", "set_fan_mode", "set_preset_mode"}:
        payload: dict[str, Any] = {
            "outcome": "action",
            "intent": user_text,
            "domain": "climate",
        }
        targets = _semantic_targets(arguments, devices_by_id)
        if targets:
            payload["targets"] = targets
        if "temperature" in arguments:
            payload["value"] = float(arguments["temperature"])
        hvac_mode = arguments.get("hvac_mode")
        if hvac_mode is not None:
            mode = _climate_mode(str(hvac_mode))
            if mode is None:
                return None
            payload["mode"] = mode
        if "value" not in payload and "mode" not in payload:
            return None
        return payload

    if action in {"set_humidity", "set_humidifier_mode", "set_cover_position"}:
        return None

    return None


def _action(
    user_text: str,
    domain: str,
    devices_by_id: dict[str, ParsedDevice],
    arguments: dict[str, Any],
    *,
    state: str | None = None,
    value: float | int | None = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "outcome": "action",
        "intent": user_text,
        "domain": domain,
    }
    targets = _semantic_targets(arguments, devices_by_id)
    if targets:
        payload["targets"] = targets
    if state is not None:
        payload["state"] = state
    if value is not None:
        payload["value"] = value
    return payload


def _argument_domain(args: dict[str, Any], devices_by_id: dict[str, ParsedDevice]) -> str | None:
    entity_id = _entity_reference(args)
    if entity_id is not None:
        device = devices_by_id.get(entity_id)
        if device is None:
            domain = entity_id.split(".", 1)[0]
            return SERVICE_DOMAIN_ALIASES.get(domain, domain)
        return SERVICE_DOMAIN_ALIASES.get(device.domain, device.domain)

    domain = args.get("domain")
    if isinstance(domain, list) and domain:
        first = domain[0]
        if isinstance(first, str):
            return SERVICE_DOMAIN_ALIASES.get(first, first)
    if isinstance(domain, str):
        return SERVICE_DOMAIN_ALIASES.get(domain, domain)
    return None


def _entity_reference(args: dict[str, Any]) -> str | None:
    for key in ("entity_id", "name"):
        value = args.get(key)
        if isinstance(value, str) and ENTITY_ID_PATTERN.match(value):
            return value
    return None


def _semantic_targets(
    args: dict[str, Any],
    devices_by_id: dict[str, ParsedDevice],
) -> list[str] | None:
    entity_id = _entity_reference(args)
    if entity_id is not None:
        device = devices_by_id.get(entity_id)
        if device is None:
            return None
        return [_semantic_label(device.friendly_name)]

    name = args.get("name")
    if isinstance(name, str):
        if is_entity_id(name):
            device = devices_by_id.get(name)
            if device is None:
                return None
            return [_semantic_label(device.friendly_name)]
        if ENTITY_ID_PATTERN.match(name):
            return None
        return [_semantic_label(name)]

    return None


def _semantic_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _climate_mode(raw: str) -> str | None:
    normalized = raw.lower()
    if normalized in {"heat", "cool", "auto", "off"}:
        return normalized
    if normalized in {"heat_cool"}:
        return ClimateMode.AUTO.value
    return None


def control_plan_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, sort_keys=True, separators=(",", ":"))
