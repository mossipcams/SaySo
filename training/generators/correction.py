
from __future__ import annotations

import copy
import re
from typing import Any

LOCKED_SCHEMA_FINGERPRINT = (
    "sha256:61bdafb650a1a1c54ea66ea44fdb6e30cfa89e67e832b3e1193807df34f31423"
)


def entity_name_contains_area(entity: dict[str, Any]) -> bool:
    area = (entity.get("area") or "").strip()
    name = (entity.get("name") or "").strip()
    if not area or not name:
        return False
    area_cf = area.casefold()
    name_cf = name.casefold()
    if name_cf == area_cf or name_cf.startswith(f"{area_cf} "):
        return True
    return bool(re.search(rf"\b{re.escape(area_cf)}\b", name_cf))


def format_synthetic_validation_error(
    *,
    code: str,
    message: str,
    allowed_tools: list[str],
    fingerprint: str = LOCKED_SCHEMA_FINGERPRINT,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "allowed_tools": allowed_tools,
            "schema_fingerprint": fingerprint,
        }
    }


def wrong_name_tool_call(correct_call: dict[str, Any], wrong_name: str) -> dict[str, Any]:
    call = copy.deepcopy(correct_call)
    arguments = dict(call.get("arguments") or {})
    arguments["name"] = wrong_name
    call["arguments"] = arguments
    return call


def validation_error_message(wrong_name: str) -> str:
    return f"No matching entity for name '{wrong_name}'"
