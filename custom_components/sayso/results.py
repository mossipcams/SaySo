"""Typed action result payloads for SaySo action_request handling."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ActionResultStatus(StrEnum):
    """Lifecycle status for an inbound action_request."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    COMPLETED = "completed"


def build_action_result_payload(
    *,
    request_id: str,
    status: ActionResultStatus,
    reason: str | None = None,
) -> dict[str, Any]:
    """Return the action_result message payload for a correlated request."""

    payload: dict[str, Any] = {
        "request_id": request_id,
        "status": status,
    }
    if reason is not None:
        payload["reason"] = reason
    return payload
