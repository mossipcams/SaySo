"""Utterance template registry."""

from __future__ import annotations

_TEMPLATES = {
    "turn_on": "Turn on {target}",
    "turn_off": "Turn off {target}",
    "lock": "Lock {target}",
    "unlock": "Unlock {target}",
    "open": "Open {target}",
    "close": "Close {target}",
    "brightness": "Set {target} to {value} percent",
    "status": "What is the status of {target}",
    "cancel_timers": "Cancel all timers",
}


def get_template(operation: str) -> str:
    return _TEMPLATES.get(operation, "Control {target}")
