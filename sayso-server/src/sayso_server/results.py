"""Typed action results for the execution orchestrator pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ActionResultStatus(StrEnum):
    """Lifecycle status emitted by the Home Assistant integration."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    COMPLETED = "completed"


class ExecutionCategory(StrEnum):
    """Exact success or failure category for a completed orchestrator run."""

    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    NO_ACTION = "no_action"
    INCOMPLETE_RESULTS = "incomplete_results"
    MISORDERED_RESULTS = "misordered_results"


@dataclass(frozen=True)
class ActionResult:
    request_id: str
    status: ActionResultStatus
    reason: str | None = None


@dataclass(frozen=True)
class ExecutionOutcome:
    category: ExecutionCategory
    plan: object
    request_id: str | None = None
    results: tuple[ActionResult, ...] = ()
    reason: str | None = None
