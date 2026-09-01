"""Await Home Assistant state feedback after an accepted action request."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum

from homeassistant.core import Event, HomeAssistant, callback


class StateVerificationOutcome(StrEnum):
    """Result of waiting for entity state feedback."""

    CHANGED = "changed"
    UNCHANGED = "unchanged"
    TIMEOUT = "timeout"


def baseline_state(hass: HomeAssistant, entity_id: str) -> str | None:
    """Return the entity's current state string, if any."""

    state = hass.states.get(entity_id)
    if state is None:
        return None
    return state.state


async def verify_state_after_action(
    hass: HomeAssistant,
    entity_id: str,
    *,
    baseline: str | None,
    timeout: float,
    action: Callable[[], Awaitable[None]],
) -> StateVerificationOutcome:
    """Run an action and await relevant state feedback."""

    if timeout <= 0:
        await action()
        return _compare_state(hass, entity_id, baseline)

    loop = asyncio.get_running_loop()
    future: asyncio.Future[StateVerificationOutcome] = loop.create_future()

    @callback
    def _on_state_changed(event: Event) -> None:
        if event.data.get("entity_id") != entity_id:
            return
        new_state = event.data.get("new_state")
        if new_state is None:
            return
        outcome = (
            StateVerificationOutcome.CHANGED
            if new_state.state != baseline
            else StateVerificationOutcome.UNCHANGED
        )
        if not future.done():
            future.set_result(outcome)

    unsub = hass.bus.async_listen("state_changed", _on_state_changed)
    try:
        await action()
        if future.done():
            return future.result()
        return await asyncio.wait_for(future, timeout=timeout)
    except TimeoutError:
        return StateVerificationOutcome.TIMEOUT
    finally:
        unsub()


def _compare_state(
    hass: HomeAssistant,
    entity_id: str,
    baseline: str | None,
) -> StateVerificationOutcome:
    current = baseline_state(hass, entity_id)
    if current != baseline:
        return StateVerificationOutcome.CHANGED
    return StateVerificationOutcome.UNCHANGED
