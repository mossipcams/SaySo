from __future__ import annotations

import inspect
from collections import Counter
from types import FrameType
from unittest.mock import AsyncMock, patch

from homeassistant.components.homeassistant import exposed_entities
from homeassistant.core import HomeAssistant, State, StateMachine

from custom_components.sayso.client import ChatCompletionResult, LlamaCppClient
from tests.test_conversation import (
    _converse,
    _create_entry,
    assist_light,
    mock_llama_client,
)

SAYSO_PREFIX = "custom_components.sayso"
EXPOSED_MODULE = exposed_entities.__name__


def _caller_module(frame: FrameType | None) -> str:
    while frame is not None and frame.f_globals["__name__"] == EXPOSED_MODULE:
        frame = frame.f_back
    assert frame is not None
    return str(frame.f_globals["__name__"])


async def test_turn_scans_exposed_states_once(
    hass: HomeAssistant,
    mock_llama_client: None,
    assist_light: None,
) -> None:
    entry = await _create_entry(hass)
    state_ids = sorted(state.entity_id for state in hass.states.async_all())
    original_all = StateMachine.async_all
    original_should_expose = exposed_entities.ExposedEntities.async_should_expose
    scans: list[str] = []
    checks: list[tuple[str, str]] = []

    def record_all(self: StateMachine, *args: object, **kwargs: object) -> list[State]:
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        scans.append(_caller_module(frame.f_back))
        return original_all(self, *args, **kwargs)

    def record_should_expose(
        self: exposed_entities.ExposedEntities, assistant: str, entity_id: str
    ) -> bool:
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        checks.append((_caller_module(frame.f_back), entity_id))
        return original_should_expose(self, assistant, entity_id)

    with (
        patch.object(StateMachine, "async_all", new=record_all),
        patch.object(
            exposed_entities.ExposedEntities,
            "async_should_expose",
            new=record_should_expose,
        ),
        patch.object(
            LlamaCppClient,
            "chat_completion",
            new=AsyncMock(
                return_value=ChatCompletionResult(content="Done.", tool_calls=[])
            ),
        ),
    ):
        await _converse(hass, entry, "Is the light on?")

    sayso_scans = [module for module in scans if module.startswith(SAYSO_PREFIX)]
    assert len(sayso_scans) == 1
    sayso_checks = Counter(
        entity_id for module, entity_id in checks if module.startswith(SAYSO_PREFIX)
    )
    assert sorted(sayso_checks) == state_ids
    assert set(sayso_checks.values()) == {1}
