from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.sayso.client import LlamaCppClient
from custom_components.sayso.conversation import _Turn
from custom_components.sayso.ducking import (
    MediaDucker,
    async_setup_ducking,
    async_wait_all_restores,
)
from tests.test_config_flow import MODEL_ID
from tests.test_init import _create_entry


@pytest.fixture
def mock_llama_client() -> Any:
    with (
        patch.object(
            LlamaCppClient,
            "list_models",
            new=AsyncMock(return_value=[MODEL_ID]),
        ),
        patch.object(
            LlamaCppClient,
            "validate_model",
            new=AsyncMock(return_value=None),
        ),
    ):
        yield


async def _no_results(content: Any) -> AsyncIterator[Any]:
    for result in ():
        yield result


def _turn(hass: HomeAssistant, ducker: MediaDucker | None) -> _Turn:
    return _Turn(
        entry_id="entry",
        agent_id="agent",
        runtime=None,
        user_input=None,
        chat_log=SimpleNamespace(
            hass=hass,
            async_add_assistant_content=_no_results,
        ),
        trace=None,
        ducker=ducker,
    )


async def test_main_entry_holds_ducker_on_runtime_data(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_entry(hass)
    await hass.async_block_till_done()

    assert isinstance(entry.runtime_data.ducker, MediaDucker)


async def test_unload_tears_down_entry_ducker(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    teardown = AsyncMock()
    with patch.object(MediaDucker, "async_teardown", teardown):
        entry = await _create_entry(hass)
        await hass.async_block_till_done()
        teardown.assert_not_awaited()

        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

    teardown.assert_awaited_once()


async def test_turn_waits_on_restores_when_ducker_set(hass: HomeAssistant) -> None:
    ducker = MediaDucker(hass, 0.02)
    span = SimpleNamespace(metadata={})
    with patch.object(ducker, "async_wait_restores", AsyncMock()) as wait:
        result = await _turn(hass, ducker).execute([], {}, span)

    wait.assert_awaited_once()
    assert result == (False, None, None)


async def test_turn_without_ducker_does_not_wait(hass: HomeAssistant) -> None:
    span = SimpleNamespace(metadata={})

    result = await _turn(hass, None).execute([], {}, span)

    assert result == (False, None, None)


async def test_torn_down_ducker_is_not_waited_on(hass: HomeAssistant) -> None:
    ducker = await async_setup_ducking(hass, 0.02)
    await ducker.async_teardown()

    with patch.object(ducker, "async_wait_restores", AsyncMock()) as wait:
        await async_wait_all_restores(hass)

    wait.assert_not_awaited()


async def test_every_active_ducker_is_waited_on(hass: HomeAssistant) -> None:
    first = await async_setup_ducking(hass, 0.02)
    second = await async_setup_ducking(hass, 0.03)

    with (
        patch.object(first, "async_wait_restores", AsyncMock()) as first_wait,
        patch.object(second, "async_wait_restores", AsyncMock()) as second_wait,
    ):
        await async_wait_all_restores(hass)

    first_wait.assert_awaited_once()
    second_wait.assert_awaited_once()
