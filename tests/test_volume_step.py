from __future__ import annotations

import sys
from typing import Any

import pytest
from homeassistant.components.media_player import (
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
)
from homeassistant.components.media_player.const import INTENT_SET_VOLUME_RELATIVE
from homeassistant.components.media_player.intent import MediaSetVolumeRelativeHandler
from homeassistant.const import STATE_PLAYING
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import setup_test_component_platform

from custom_components.sayso.volume_step import StepVolumeRelativeHandler


class _FakePlayer(MediaPlayerEntity):

    _attr_has_entity_name = False
    _attr_state = STATE_PLAYING

    def __init__(
        self,
        name: str,
        features: MediaPlayerEntityFeature,
        volume_level: float | None = None,
    ) -> None:
        self._attr_name = name
        self._attr_supported_features = features
        self._attr_volume_level = volume_level
        self.calls: list[tuple[Any, ...]] = []

    async def async_volume_up(self) -> None:
        self.calls.append(("up",))

    async def async_volume_down(self) -> None:
        self.calls.append(("down",))

    async def async_set_volume_level(self, volume: float) -> None:
        self.calls.append(("set", volume))


STEP_ONLY = MediaPlayerEntityFeature.VOLUME_STEP
VOLUME_SET = MediaPlayerEntityFeature.VOLUME_SET


async def _async_setup_players(hass: HomeAssistant, players: list[_FakePlayer]) -> None:
    setup_test_component_platform(hass, "media_player", players)
    assert await async_setup_component(
        hass, "media_player", {"media_player": {"platform": "test"}}
    )
    await hass.async_block_till_done()
    assert await async_setup_component(hass, "sayso", {})
    await hass.async_block_till_done()


async def _async_handle(hass: HomeAssistant, **slots: Any) -> intent.IntentResponse:
    return await intent.async_handle(
        hass,
        "sayso",
        INTENT_SET_VOLUME_RELATIVE,
        {key: {"value": value} for key, value in slots.items()},
    )


async def test_step_only_player_volume_down(hass: HomeAssistant) -> None:
    tv = _FakePlayer("TV", STEP_ONLY)
    await _async_setup_players(hass, [tv])

    response = await _async_handle(hass, name="TV", volume_step="down")

    assert tv.calls == [("down",)]
    assert response.response_type is intent.IntentResponseType.ACTION_DONE


async def test_step_only_player_volume_up(hass: HomeAssistant) -> None:
    tv = _FakePlayer("TV", STEP_ONLY)
    await _async_setup_players(hass, [tv])

    await _async_handle(hass, name="TV", volume_step="up")

    assert tv.calls == [("up",)]


async def test_step_only_player_numeric_negative_is_one_down_step(
    hass: HomeAssistant,
) -> None:
    tv = _FakePlayer("TV", STEP_ONLY)
    await _async_setup_players(hass, [tv])

    await _async_handle(hass, name="TV", volume_step=-10)

    assert tv.calls == [("down",)]


async def test_volume_set_player_numeric_still_uses_ha_path(
    hass: HomeAssistant,
) -> None:
    speaker = _FakePlayer("Speaker", VOLUME_SET, volume_level=0.5)
    await _async_setup_players(hass, [speaker])

    await _async_handle(hass, name="Speaker", volume_step=10)

    assert len(speaker.calls) == 1
    assert speaker.calls[0][0] == "set"
    assert speaker.calls[0][1] == pytest.approx(0.6)


async def test_no_step_capable_target_reraises_original_error(
    hass: HomeAssistant,
) -> None:
    lamp = _FakePlayer("Lamp", MediaPlayerEntityFeature(0))
    await _async_setup_players(hass, [lamp])

    with pytest.raises(intent.MatchFailedError) as err:
        await _async_handle(hass, name="Lamp", volume_step="down")

    assert err.value.result.no_match_reason is intent.MatchFailedReason.FEATURE
    assert lamp.calls == []


async def test_registered_handler_keeps_ha_tool_schema(hass: HomeAssistant) -> None:
    await _async_setup_players(hass, [])

    handler = hass.data[intent.DATA_KEY][INTENT_SET_VOLUME_RELATIVE]

    assert isinstance(handler, StepVolumeRelativeHandler)
    assert isinstance(handler, MediaSetVolumeRelativeHandler)
    assert handler.slot_schema is MediaSetVolumeRelativeHandler.slot_schema


async def test_volume_step_import_failure_logs_and_keeps_ha_handler(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setitem(sys.modules, "custom_components.sayso.volume_step", None)

    await _async_setup_players(hass, [])

    assert any(
        record.levelname == "ERROR"
        and "Volume step fallback disabled" in record.getMessage()
        for record in caplog.records
    )
    handler = hass.data[intent.DATA_KEY][INTENT_SET_VOLUME_RELATIVE]
    assert type(handler) is MediaSetVolumeRelativeHandler
