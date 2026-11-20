from __future__ import annotations

import asyncio
from typing import Any

from homeassistant.components.media_player import (
    MediaPlayerEntity,
    MediaPlayerEntityFeature,
)
from homeassistant.components.media_player.const import (
    INTENT_SET_VOLUME,
    INTENT_SET_VOLUME_RELATIVE,
)
from homeassistant.const import STATE_PLAYING
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import (
    async_mock_service,
    setup_test_component_platform,
)

from custom_components.sayso.ducking import async_setup_ducking
from custom_components.sayso.volume_step import async_register_volume_step


class _FakePlayer(MediaPlayerEntity):
    _attr_has_entity_name = False
    _attr_state = STATE_PLAYING

    def __init__(self, name: str) -> None:
        self._attr_name = name
        self._attr_unique_id = name
        self._attr_supported_features = (
            MediaPlayerEntityFeature.VOLUME_SET | MediaPlayerEntityFeature.VOLUME_STEP
        )
        self._attr_volume_level = 0.5
        self.calls: list[tuple[Any, ...]] = []
        self.events: list[str] = []

    async def async_volume_up(self) -> None:
        self.calls.append(("up",))
        self.events.append("up")

    async def async_set_volume_level(self, volume: float) -> None:
        self.calls.append(("set", volume))


async def _async_setup(hass: HomeAssistant) -> _FakePlayer:
    area_id = ar.async_get(hass).async_create("Kitchen").id
    registry = er.async_get(hass)
    satellite = registry.async_get_or_create(
        "assist_satellite", "test", "Satellite", suggested_object_id="kitchen_sat"
    )
    registry.async_update_entity(satellite.entity_id, area_id=area_id)
    hass.states.async_set(satellite.entity_id, "idle")

    tv = _FakePlayer("TV")
    setup_test_component_platform(hass, "media_player", [tv])
    assert await async_setup_component(
        hass, "media_player", {"media_player": {"platform": "test"}}
    )
    await hass.async_block_till_done()

    tv_entry = registry.async_get("media_player.tv")
    registry.async_update_entity(tv_entry.entity_id, area_id=area_id)

    async_register_volume_step(hass)
    await hass.async_block_till_done()
    return tv


def _gate_service(
    hass: HomeAssistant, events: list[str], gate: asyncio.Event
) -> asyncio.Event:
    started = asyncio.Event()

    async def _volume_set(call) -> None:
        level = call.data.get("volume_level")
        if level == 0.5:
            started.set()
            await gate.wait()
            events.append("restore")
        elif level == 0.02:
            events.append("duck:0.02")
        else:
            events.append(f"set:{level}")

    hass.services.async_register("media_player", "volume_set", _volume_set)
    return started


async def test_relative_volume_waits_for_pending_restore(hass: HomeAssistant) -> None:
    tv = await _async_setup(hass)
    gate = asyncio.Event()
    events = tv.events
    started = _gate_service(hass, events, gate)
    ducker = await async_setup_ducking(hass, 0.02)

    hass.states.async_set("assist_satellite.kitchen_sat", "listening")
    await hass.async_block_till_done()
    assert events == ["duck:0.02"]

    hass.states.async_set("assist_satellite.kitchen_sat", "idle")
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)

        task = asyncio.create_task(
            intent.async_handle(
                hass,
                "sayso",
                INTENT_SET_VOLUME_RELATIVE,
                {"name": {"value": "TV"}, "volume_step": {"value": "up"}},
            )
        )
        for _ in range(10):
            await asyncio.sleep(0)
        assert not task.done()
        assert tv.calls == []
        assert events == ["duck:0.02"]

        gate.set()
        response = await task
        assert response.response_type is intent.IntentResponseType.ACTION_DONE
    finally:
        gate.set()
        await hass.async_block_till_done()

    assert tv.calls == [("up",)]
    assert events == ["duck:0.02", "restore", "up"]
    assert not ducker._restore_tasks


async def test_absolute_volume_waits_for_pending_restore(hass: HomeAssistant) -> None:
    await _async_setup(hass)
    registered = hass.data[intent.DATA_KEY][INTENT_SET_VOLUME]
    assert type(registered).__module__ == "custom_components.sayso.volume_step"

    gate = asyncio.Event()
    events: list[str] = []
    started = _gate_service(hass, events, gate)
    ducker = await async_setup_ducking(hass, 0.02)

    hass.states.async_set("assist_satellite.kitchen_sat", "listening")
    await hass.async_block_till_done()
    assert events == ["duck:0.02"]

    hass.states.async_set("assist_satellite.kitchen_sat", "idle")
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)

        task = asyncio.create_task(
            intent.async_handle(
                hass,
                "sayso",
                INTENT_SET_VOLUME,
                {"name": {"value": "TV"}, "volume_level": {"value": 30}},
            )
        )
        for _ in range(10):
            await asyncio.sleep(0)
        assert not task.done()
        assert events == ["duck:0.02"]

        gate.set()
        response = await task
        assert response.response_type is intent.IntentResponseType.ACTION_DONE
    finally:
        gate.set()
        await hass.async_block_till_done()

    assert events == ["duck:0.02", "restore", "set:0.3"]
    assert not ducker._restore_tasks


async def test_relative_volume_without_ducker(hass: HomeAssistant) -> None:
    tv = await _async_setup(hass)

    response = await intent.async_handle(
        hass,
        "sayso",
        INTENT_SET_VOLUME_RELATIVE,
        {"name": {"value": "TV"}, "volume_step": {"value": "up"}},
    )

    assert response.response_type is intent.IntentResponseType.ACTION_DONE
    assert tv.calls == [("up",)]


async def test_absolute_volume_without_ducker(hass: HomeAssistant) -> None:
    await _async_setup(hass)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")

    response = await intent.async_handle(
        hass,
        "sayso",
        INTENT_SET_VOLUME,
        {"name": {"value": "TV"}, "volume_level": {"value": 40}},
    )

    assert response.response_type is intent.IntentResponseType.ACTION_DONE
    assert [dict(call.data) for call in volume_calls] == [
        {"entity_id": "media_player.tv", "volume_level": 0.4}
    ]
