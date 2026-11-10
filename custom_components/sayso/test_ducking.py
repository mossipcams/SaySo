from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.components.media_player import (
    ATTR_MEDIA_VOLUME_LEVEL,
    ATTR_MEDIA_VOLUME_MUTED,
    MediaPlayerEntityFeature,
)
from homeassistant.const import ATTR_SUPPORTED_FEATURES, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from freezegun.api import FrozenDateTimeFactory
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
    async_mock_service,
)

from custom_components.sayso.const import DOMAIN, DUCK_TIMEOUT
from custom_components.sayso.ducking import (
    DUCKER_KEY,
    ENTRY_COUNT_KEY,
    async_release_ducking,
    async_setup_ducking,
    async_wait_for_restores,
)

VOLUME_ATTRS = {
    ATTR_MEDIA_VOLUME_LEVEL: 0.5,
    ATTR_SUPPORTED_FEATURES: MediaPlayerEntityFeature.VOLUME_SET,
}
MUTE_ATTRS = {
    ATTR_MEDIA_VOLUME_MUTED: False,
    ATTR_SUPPORTED_FEATURES: MediaPlayerEntityFeature.VOLUME_MUTE,
}
PAUSE_ATTRS = {ATTR_SUPPORTED_FEATURES: MediaPlayerEntityFeature.PAUSE}
PAUSE_MUTE_ATTRS = {
    ATTR_MEDIA_VOLUME_MUTED: False,
    ATTR_SUPPORTED_FEATURES: (
        MediaPlayerEntityFeature.PAUSE | MediaPlayerEntityFeature.VOLUME_MUTE
    ),
}


def _area(hass: HomeAssistant, name: str) -> str:
    return ar.async_get(hass).async_create(name).id


def _device(hass: HomeAssistant, identifier: str, area_id: str | None = None) -> str:
    config_entry = MockConfigEntry(domain="test")
    config_entry.add_to_hass(hass)
    registry = dr.async_get(hass)
    device = registry.async_get_or_create(
        config_entry_id=config_entry.entry_id,
        identifiers={("test", identifier)},
        name=identifier,
    )
    if area_id is not None:
        registry.async_update_device(device.id, area_id=area_id)
    return device.id


def _register(
    hass: HomeAssistant,
    domain: str,
    name: str,
    device_id: str | None = None,
    area_id: str | None = None,
) -> str:
    registry = er.async_get(hass)
    entry = registry.async_get_or_create(
        domain,
        "test",
        name,
        suggested_object_id=name,
        device_id=device_id,
    )
    if area_id is not None:
        registry.async_update_entity(entry.entity_id, area_id=area_id)
    return entry.entity_id


async def _set(
    hass: HomeAssistant,
    entity_id: str,
    state: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    hass.states.async_set(entity_id, state, attributes or {})
    await hass.async_block_till_done()


def _data(calls: list[Any]) -> list[dict[str, Any]]:
    return [dict(call.data) for call in calls]


async def test_duck_and_restore(hass: HomeAssistant) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert _data(volume_calls) == [{"entity_id": tv, "volume_level": 0.02}]

    await _set(hass, satellite, "idle")
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}
    assert len(volume_calls) == 2


async def test_second_wake_while_ducked_keeps_true_original(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    first = _register(hass, "assist_satellite", "first_sat", area_id=kitchen)
    second = _register(hass, "assist_satellite", "second_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, first, "idle")
    await _set(hass, second, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, first, "listening")
    await _set(hass, tv, "playing", {**VOLUME_ATTRS, ATTR_MEDIA_VOLUME_LEVEL: 0.02})
    await _set(hass, second, "listening")
    await _set(hass, first, "idle")
    await _set(hass, second, "idle")

    assert _data(volume_calls) == [
        {"entity_id": tv, "volume_level": 0.02},
        {"entity_id": tv, "volume_level": 0.5},
    ]


async def test_rewake_while_tv_reports_ducked_level_restores_original(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    await _set(hass, satellite, "idle")
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}

    await _set(hass, tv, "playing", {**VOLUME_ATTRS, ATTR_MEDIA_VOLUME_LEVEL: 0.02})
    await _set(hass, satellite, "listening")
    await _set(hass, satellite, "idle")

    assert _data(volume_calls) == [
        {"entity_id": tv, "volume_level": 0.02},
        {"entity_id": tv, "volume_level": 0.5},
        {"entity_id": tv, "volume_level": 0.02},
        {"entity_id": tv, "volume_level": 0.5},
    ]


async def test_stale_restore_memory_is_not_used(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    await _set(hass, satellite, "idle")
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}

    freezer.tick(timedelta(seconds=DUCK_TIMEOUT + 1))
    await _set(hass, tv, "playing", {**VOLUME_ATTRS, ATTR_MEDIA_VOLUME_LEVEL: 0.02})
    await _set(hass, satellite, "listening")
    await _set(hass, satellite, "idle")

    assert len(volume_calls) == 2
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}


async def test_restore_memory_is_single_use_and_rearmed_by_each_restore(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", {**VOLUME_ATTRS, ATTR_MEDIA_VOLUME_LEVEL: 0.02})
    await _set(hass, satellite, "listening")
    await _set(hass, satellite, "idle")
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}

    await _set(hass, tv, "playing", {**VOLUME_ATTRS, ATTR_MEDIA_VOLUME_LEVEL: 0.02})
    await _set(hass, satellite, "listening")
    await _set(hass, satellite, "idle")

    assert _data(volume_calls) == [
        {"entity_id": tv, "volume_level": 0.02},
        {"entity_id": tv, "volume_level": 0.5},
        {"entity_id": tv, "volume_level": 0.02},
        {"entity_id": tv, "volume_level": 0.5},
        {"entity_id": tv, "volume_level": 0.02},
        {"entity_id": tv, "volume_level": 0.5},
    ]


async def test_restore_waits_for_both_satellites(hass: HomeAssistant) -> None:
    kitchen = _area(hass, "Kitchen")
    first = _register(hass, "assist_satellite", "first_sat", area_id=kitchen)
    second = _register(hass, "assist_satellite", "second_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, first, "idle")
    await _set(hass, second, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, first, "listening")
    await _set(hass, second, "listening")
    await _set(hass, first, "listening", {"source": "again"})
    assert len(volume_calls) == 1

    await _set(hass, first, "idle")
    assert len(volume_calls) == 1

    await _set(hass, second, "idle")
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}
    assert len(volume_calls) == 2


async def test_timeout_restores_stuck_volume(hass: HomeAssistant) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(seconds=DUCK_TIMEOUT + 1)
    )
    await hass.async_block_till_done()
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}

    await _set(hass, satellite, "listening", {"source": "still"})
    await _set(hass, satellite, "idle")
    assert len(volume_calls) == 2


async def test_teardown_restores_and_stops_listening(hass: HomeAssistant) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    ducker = await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    await ducker.async_teardown()
    await hass.async_block_till_done()
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}

    await _set(hass, satellite, "idle")
    await _set(hass, satellite, "listening")
    assert len(volume_calls) == 2


async def test_mute_fallback_for_players_without_volume_level(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    roku = _register(hass, "media_player", "kitchen_roku", area_id=kitchen)
    muted = _register(hass, "media_player", "kitchen_muted", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, roku, "playing", MUTE_ATTRS)
    await _set(hass, muted, "playing", {**MUTE_ATTRS, ATTR_MEDIA_VOLUME_MUTED: True})
    mute_calls = async_mock_service(hass, "media_player", "volume_mute")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert _data(mute_calls) == [{"entity_id": roku, ATTR_MEDIA_VOLUME_MUTED: True}]

    await _set(hass, satellite, "idle")
    assert _data(mute_calls)[-1] == {"entity_id": roku, ATTR_MEDIA_VOLUME_MUTED: False}
    assert len(mute_calls) == 2


async def test_pause_fallback_for_playing_players_without_volume_level(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    roku = _register(hass, "media_player", "kitchen_roku", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, roku, "playing", PAUSE_ATTRS)
    pause_calls = async_mock_service(hass, "media_player", "media_pause")
    play_calls = async_mock_service(hass, "media_player", "media_play")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert _data(pause_calls) == [{"entity_id": roku}]
    assert play_calls == []

    await _set(hass, satellite, "idle")
    assert _data(play_calls) == [{"entity_id": roku}]
    assert len(pause_calls) == 1


async def test_pause_capable_player_in_on_state_is_muted_not_paused(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    player = _register(hass, "media_player", "kitchen_player", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, player, "on", PAUSE_MUTE_ATTRS)
    pause_calls = async_mock_service(hass, "media_player", "media_pause")
    mute_calls = async_mock_service(hass, "media_player", "volume_mute")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert pause_calls == []
    assert _data(mute_calls) == [{"entity_id": player, ATTR_MEDIA_VOLUME_MUTED: True}]

    await _set(hass, satellite, "idle")
    assert _data(mute_calls)[-1] == {
        "entity_id": player,
        ATTR_MEDIA_VOLUME_MUTED: False,
    }
    assert len(mute_calls) == 2


async def test_volume_level_player_with_pause_support_uses_volume_set(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(
        hass,
        tv,
        "playing",
        {
            **VOLUME_ATTRS,
            ATTR_SUPPORTED_FEATURES: (
                MediaPlayerEntityFeature.VOLUME_SET | MediaPlayerEntityFeature.PAUSE
            ),
        },
    )
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    pause_calls = async_mock_service(hass, "media_player", "media_pause")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert _data(volume_calls) == [{"entity_id": tv, "volume_level": 0.02}]
    assert pause_calls == []

    await _set(hass, satellite, "idle")
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}
    assert pause_calls == []


async def test_satellite_own_device_player_excluded(hass: HomeAssistant) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite_device = _device(hass, "kitchen_speaker", area_id=kitchen)
    satellite = _register(
        hass, "assist_satellite", "kitchen_sat", device_id=satellite_device
    )
    own_player = _register(
        hass, "media_player", "speaker_player", device_id=satellite_device
    )
    await _set(hass, satellite, "idle")
    await _set(hass, own_player, "playing", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert volume_calls == []


async def test_players_in_other_areas_untouched(hass: HomeAssistant) -> None:
    kitchen = _area(hass, "Kitchen")
    office = _area(hass, "Office")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    office_tv = _register(hass, "media_player", "office_tv", area_id=office)
    off_tv = _register(hass, "media_player", "kitchen_off", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, office_tv, "playing", VOLUME_ATTRS)
    await _set(hass, off_tv, "off", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert volume_calls == []


async def test_device_area_fallback_and_later_added_satellite(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    speaker_device = _device(hass, "kitchen_hub", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", device_id=speaker_device)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    ducker = await async_setup_ducking(hass, 0.02)

    satellite = _register(hass, "assist_satellite", "late_sat", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, satellite, "listening")
    assert _data(volume_calls) == [{"entity_id": tv, "volume_level": 0.02}]

    await ducker.async_teardown()
    await hass.async_block_till_done()
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}


async def test_on_state_tv_with_stale_volume_level_is_muted_not_set(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(
        hass,
        tv,
        "on",
        {
            ATTR_MEDIA_VOLUME_LEVEL: 0.19,
            ATTR_MEDIA_VOLUME_MUTED: False,
            ATTR_SUPPORTED_FEATURES: (
                MediaPlayerEntityFeature.VOLUME_SET
                | MediaPlayerEntityFeature.VOLUME_MUTE
            ),
        },
    )
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    mute_calls = async_mock_service(hass, "media_player", "volume_mute")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert volume_calls == []
    assert _data(mute_calls) == [{"entity_id": tv, ATTR_MEDIA_VOLUME_MUTED: True}]

    await _set(hass, satellite, "idle")
    assert _data(mute_calls)[-1] == {"entity_id": tv, ATTR_MEDIA_VOLUME_MUTED: False}
    assert volume_calls == []
    assert len(mute_calls) == 2


@pytest.mark.parametrize("level", [0.01, 0.02])
async def test_quiet_player_is_not_touched(hass: HomeAssistant, level: float) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(
        hass,
        tv,
        "playing",
        {
            ATTR_MEDIA_VOLUME_LEVEL: level,
            ATTR_SUPPORTED_FEATURES: (
                MediaPlayerEntityFeature.VOLUME_SET | MediaPlayerEntityFeature.PAUSE
            ),
        },
    )
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    pause_calls = async_mock_service(hass, "media_player", "media_pause")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert volume_calls == []
    assert pause_calls == []

    await _set(hass, satellite, "idle")
    assert volume_calls == []


async def test_player_just_above_duck_level_is_ducked(hass: HomeAssistant) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", {**VOLUME_ATTRS, ATTR_MEDIA_VOLUME_LEVEL: 0.021})
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    await _set(hass, satellite, "idle")
    assert _data(volume_calls) == [
        {"entity_id": tv, "volume_level": 0.02},
        {"entity_id": tv, "volume_level": 0.021},
    ]


async def test_volume_level_without_set_support_is_paused(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    player = _register(hass, "media_player", "kitchen_player", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(
        hass,
        player,
        "playing",
        {
            ATTR_MEDIA_VOLUME_LEVEL: 0.5,
            ATTR_SUPPORTED_FEATURES: MediaPlayerEntityFeature.PAUSE,
        },
    )
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    pause_calls = async_mock_service(hass, "media_player", "media_pause")
    play_calls = async_mock_service(hass, "media_player", "media_play")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert volume_calls == []
    assert _data(pause_calls) == [{"entity_id": player}]

    await _set(hass, satellite, "idle")
    assert _data(play_calls) == [{"entity_id": player}]
    assert len(pause_calls) == 1
    assert volume_calls == []


async def test_two_entries_share_one_ducker(hass: HomeAssistant) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")

    shared_data = hass.data.setdefault(DOMAIN, {})
    shared_data["unrelated"] = object()
    first = await async_setup_ducking(hass, 0.02)
    second = await async_setup_ducking(hass, 0.03)
    assert first is second
    assert shared_data[DUCKER_KEY] is first
    assert shared_data[ENTRY_COUNT_KEY] == 2

    await _set(hass, satellite, "listening")
    assert _data(volume_calls) == [{"entity_id": tv, "volume_level": 0.03}]

    await async_release_ducking(hass)()
    assert len(volume_calls) == 1
    assert shared_data[ENTRY_COUNT_KEY] == 1

    await _set(hass, satellite, "listening", {"source": "again"})
    assert len(volume_calls) == 1

    await _set(hass, satellite, "idle")
    assert len(volume_calls) == 2
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}

    await _set(hass, satellite, "listening")
    assert len(volume_calls) == 3
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.03}
    await async_release_ducking(hass)()
    assert len(volume_calls) == 4
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}
    assert hass.data[DOMAIN] is shared_data
    assert set(shared_data) == {"unrelated"}

    await _set(hass, satellite, "idle")
    await _set(hass, satellite, "listening")
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()
    assert len(volume_calls) == 4


async def test_stop_event_restores_ducked_players(hass: HomeAssistant) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    volume_calls = async_mock_service(hass, "media_player", "volume_set")
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert len(volume_calls) == 1

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()
    assert len(volume_calls) == 2
    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.5}


async def test_failing_restore_does_not_block_other_restores(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    broken = _register(hass, "media_player", "broken_tv", area_id=kitchen)
    sound = _register(hass, "media_player", "soundbar", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, broken, "playing", VOLUME_ATTRS)
    await _set(hass, sound, "playing", VOLUME_ATTRS)

    calls: list[dict[str, Any]] = []

    async def _volume_set(call) -> None:
        if (
            call.data.get("entity_id") == broken
            and call.data.get("volume_level") == 0.5
        ):
            raise RuntimeError("player unreachable")
        calls.append(dict(call.data))

    hass.services.async_register("media_player", "volume_set", _volume_set)
    await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert len(calls) == 2

    await _set(hass, satellite, "idle")
    assert len(calls) == 3
    assert calls[-1] == {"entity_id": sound, "volume_level": 0.5}


async def test_wait_restores_returns_when_restores_complete(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    restore_started = asyncio.Event()
    restore_allowed = asyncio.Event()
    restore_finished = asyncio.Event()

    async def _volume_set(call) -> None:
        if call.data["volume_level"] == 0.5:
            restore_started.set()
            await restore_allowed.wait()
            restore_finished.set()

    hass.services.async_register("media_player", "volume_set", _volume_set)
    ducker = await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    hass.states.async_set(satellite, "idle")
    await asyncio.wait_for(restore_started.wait(), timeout=1.0)
    waiter = asyncio.create_task(ducker.async_wait_restores())
    try:
        await asyncio.sleep(0)
        assert not waiter.done()
        assert not restore_finished.is_set()
        restore_allowed.set()
        await asyncio.wait_for(waiter, timeout=1.0)
        assert restore_finished.is_set()
        assert not ducker._restore_tasks
    finally:
        restore_allowed.set()
        await waiter
        await hass.async_block_till_done()


@pytest.mark.parametrize(
    ("attributes", "duck_service", "restore_service"),
    [
        (VOLUME_ATTRS, "volume_set", "volume_set"),
        (PAUSE_ATTRS, "media_pause", "media_play"),
        (MUTE_ATTRS, "volume_mute", "volume_mute"),
    ],
)
async def test_service_failure_is_isolated_and_restore_is_blocking(
    hass: HomeAssistant,
    attributes: dict[str, Any],
    duck_service: str,
    restore_service: str,
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    broken = _register(hass, "media_player", "broken_tv", area_id=kitchen)
    sound = _register(hass, "media_player", "soundbar", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, broken, "playing", attributes)
    await _set(hass, sound, "playing", attributes)
    await async_setup_ducking(hass, 0.02)

    calls: list[tuple[str, str, bool]] = []

    async def _async_call(domain, service, data, *, blocking) -> None:
        assert domain == "media_player"
        calls.append((data["entity_id"], service, blocking))
        if data["entity_id"] == broken:
            raise RuntimeError("player unreachable")

    with patch.object(type(hass.services), "async_call", side_effect=_async_call):
        await _set(hass, satellite, "listening")
        assert calls == [(broken, duck_service, False), (sound, duck_service, False)]
        await _set(hass, satellite, "idle")
        assert len(calls) == 4
        assert set(calls[2:]) == {
            (broken, restore_service, True),
            (sound, restore_service, True),
        }


async def test_wait_restores_respects_timeout_cap(hass: HomeAssistant) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)

    gate = asyncio.Event()
    restore_started = asyncio.Event()
    restore_finished = asyncio.Event()

    async def _volume_set(call) -> None:
        if call.data.get("volume_level") == 0.5:
            restore_started.set()
            await gate.wait()
            restore_finished.set()

    hass.services.async_register("media_player", "volume_set", _volume_set)
    ducker = await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert len(ducker._restore_tasks) == 0

    hass.states.async_set(satellite, "idle")
    await asyncio.wait_for(restore_started.wait(), timeout=1.0)
    try:
        assert len(ducker._restore_tasks) == 1
        started = asyncio.get_running_loop().time()
        await asyncio.wait_for(ducker.async_wait_restores(timeout=0.02), timeout=0.5)
        assert asyncio.get_running_loop().time() - started < 0.5
        assert len(ducker._restore_tasks) == 1
        assert not restore_finished.is_set()
    finally:
        gate.set()
        await hass.async_block_till_done()
    assert restore_finished.is_set()
    assert not ducker._restore_tasks


async def test_hung_restore_is_abandoned_after_timeout(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    hang = asyncio.Event()

    async def _volume_set(call) -> None:
        if call.data["volume_level"] == 0.5:
            await hang.wait()

    hass.services.async_register("media_player", "volume_set", _volume_set)
    ducker = await async_setup_ducking(hass, 0.02)

    with patch("custom_components.sayso.ducking.RESTORE_CALL_TIMEOUT", 0.05):
        await _set(hass, satellite, "listening")
        await _set(hass, satellite, "idle")

    assert not ducker._restore_tasks
    assert any(
        record.levelno == logging.WARNING
        and "timed out" in record.message
        and tv in record.message
        and "volume_set" in record.message
        for record in caplog.records
    )


async def test_rewake_ducks_only_after_pending_restore(hass: HomeAssistant) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    restore_started = asyncio.Event()
    restore_gate = asyncio.Event()
    volume_calls = []

    async def _volume_set(call) -> None:
        if call.data["volume_level"] == 0.5:
            restore_started.set()
            await restore_gate.wait()
        volume_calls.append(call)

    hass.services.async_register("media_player", "volume_set", _volume_set)
    ducker = await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    hass.states.async_set(satellite, "idle")
    await asyncio.wait_for(restore_started.wait(), timeout=1.0)

    hass.states.async_set(tv, "playing", VOLUME_ATTRS)
    hass.states.async_set(satellite, "listening")
    for _ in range(5):
        await asyncio.sleep(0)

    assert _data(volume_calls) == [{"entity_id": tv, "volume_level": 0.02}]
    assert tv in ducker._restore_tasks
    restore_gate.set()
    await hass.async_block_till_done()
    assert _data(volume_calls) == [
        {"entity_id": tv, "volume_level": 0.02},
        {"entity_id": tv, "volume_level": 0.5},
        {"entity_id": tv, "volume_level": 0.02},
    ]


async def test_stop_event_then_teardown_does_not_remove_fired_listener(
    hass: HomeAssistant, caplog: pytest.LogCaptureFixture
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    ducker = await async_setup_ducking(hass, 0.02)

    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()
    with caplog.at_level(logging.WARNING):
        await ducker.async_teardown()

    assert "Unable to remove unknown job listener" not in caplog.text


async def test_wait_for_restores_is_noop_without_ducker(
    hass: HomeAssistant,
) -> None:
    await async_wait_for_restores(hass)
    shared_data = hass.data.setdefault(DOMAIN, {})
    shared_data["unrelated"] = object()
    await async_wait_for_restores(hass)


async def test_rewake_then_leave_before_restore_does_not_reduck(
    hass: HomeAssistant,
) -> None:
    kitchen = _area(hass, "Kitchen")
    satellite = _register(hass, "assist_satellite", "kitchen_sat", area_id=kitchen)
    tv = _register(hass, "media_player", "kitchen_tv", area_id=kitchen)
    await _set(hass, satellite, "idle")
    await _set(hass, tv, "playing", VOLUME_ATTRS)
    restore_started = asyncio.Event()
    restore_gate = asyncio.Event()

    volume_calls = []

    async def _gated_and_recorded(call) -> None:
        if call.data["volume_level"] == 0.5:
            restore_started.set()
            await restore_gate.wait()
        volume_calls.append(dict(call.data))

    hass.services.async_register("media_player", "volume_set", _gated_and_recorded)
    ducker = await async_setup_ducking(hass, 0.02)

    await _set(hass, satellite, "listening")
    assert volume_calls == [{"entity_id": tv, "volume_level": 0.02}]

    hass.states.async_set(satellite, "idle")
    await asyncio.wait_for(restore_started.wait(), timeout=1.0)

    hass.states.async_set(tv, "playing", {**VOLUME_ATTRS, ATTR_MEDIA_VOLUME_LEVEL: 0.02})
    hass.states.async_set(satellite, "listening")
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(volume_calls) == 1
    assert tv in ducker._originals

    hass.states.async_set(satellite, "idle")
    for _ in range(5):
        await asyncio.sleep(0)
    assert tv not in ducker._originals
    restore_gate.set()
    await hass.async_block_till_done()

    levels = [c["volume_level"] for c in volume_calls]
    assert levels.count(0.02) == 1
    restores = [i for i, level in enumerate(levels) if level == 0.5]
    assert restores and min(restores) > levels.index(0.02)
    assert levels == [0.02, 0.5, 0.5]
    assert not ducker._restore_tasks
