from __future__ import annotations

from datetime import timedelta
from typing import Any

from homeassistant.components.media_player import (
    ATTR_MEDIA_VOLUME_LEVEL,
    ATTR_MEDIA_VOLUME_MUTED,
    MediaPlayerEntityFeature,
)
from homeassistant.const import ATTR_SUPPORTED_FEATURES
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

from custom_components.sayso.const import DUCK_TIMEOUT
from custom_components.sayso.ducking import async_setup_ducking

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

    assert _data(volume_calls)[-1] == {"entity_id": tv, "volume_level": 0.02}


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
