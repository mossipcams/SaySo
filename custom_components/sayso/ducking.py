from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.media_player import (
    ATTR_MEDIA_VOLUME_LEVEL,
    ATTR_MEDIA_VOLUME_MUTED,
    DOMAIN as MEDIA_PLAYER_DOMAIN,
    SERVICE_MEDIA_PAUSE,
    SERVICE_MEDIA_PLAY,
    SERVICE_VOLUME_MUTE,
    SERVICE_VOLUME_SET,
    MediaPlayerEntityFeature,
    MediaPlayerState,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_SUPPORTED_FEATURES,
    EVENT_HOMEASSISTANT_STOP,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_added_domain,
    async_track_state_change_event,
)
from homeassistant.util import dt as dt_util

from .const import DOMAIN, DUCK_TIMEOUT
from .conversation import area_id_for
from .vad_sensitivity import SATELLITE_DOMAIN

_LOGGER = logging.getLogger(__name__)

DUCKER_KEY = "ducker"
ENTRY_COUNT_KEY = "ducker_entries"
SATELLITE_LISTENING = "listening"
RESTORE_WAIT_TIMEOUT = 2.0
RESTORE_CALL_TIMEOUT = 10.0
DUCK_LEVEL_TOLERANCE = 0.005
INACTIVE_STATES = (
    MediaPlayerState.OFF,
    MediaPlayerState.STANDBY,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)

Call = tuple[str, str, dict[str, Any]]


class MediaDucker:
    def __init__(self, hass: HomeAssistant, duck_volume: float) -> None:
        self._hass = hass
        self._duck_volume = duck_volume
        self._originals: dict[str, tuple[str, dict[str, Any]]] = {}
        self._restored: dict[str, tuple[float, datetime]] = {}
        self._counts: dict[str, int] = {}
        self._holds: dict[str, set[str]] = {}
        self._timers: dict[str, CALLBACK_TYPE] = {}
        self._unsubs: list[CALLBACK_TYPE] = []
        self._stop_unsub: CALLBACK_TYPE | None = None
        self._restore_tasks: dict[str, asyncio.Task[None]] = {}
        self._satellite_unsubs: dict[str, CALLBACK_TYPE] = {}

    def set_duck_volume(self, duck_volume: float) -> None:
        self._duck_volume = duck_volume

    def start(self) -> None:
        self._stop_unsub = self._hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STOP, self._async_stop
        )
        self._unsubs.append(
            async_track_state_added_domain(
                self._hass, SATELLITE_DOMAIN, self._async_satellite_added
            )
        )
        for entity_id in self._hass.states.async_entity_ids(SATELLITE_DOMAIN):
            self._track(entity_id)

    @callback
    def _async_stop(self, _event: Event) -> None:
        self._stop_unsub = None
        self._schedule_restores(self._async_all_restores())

    def _async_all_restores(self) -> list[Call]:
        restores: list[Call] = []
        for satellite_id in list(self._holds):
            restores.extend(self._release(satellite_id))
        return restores

    def _schedule_restores(self, restores: list[Call]) -> None:
        for call in restores:
            task = self._hass.async_create_task(self._async_send([call], blocking=True))
            self._restore_tasks[call[0]] = task
            task.add_done_callback(
                lambda finished, entity_id=call[0]: self._forget_restore(entity_id, finished)
            )

    def _forget_restore(self, entity_id: str, finished: asyncio.Task[None]) -> None:
        if self._restore_tasks.get(entity_id) is finished:
            del self._restore_tasks[entity_id]

    async def _async_duck_after_restore(
        self, restore: asyncio.Task[None], call: Call
    ) -> None:
        await asyncio.wait([restore], timeout=RESTORE_WAIT_TIMEOUT)
        if call[0] not in self._originals:
            return
        await self._async_send([call])

    async def async_wait_restores(self, timeout: float = RESTORE_WAIT_TIMEOUT) -> None:
        if not self._restore_tasks:
            return
        await asyncio.wait(list(self._restore_tasks.values()), timeout=timeout)

    async def async_teardown(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()
        if self._stop_unsub is not None:
            self._stop_unsub()
            self._stop_unsub = None
        for unsub in self._satellite_unsubs.values():
            unsub()
        self._satellite_unsubs.clear()
        self._schedule_restores(self._async_all_restores())
        await self.async_wait_restores()

    @callback
    def _async_satellite_added(self, event: Event[EventStateChangedData]) -> None:
        self._track(event.data["entity_id"])

    def _track(self, satellite_id: str) -> None:
        if satellite_id in self._satellite_unsubs:
            return
        self._satellite_unsubs[satellite_id] = async_track_state_change_event(
            self._hass, satellite_id, self._async_on_satellite_change
        )

    async def _async_on_satellite_change(
        self, event: Event[EventStateChangedData]
    ) -> None:
        satellite_id = event.data["entity_id"]
        new_state = event.data["new_state"]
        if new_state is not None and new_state.state == SATELLITE_LISTENING:
            await self._async_duck(satellite_id)
        else:
            self._schedule_restores(self._release(satellite_id))

    async def _async_duck(self, satellite_id: str) -> None:
        if satellite_id in self._holds:
            return
        area_id = area_id_for(self._hass, satellite_id)
        if area_id is None:
            return
        calls = self._claim(satellite_id, area_id)
        if self._holds[satellite_id]:
            self._timers[satellite_id] = async_call_later(
                self._hass, DUCK_TIMEOUT, self._make_timeout(satellite_id)
            )
        await self._async_send(calls)

    def _make_timeout(self, satellite_id: str) -> Callable[[datetime], Awaitable[None]]:
        async def _async_timed_out(_now: datetime) -> None:
            restores = self._release(satellite_id)
            self._holds[satellite_id] = set()
            self._schedule_restores(restores)

        return _async_timed_out

    def _claim(self, satellite_id: str, area_id: str) -> list[Call]:
        holds = self._holds[satellite_id] = set()
        satellite_device_id = self._device_id(satellite_id)
        calls: list[Call] = []
        for state in self._hass.states.async_all(MEDIA_PLAYER_DOMAIN):
            if not self._is_target(state, area_id, satellite_device_id):
                continue
            entity_id = state.entity_id
            if entity_id in self._originals:
                self._counts[entity_id] += 1
            else:
                plan = self._duck_plan(state)
                if plan is None:
                    continue
                restore, duck = plan
                self._originals[entity_id] = restore
                self._counts[entity_id] = 1
                pending = self._restore_tasks.get(entity_id)
                if pending is not None:
                    self._hass.async_create_task(
                        self._async_duck_after_restore(pending, (entity_id, *duck))
                    )
                else:
                    calls.append((entity_id, *duck))
            holds.add(entity_id)
        return calls

    def _release(self, satellite_id: str) -> list[Call]:
        timer = self._timers.pop(satellite_id, None)
        if timer is not None:
            timer()
        holds = self._holds.pop(satellite_id, None)
        if holds is None:
            return []
        restores: list[Call] = []
        for entity_id in holds:
            self._counts[entity_id] -= 1
            if self._counts[entity_id] == 0:
                del self._counts[entity_id]
                service, data = self._originals.pop(entity_id)
                if service == SERVICE_VOLUME_SET:
                    self._restored[entity_id] = (
                        data[ATTR_MEDIA_VOLUME_LEVEL],
                        dt_util.utcnow(),
                    )
                restores.append((entity_id, service, data))
        return restores

    def _is_target(
        self, state: State, area_id: str, satellite_device_id: str | None
    ) -> bool:
        if state.state in INACTIVE_STATES:
            return False
        device_id = self._device_id(state.entity_id)
        if satellite_device_id is not None and device_id == satellite_device_id:
            return False
        return area_id_for(self._hass, state.entity_id) == area_id

    def _device_id(self, entity_id: str) -> str | None:
        entry = er.async_get(self._hass).async_get(entity_id)
        return entry.device_id if entry is not None else None

    def _duck_plan(
        self, state: State
    ) -> tuple[tuple[str, dict[str, Any]], tuple[str, dict[str, Any]]] | None:
        remembered = self._restored.pop(state.entity_id, None)
        level = state.attributes.get(ATTR_MEDIA_VOLUME_LEVEL)
        features = state.attributes.get(ATTR_SUPPORTED_FEATURES) or 0
        if (
            level is not None
            and state.state == MediaPlayerState.PLAYING
            and features & MediaPlayerEntityFeature.VOLUME_SET
        ):
            if (
                remembered is not None
                and abs(level - self._duck_volume) <= DUCK_LEVEL_TOLERANCE
                and dt_util.utcnow() - remembered[1] < timedelta(seconds=DUCK_TIMEOUT)
            ):
                level = remembered[0]
            if level <= self._duck_volume:
                return None
            return (
                (SERVICE_VOLUME_SET, {ATTR_MEDIA_VOLUME_LEVEL: level}),
                (SERVICE_VOLUME_SET, {ATTR_MEDIA_VOLUME_LEVEL: self._duck_volume}),
            )
        if (
            features & MediaPlayerEntityFeature.PAUSE
            and state.state == MediaPlayerState.PLAYING
        ):
            return ((SERVICE_MEDIA_PLAY, {}), (SERVICE_MEDIA_PAUSE, {}))
        if features & MediaPlayerEntityFeature.VOLUME_MUTE and not state.attributes.get(
            ATTR_MEDIA_VOLUME_MUTED
        ):
            return (
                (SERVICE_VOLUME_MUTE, {ATTR_MEDIA_VOLUME_MUTED: False}),
                (SERVICE_VOLUME_MUTE, {ATTR_MEDIA_VOLUME_MUTED: True}),
            )
        return None

    async def _async_send(self, calls: list[Call], blocking: bool = False) -> None:
        for entity_id, service, data in calls:
            try:
                async with asyncio.timeout(RESTORE_CALL_TIMEOUT):
                    await self._hass.services.async_call(
                        MEDIA_PLAYER_DOMAIN,
                        service,
                        {ATTR_ENTITY_ID: entity_id, **data},
                        blocking=blocking,
                    )
            except TimeoutError:
                _LOGGER.warning("Ducking restore timed out %s %s", entity_id, service)
            except Exception:
                _LOGGER.exception("Ducking call failed %s %s", entity_id, service)


async def async_setup_ducking(hass: HomeAssistant, duck_volume: float) -> MediaDucker:
    data = hass.data.setdefault(DOMAIN, {})
    ducker = data.get(DUCKER_KEY)
    if ducker is None:
        ducker = MediaDucker(hass, duck_volume)
        ducker.start()
        data[DUCKER_KEY] = ducker
    else:
        ducker.set_duck_volume(duck_volume)
    data[ENTRY_COUNT_KEY] = data.get(ENTRY_COUNT_KEY, 0) + 1
    return ducker


async def async_wait_for_restores(hass: HomeAssistant) -> None:
    data = hass.data.get(DOMAIN)
    if not isinstance(data, dict):
        return
    ducker = data.get(DUCKER_KEY)
    if ducker is not None:
        await ducker.async_wait_restores()


def async_release_ducking(
    hass: HomeAssistant,
) -> Callable[[], Coroutine[Any, Any, None]]:
    async def _async_release() -> None:
        data = hass.data.get(DOMAIN)
        if not data:
            return
        ducker: MediaDucker | None = data.get(DUCKER_KEY)
        count = data.get(ENTRY_COUNT_KEY, 0) - 1
        if count > 0:
            data[ENTRY_COUNT_KEY] = count
            return
        data.pop(DUCKER_KEY, None)
        data.pop(ENTRY_COUNT_KEY, None)
        if ducker is not None:
            await ducker.async_teardown()

    return _async_release
