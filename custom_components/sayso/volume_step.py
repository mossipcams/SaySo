from __future__ import annotations

import asyncio
import logging
from typing import override

import voluptuous as vol

from homeassistant.components.media_player import (
    ATTR_MEDIA_VOLUME_LEVEL,
    DATA_COMPONENT,
    DOMAIN,
    MediaPlayerDeviceClass,
    MediaPlayerEntityFeature,
    SERVICE_VOLUME_SET,
)
from homeassistant.components.media_player.const import INTENT_SET_VOLUME
from homeassistant.components.media_player.intent import MediaSetVolumeRelativeHandler
from homeassistant.const import STATE_PLAYING
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent

_LOGGER = logging.getLogger(__name__)


class StepVolumeRelativeHandler(MediaSetVolumeRelativeHandler):

    @override
    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        from .ducking import async_wait_for_restores

        await async_wait_for_restores(intent_obj.hass)
        try:
            return await super().async_handle(intent_obj)
        except intent.MatchFailedError as err:
            if err.result.no_match_reason is not intent.MatchFailedReason.FEATURE:
                raise

            slots = self.async_validate_slots(intent_obj.slots)
            constraints = intent.MatchTargetsConstraints(
                name=slots.get("name", {}).get("value"),
                area_name=slots.get("area", {}).get("value"),
                floor_name=slots.get("floor", {}).get("value"),
                domains={DOMAIN},
                assistant=intent_obj.assistant,
                features=MediaPlayerEntityFeature.VOLUME_STEP,
            )
            preferences = intent.MatchTargetsPreferences(
                area_id=slots.get("preferred_area_id", {}).get("value"),
                floor_id=slots.get("preferred_floor_id", {}).get("value"),
            )
            match_result = intent.async_match_targets(
                intent_obj.hass, constraints, preferences
            )
            if not match_result.is_match:
                raise

            states = match_result.states
            if len(states) > 1 and "name" not in intent_obj.slots:
                states = [s for s in states if s.state == STATE_PLAYING]
                if not states:
                    raise intent.MatchFailedError(
                        result=intent.MatchTargetsResult(
                            is_match=False,
                            no_match_reason=intent.MatchFailedReason.STATE,
                        ),
                        constraints=constraints,
                        preferences=preferences,
                    ) from err

            entity_ids = {s.entity_id for s in states}
            entities = [
                e
                for e in intent_obj.hass.data[DATA_COMPONENT].entities
                if e.entity_id in entity_ids
            ]

            volume_step = slots["volume_step"]["value"]
            if volume_step == "up":
                direction = 1
            elif volume_step == "down":
                direction = -1
            else:
                direction = (volume_step > 0) - (volume_step < 0)

            if direction > 0:
                coros = [e.async_volume_up() for e in entities]
            elif direction < 0:
                coros = [e.async_volume_down() for e in entities]
            else:
                coros = []

            try:
                await asyncio.gather(*coros)
            except HomeAssistantError as step_err:
                _LOGGER.error("Error setting relative volume: %s", step_err)
                raise intent.IntentHandleError(
                    f"Error setting relative volume: {step_err}"
                ) from step_err

            response = intent_obj.create_response()
            response.async_set_states(states)
            return response


class StepSetVolumeHandler(intent.ServiceIntentHandler):
    def __init__(self) -> None:
        super().__init__(
            INTENT_SET_VOLUME,
            DOMAIN,
            SERVICE_VOLUME_SET,
            required_domains={DOMAIN},
            required_features=MediaPlayerEntityFeature.VOLUME_SET,
            required_slots={
                ATTR_MEDIA_VOLUME_LEVEL: intent.IntentSlotInfo(
                    description="The volume percentage of the media player",
                    value_schema=vol.All(
                        vol.Coerce(int),
                        vol.Range(min=0, max=100),
                        lambda val: val / 100,
                    ),
                ),
            },
            description="Sets the volume percentage of a media player",
            platforms={DOMAIN},
            device_classes={MediaPlayerDeviceClass},
        )

    @override
    async def async_handle(self, intent_obj: intent.Intent) -> intent.IntentResponse:
        from .ducking import async_wait_for_restores

        await async_wait_for_restores(intent_obj.hass)
        return await super().async_handle(intent_obj)


def async_register_volume_step(hass: HomeAssistant) -> None:
    intent.async_register(hass, StepVolumeRelativeHandler())
    intent.async_register(hass, StepSetVolumeHandler())
