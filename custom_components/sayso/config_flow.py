"""Config flow for the SaySo Home Assistant integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.selector import (
    AreaSelector,
    EntitySelector,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    ACTION_OPTIONS,
    CONF_ACTION_ALLOWLIST,
    CONF_AREA_IDS,
    CONF_DOMAIN_ALLOWLIST,
    CONF_ENTITY_IDS,
    CONF_EXPOSURE_MODE,
    CONF_TOKEN,
    CONF_URL,
    DOMAIN,
    DOMAIN_OPTIONS,
    EXPOSURE_MODE_ALL,
    EXPOSURE_MODE_AREA,
    EXPOSURE_MODE_ENTITY,
    get_entry_options,
    normalize_options,
)

_LOGGER = logging.getLogger(__name__)

HEALTH_PATH = "/api/v1/health"


class CannotConnect(Exception):
    """SaySo server is unreachable or returned an unexpected response."""


class InvalidAuth(Exception):
    """SaySo server rejected the provided token."""


async def probe_connection(url: str, token: str) -> None:
    """Verify the SaySo server accepts the token at the configured URL."""

    def _probe() -> None:
        request = Request(
            f"{url.rstrip('/')}{HEALTH_PATH}",
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=10) as response:
                if response.status >= 400:
                    raise CannotConnect()
        except HTTPError as exc:
            if exc.code == 401:
                raise InvalidAuth() from exc
            raise CannotConnect() from exc
        except URLError as exc:
            raise CannotConnect() from exc

    await asyncio.to_thread(_probe)


def _normalize_url(raw_url: str) -> str:
    return raw_url.strip().rstrip("/")


def _validate_url(raw_url: str) -> str | None:
    normalized = _normalize_url(raw_url)
    if not normalized:
        return "required"
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "invalid_url"
    return None


class SaySoConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle SaySo integration setup from the UI."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        return SaySoOptionsFlowHandler()

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            url = _normalize_url(user_input.get(CONF_URL, ""))
            token = user_input.get(CONF_TOKEN, "").strip()

            url_error = _validate_url(url)
            if url_error:
                errors[CONF_URL] = url_error
            if not token:
                errors[CONF_TOKEN] = "required"

            if not errors:
                try:
                    await probe_connection(url, token)
                except InvalidAuth:
                    errors["base"] = "invalid_auth"
                except CannotConnect:
                    _LOGGER.warning("SaySo connection probe failed for %s", url)
                    errors["base"] = "cannot_connect"
                else:
                    await self.async_set_unique_id(url)
                    self._abort_if_unique_id_configured()
                    return self.async_create_entry(
                        title="SaySo",
                        data={CONF_URL: url, CONF_TOKEN: token},
                    )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )


STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_URL): str,
        vol.Required(CONF_TOKEN): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD),
        ),
    }
)


class SaySoOptionsFlowHandler(config_entries.OptionsFlowWithReload):
    """Handle SaySo integration options from the UI."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        if user_input is not None:
            return self.async_create_entry(
                title="",
                data=normalize_options(user_input),
            )

        current = get_entry_options(self.config_entry)
        return self.async_show_form(
            step_id="init",
            data_schema=_options_schema(current),
        )


def _options_schema(current: dict[str, list[str] | str]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Optional(
                CONF_DOMAIN_ALLOWLIST,
                default=list(current[CONF_DOMAIN_ALLOWLIST]),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=d, label=d) for d in DOMAIN_OPTIONS
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                    multiple=True,
                ),
            ),
            vol.Optional(
                CONF_ACTION_ALLOWLIST,
                default=list(current[CONF_ACTION_ALLOWLIST]),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=a, label=a) for a in ACTION_OPTIONS
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                    multiple=True,
                ),
            ),
            vol.Required(
                CONF_EXPOSURE_MODE,
                default=str(current[CONF_EXPOSURE_MODE]),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=EXPOSURE_MODE_ALL, label="All entities"),
                        SelectOptionDict(
                            value=EXPOSURE_MODE_AREA,
                            label="Selected areas",
                        ),
                        SelectOptionDict(
                            value=EXPOSURE_MODE_ENTITY,
                            label="Selected entities",
                        ),
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                ),
            ),
            vol.Optional(
                CONF_AREA_IDS,
                default=list(current[CONF_AREA_IDS]),
            ): AreaSelector({"multiple": True}),
            vol.Optional(
                CONF_ENTITY_IDS,
                default=list(current[CONF_ENTITY_IDS]),
            ): EntitySelector({"multiple": True}),
        }
    )
