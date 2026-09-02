"""Config flow for SaySo."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL, CONF_PROMPT, CONF_URL
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.llm import LLM_API_ASSIST
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .client import LlamaCppClient, normalize_base_url
from .const import (
    CONF_MAX_OUTPUT_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DOMAIN,
)
from .exceptions import (
    SaySoAuthError,
    SaySoConnectionError,
    SaySoError,
    SaySoInvalidResponseError,
    SaySoModelNotFoundError,
    SaySoTimeoutError,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_URL, default=DEFAULT_BASE_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
)


def redact_api_key(message: str, api_key: str | None) -> str:
    """Remove API key material from user-visible error text."""
    if not api_key:
        return message
    redacted = message.replace(api_key, "***")
    bearer = f"Bearer {api_key}"
    if bearer in redacted:
        redacted = redacted.replace(bearer, "Bearer ***")
    return redacted


def _default_options(model: str) -> dict[str, Any]:
    return {
        CONF_MODEL: model,
        CONF_TIMEOUT: DEFAULT_TIMEOUT,
        CONF_LLM_HASS_API: LLM_API_ASSIST,
        CONF_PROMPT: DEFAULT_SYSTEM_PROMPT,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_MAX_OUTPUT_TOKENS: DEFAULT_MAX_OUTPUT_TOKENS,
        CONF_MAX_TOOL_ITERATIONS: DEFAULT_MAX_TOOL_ITERATIONS,
    }


def _connection_error_key(error: SaySoError) -> str:
    if isinstance(error, SaySoAuthError):
        return "invalid_auth"
    if isinstance(error, (SaySoConnectionError, SaySoTimeoutError)):
        return "cannot_connect"
    if isinstance(error, SaySoInvalidResponseError):
        return "invalid_response"
    if isinstance(error, SaySoModelNotFoundError):
        return "model_not_found"
    return "unknown"


def _entry_title(base_url: str, model: str) -> str:
    parsed = urlparse(normalize_base_url(base_url))
    host = parsed.netloc or parsed.path
    return f"{model} @ {host}"


def _options_schema(
    hass: HomeAssistant, options: dict[str, Any], models: list[str]
) -> vol.Schema:
    llm_options = [
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    ]
    if not llm_options:
        llm_options = [SelectOptionDict(label="Assist", value=LLM_API_ASSIST)]

    model_options = [
        SelectOptionDict(label=model_id, value=model_id) for model_id in models
    ]
    current_model = options.get(CONF_MODEL)
    if isinstance(current_model, str) and current_model and current_model not in models:
        model_options.insert(0, SelectOptionDict(label=current_model, value=current_model))

    return vol.Schema(
        {
            vol.Required(
                CONF_MODEL,
                description={"suggested_value": options.get(CONF_MODEL)},
            ): SelectSelector(
                SelectSelectorConfig(options=model_options, custom_value=True)
            ),
            vol.Required(
                CONF_TIMEOUT,
                description={"suggested_value": options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)},
            ): NumberSelector(
                NumberSelectorConfig(min=1, max=300, step=1, mode=NumberSelectorMode.BOX)
            ),
            vol.Required(
                CONF_LLM_HASS_API,
                description={"suggested_value": options.get(CONF_LLM_HASS_API, LLM_API_ASSIST)},
            ): SelectSelector(SelectSelectorConfig(options=llm_options)),
            vol.Optional(
                CONF_PROMPT,
                description={"suggested_value": options.get(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT)},
            ): TemplateSelector(),
            vol.Required(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)},
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0,
                    max=2,
                    step=0.1,
                    mode=NumberSelectorMode.BOX,
                )
            ),
            vol.Required(
                CONF_MAX_OUTPUT_TOKENS,
                description={
                    "suggested_value": options.get(
                        CONF_MAX_OUTPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS
                    )
                },
            ): NumberSelector(
                NumberSelectorConfig(min=1, max=4096, step=1, mode=NumberSelectorMode.BOX)
            ),
            vol.Required(
                CONF_MAX_TOOL_ITERATIONS,
                description={
                    "suggested_value": options.get(
                        CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS
                    )
                },
            ): NumberSelector(
                NumberSelectorConfig(min=0, max=10, step=1, mode=NumberSelectorMode.BOX)
            ),
        }
    )


class SaySoConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for SaySo."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._base_url: str | None = None
        self._api_key: str | None = None
        self._models: list[str] = []

    async def _async_create_client(self) -> LlamaCppClient:
        assert self._base_url is not None
        return LlamaCppClient.from_hass(
            self.hass,
            self._base_url,
            api_key=self._api_key,
            timeout=DEFAULT_TIMEOUT,
        )

    async def _async_fetch_models(self) -> dict[str, str]:
        errors: dict[str, str] = {}
        try:
            client = await self._async_create_client()
            self._models = await client.list_models()
        except SaySoError as err:
            _LOGGER.debug(
                "SaySo connection failed during config flow: %s",
                redact_api_key(str(err), self._api_key),
            )
            errors["base"] = _connection_error_key(err)
        except Exception:
            _LOGGER.exception("Unexpected exception during SaySo config flow")
            errors["base"] = "unknown"
        return errors

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            base_url = user_input[CONF_URL].strip()
            api_key = user_input.get(CONF_API_KEY)
            if isinstance(api_key, str):
                api_key = api_key.strip() or None

            try:
                cv.url(base_url)
            except vol.Invalid:
                errors["base"] = "invalid_url"
                return self.async_show_form(
                    step_id="user",
                    data_schema=self.add_suggested_values_to_schema(
                        STEP_USER_DATA_SCHEMA, user_input
                    ),
                    errors=errors,
                )

            self._base_url = normalize_base_url(base_url)
            self._api_key = api_key

            errors = await self._async_fetch_models()
            if errors:
                return self.async_show_form(
                    step_id="user",
                    data_schema=self.add_suggested_values_to_schema(
                        STEP_USER_DATA_SCHEMA, user_input
                    ),
                    errors=errors,
                )

            return await self.async_step_model()

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Select the llama.cpp model identifier."""
        assert self._base_url is not None

        if user_input is not None:
            model = user_input[CONF_MODEL].strip()
            try:
                client = await self._async_create_client()
                await client.validate_model(model)
            except SaySoError as err:
                _LOGGER.debug(
                    "SaySo model validation failed: %s",
                    redact_api_key(str(err), self._api_key),
                )
                return self.async_show_form(
                    step_id="model",
                    data_schema=self.add_suggested_values_to_schema(
                        _model_step_schema(self._models), user_input
                    ),
                    errors={"base": _connection_error_key(err)},
                )

            unique_id = f"{self._base_url}|{model}"
            await self.async_set_unique_id(unique_id)
            self._abort_if_unique_id_configured()

            data: dict[str, Any] = {CONF_URL: self._base_url}
            if self._api_key:
                data[CONF_API_KEY] = self._api_key

            return self.async_create_entry(
                title=_entry_title(self._base_url, model),
                data=data,
                options=_default_options(model),
            )

        return self.async_show_form(
            step_id="model",
            data_schema=_model_step_schema(self._models),
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> SaySoOptionsFlowHandler:
        """Return the options flow handler."""
        return SaySoOptionsFlowHandler()


def _model_step_schema(models: list[str]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_MODEL): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(label=model_id, value=model_id)
                        for model_id in models
                    ],
                    custom_value=True,
                )
            )
        }
    )


class SaySoOptionsFlowHandler(OptionsFlowWithReload):
    """Handle SaySo options."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage SaySo options."""
        entry = self.config_entry
        models: list[str] = []

        try:
            client = LlamaCppClient.from_hass(
                self.hass,
                entry.data[CONF_URL],
                api_key=entry.data.get(CONF_API_KEY),
                timeout=entry.options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
            )
            models = await client.list_models()
        except SaySoError as err:
            _LOGGER.debug(
                "SaySo options flow could not fetch models: %s",
                redact_api_key(str(err), entry.data.get(CONF_API_KEY)),
            )

        if user_input is not None:
            model = user_input[CONF_MODEL].strip()
            try:
                client = LlamaCppClient.from_hass(
                    self.hass,
                    entry.data[CONF_URL],
                    api_key=entry.data.get(CONF_API_KEY),
                    timeout=user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
                )
                await client.validate_model(model)
            except SaySoError as err:
                _LOGGER.debug(
                    "SaySo options validation failed: %s",
                    redact_api_key(str(err), entry.data.get(CONF_API_KEY)),
                )
                return self.async_show_form(
                    step_id="init",
                    data_schema=self.add_suggested_values_to_schema(
                        _options_schema(self.hass, entry.options, models),
                        user_input,
                    ),
                    errors={"base": _connection_error_key(err)},
                )

            return self.async_create_entry(data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                _options_schema(self.hass, entry.options, models),
                entry.options,
            ),
        )