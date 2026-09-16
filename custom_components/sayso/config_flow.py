"""Config flow for SaySo."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL, CONF_URL
from homeassistant.core import HomeAssistant, callback

from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.llm import LLM_API_ASSIST
from homeassistant.helpers.selector import (
    BooleanSelector,
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
    BACKEND_EMBEDDED,
    CONF_BACKEND,
    CONF_MAX_OUTPUT_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_MODEL_PATH,
    CONF_N_CTX,
    CONF_N_THREADS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    CONF_TRACE_MAX_INTERACTIONS,
    CONF_TRACE_RETENTION_DAYS,
    CONF_TRACE_STORE_UTTERANCES,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_N_CTX,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TRACE_MAX_INTERACTIONS,
    DEFAULT_TRACE_RETENTION_DAYS,
    DEFAULT_TRACE_STORE_UTTERANCES,
    DOMAIN,
    MAX_TRACE_INTERACTIONS,
)
from .inference import default_thread_count, entry_backend
from .model_store import list_local_models
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
        # Optional and not prefilled on purpose: blank means local inference.
        vol.Optional(CONF_URL): TextSelector(
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


def _number(minimum: float, maximum: float, step: float = 1) -> NumberSelector:
    """A plain numeric box; every numeric option uses the same widget."""
    return NumberSelector(
        NumberSelectorConfig(
            min=minimum, max=maximum, step=step, mode=NumberSelectorMode.BOX
        )
    )


# The shared option contract, in the order the form shows it: marker, key,
# default, widget. Both the options form and the defaults a new entry starts
# with read this table, so the two cannot drift apart. The one selector that is
# None is the Home Assistant LLM API picker, whose choices come from the
# running instance rather than from a constant.
_OPTIONS: tuple[tuple[Any, str, Any, Any], ...] = (
    (vol.Required, CONF_TIMEOUT, DEFAULT_TIMEOUT, _number(1, 300)),
    (vol.Required, CONF_LLM_HASS_API, LLM_API_ASSIST, None),
    (vol.Optional, CONF_PROMPT, DEFAULT_SYSTEM_PROMPT, TemplateSelector()),
    (vol.Required, CONF_TEMPERATURE, DEFAULT_TEMPERATURE, _number(0, 2, 0.1)),
    (
        vol.Required,
        CONF_MAX_OUTPUT_TOKENS,
        DEFAULT_MAX_OUTPUT_TOKENS,
        _number(1, 4096),
    ),
    (vol.Required, CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS, _number(0, 10)),
    (
        vol.Required,
        CONF_TRACE_RETENTION_DAYS,
        DEFAULT_TRACE_RETENTION_DAYS,
        _number(1, 365),
    ),
    (
        vol.Required,
        CONF_TRACE_MAX_INTERACTIONS,
        DEFAULT_TRACE_MAX_INTERACTIONS,
        _number(1, MAX_TRACE_INTERACTIONS),
    ),
    (
        vol.Required,
        CONF_TRACE_STORE_UTTERANCES,
        DEFAULT_TRACE_STORE_UTTERANCES,
        BooleanSelector(),
    ),
)

_OPTION_DEFAULTS: dict[str, Any] = {key: default for _m, key, default, _s in _OPTIONS}

_CONNECTION_ERROR_KEYS: tuple[tuple[type[SaySoError], str], ...] = (
    (SaySoAuthError, "invalid_auth"),
    (SaySoConnectionError, "cannot_connect"),
    (SaySoTimeoutError, "cannot_connect"),
    (SaySoInvalidResponseError, "invalid_response"),
    (SaySoModelNotFoundError, "model_not_found"),
)


def _default_embedded_options() -> dict[str, Any]:
    """Options for a local install: no server, URL, port, or API key."""
    return {
        CONF_MODEL: DEFAULT_MODEL_FILENAME,
        CONF_MODEL_PATH: "",
        CONF_N_CTX: DEFAULT_N_CTX,
        CONF_N_THREADS: default_thread_count(),
        **_OPTION_DEFAULTS,
    }


def _default_options(model: str) -> dict[str, Any]:
    """Options a new external entry starts with."""
    return {CONF_MODEL: model, **_OPTION_DEFAULTS}


def _connection_error_key(error: SaySoError) -> str:
    """Map a client failure to the translation key the form shows."""
    for error_class, key in _CONNECTION_ERROR_KEYS:
        if isinstance(error, error_class):
            return key
    return "unknown"


def _entry_title(base_url: str, model: str) -> str:
    parsed = urlparse(normalize_base_url(base_url))
    host = parsed.netloc or parsed.path
    return f"{model} @ {host}"


def _suggest(options: dict[str, Any], key: str, default: Any = None) -> dict[str, Any]:
    """Prefill a field with the entry's current value."""
    return {"suggested_value": options.get(key, default)}


def _select(values: list[str], *, custom: bool = False) -> SelectSelector:
    return SelectSelector(
        SelectSelectorConfig(
            options=[SelectOptionDict(label=value, value=value) for value in values],
            custom_value=custom,
        )
    )


def _model_fields(
    options: dict[str, Any], models: list[str], *, embedded: bool
) -> dict[Any, Any]:
    """The part of the form that differs by backend.

    A local install has no model list to choose from; it has a file on disk and
    the two knobs that decide how much of the machine inference may use.
    """
    if not embedded:
        current = options.get(CONF_MODEL)
        choices = list(models)
        if isinstance(current, str) and current and current not in choices:
            choices.insert(0, current)
        return {
            vol.Required(CONF_MODEL, description=_suggest(options, CONF_MODEL)): _select(
                choices, custom=True
            )
        }

    # Files already in the model directory, labelled by name. Custom values
    # keep a GGUF stored elsewhere selectable; empty still means the default.
    choices = list(models)
    current = options.get(CONF_MODEL_PATH)
    if isinstance(current, str) and current and current not in choices:
        choices.insert(0, current)
    return {
        vol.Optional(
            CONF_MODEL_PATH, description=_suggest(options, CONF_MODEL_PATH, "")
        ): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(label=Path(path).name, value=path)
                    for path in choices
                ],
                custom_value=True,
            )
        ),
        vol.Required(
            CONF_N_THREADS,
            description=_suggest(options, CONF_N_THREADS, default_thread_count()),
        ): _number(1, 32),
        vol.Required(
            CONF_N_CTX, description=_suggest(options, CONF_N_CTX, DEFAULT_N_CTX)
        ): _number(512, 32768, 512),
    }


def _options_schema(
    hass: HomeAssistant,
    options: dict[str, Any],
    models: list[str],
    *,
    embedded: bool = False,
) -> vol.Schema:
    """Build the options form: the backend-specific model fields, then the table."""
    llm_apis = SelectSelector(
        SelectSelectorConfig(
            options=[
                SelectOptionDict(label=api.name, value=api.id)
                for api in llm.async_get_apis(hass)
            ]
            or [SelectOptionDict(label="Assist", value=LLM_API_ASSIST)]
        )
    )
    return vol.Schema(
        {
            **_model_fields(options, models, embedded=embedded),
            **{
                marker(key, description=_suggest(options, key, default)): (
                    selector or llm_apis
                )
                for marker, key, default, selector in _OPTIONS
            },
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

    def _client(self) -> LlamaCppClient:
        assert self._base_url is not None
        return LlamaCppClient.from_hass(
            self.hass, self._base_url, api_key=self._api_key, timeout=DEFAULT_TIMEOUT
        )

    def _step(
        self, step_id: str, schema: vol.Schema, user_input: Any, errors: dict[str, str]
    ) -> ConfigFlowResult:
        """Redisplay a step with the user's own values and an error."""
        return self.async_show_form(
            step_id=step_id,
            data_schema=self.add_suggested_values_to_schema(schema, user_input),
            errors=errors,
        )

    async def _async_fetch_models(self) -> dict[str, str]:
        """Populate the model list, or report why the server could not answer."""
        try:
            self._models = await self._client().list_models()
        except SaySoError as err:
            _LOGGER.debug(
                "SaySo connection failed during config flow: %s",
                redact_api_key(str(err), self._api_key),
            )
            return {"base": _connection_error_key(err)}
        except Exception:
            _LOGGER.exception("Unexpected exception during SaySo config flow")
            return {"base": "unknown"}
        return {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Set up SaySo.

        Local inference is the default: submitting without a URL runs the model
        on this device. Supplying a URL selects the advanced external backend.
        """
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        base_url = (user_input.get(CONF_URL) or "").strip()
        if not base_url:
            await self.async_set_unique_id(f"{DOMAIN}_embedded")
            self._abort_if_unique_id_configured()
            # The wheel and the model are provisioned during entry setup, not
            # here, so the flow never blocks the UI on a large download.
            return self.async_create_entry(
                title="SaySo (local)",
                data={CONF_BACKEND: BACKEND_EMBEDDED},
                options=_default_embedded_options(),
            )

        try:
            cv.url(base_url)
        except vol.Invalid:
            return self._step(
                "user", STEP_USER_DATA_SCHEMA, user_input, {"base": "invalid_url"}
            )

        api_key = user_input.get(CONF_API_KEY)
        self._base_url = normalize_base_url(base_url)
        self._api_key = api_key.strip() or None if isinstance(api_key, str) else api_key

        if errors := await self._async_fetch_models():
            return self._step("user", STEP_USER_DATA_SCHEMA, user_input, errors)
        return await self.async_step_model()

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Select the llama.cpp model identifier."""
        assert self._base_url is not None
        schema = _model_step_schema(self._models)
        if user_input is None:
            return self.async_show_form(step_id="model", data_schema=schema)

        model = user_input[CONF_MODEL].strip()
        try:
            await self._client().validate_model(model)
        except SaySoError as err:
            _LOGGER.debug(
                "SaySo model validation failed: %s",
                redact_api_key(str(err), self._api_key),
            )
            return self._step(
                "model", schema, user_input, {"base": _connection_error_key(err)}
            )

        await self.async_set_unique_id(f"{self._base_url}|{model}")
        self._abort_if_unique_id_configured()
        data: dict[str, Any] = {CONF_URL: self._base_url}
        if self._api_key:
            data[CONF_API_KEY] = self._api_key
        return self.async_create_entry(
            title=_entry_title(self._base_url, model),
            data=data,
            options=_default_options(model),
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> SaySoOptionsFlowHandler:
        """Return the options flow handler."""
        return SaySoOptionsFlowHandler()


def _model_step_schema(models: list[str]) -> vol.Schema:
    return vol.Schema({vol.Required(CONF_MODEL): _select(models, custom=True)})


class SaySoOptionsFlowHandler(OptionsFlowWithReload):
    """Handle SaySo options."""

    def _client(self, timeout: float) -> LlamaCppClient:
        """A client for the entry's external server."""
        entry = self.config_entry
        return LlamaCppClient.from_hass(
            self.hass,
            entry.data[CONF_URL],
            api_key=entry.data.get(CONF_API_KEY),
            timeout=timeout,
        )

    def _log(self, what: str, err: SaySoError) -> None:
        _LOGGER.debug(
            "SaySo options flow %s: %s",
            what,
            redact_api_key(str(err), self.config_entry.data.get(CONF_API_KEY)),
        )

    def _form(
        self,
        options: dict[str, Any],
        models: list[str],
        *,
        embedded: bool = False,
        errors: dict[str, str] | None = None,
    ) -> ConfigFlowResult:
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                _options_schema(self.hass, self.config_entry.options, models, embedded=embedded),
                options,
            ),
            errors=errors,
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage SaySo options."""
        entry = self.config_entry

        if entry_backend(entry) == BACKEND_EMBEDDED:
            # No server to reach and no model list to validate against; the
            # entry reloads on save and the engine picks up the new settings.
            if user_input is not None:
                return self.async_create_entry(data={**entry.options, **user_input})
            local_models = await self.hass.async_add_executor_job(
                list_local_models, self.hass
            )
            return self._form(entry.options, local_models, embedded=True)

        models: list[str] = []
        try:
            models = await self._client(
                entry.options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
            ).list_models()
        except SaySoError as err:
            self._log("could not fetch models", err)

        if user_input is None:
            return self._form(entry.options, models)

        try:
            await self._client(
                user_input.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
            ).validate_model(user_input[CONF_MODEL].strip())
        except SaySoError as err:
            self._log("validation failed", err)
            return self._form(
                user_input, models, errors={"base": _connection_error_key(err)}
            )

        return self.async_create_entry(data=user_input)
