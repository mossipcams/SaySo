from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowContext,
    ConfigFlowResult,
    ConfigSubentryData,
    ConfigSubentryFlow,
    OptionsFlow,
    SubentryFlowResult,
)
from homeassistant.const import (
    CONF_API_KEY,
    CONF_LLM_HASS_API,
    CONF_MODEL,
    CONF_NAME,
    CONF_URL,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowHandler

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
    AI_TASK_SUBENTRY_TYPE,
    BACKEND_EMBEDDED,
    BACKEND_EXTERNAL,
    CONF_BACKEND,
    CONF_DUCK_VOLUME,
    CONF_GRAMMAR_DECODING,
    CONF_MAX_OUTPUT_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_N_CTX,
    CONF_N_THREADS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    CONF_TRACE_MAX_INTERACTIONS,
    CONF_TRACE_RETENTION_DAYS,
    CONF_TRACE_STORE_UTTERANCES,
    CONVERSATION_SUBENTRY_TYPE,
    DEFAULT_DUCK_VOLUME,
    DEFAULT_GRAMMAR_DECODING,
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
    EMBEDDED_MODELS,
    MAX_TRACE_INTERACTIONS,
)
from .inference import default_thread_count, entry_backend
from .model_store import async_provision_model, list_local_models, models_dir
from .core.exceptions import (
    SaySoAuthError,
    SaySoConnectionError,
    SaySoDependencyError,
    SaySoError,
    SaySoInvalidResponseError,
    SaySoModelNotFoundError,
    SaySoTimeoutError,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
)


def redact_api_key(message: str, api_key: str | None) -> str:
    if not api_key:
        return message
    redacted = message.replace(api_key, "***")
    bearer = f"Bearer {api_key}"
    if bearer in redacted:
        redacted = redacted.replace(bearer, "Bearer ***")
    return redacted


def _number(minimum: float, maximum: float, step: float = 1) -> NumberSelector:
    return NumberSelector(
        NumberSelectorConfig(
            min=minimum, max=maximum, step=step, mode=NumberSelectorMode.BOX
        )
    )


_ENGINE_OPTIONS: tuple[tuple[Any, str, Any, Any], ...] = (
    (vol.Required, CONF_TIMEOUT, DEFAULT_TIMEOUT, _number(1, 300)),
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
    (
        vol.Required,
        CONF_GRAMMAR_DECODING,
        DEFAULT_GRAMMAR_DECODING,
        BooleanSelector(),
    ),
    (vol.Required, CONF_DUCK_VOLUME, DEFAULT_DUCK_VOLUME, _number(0, 1, 0.01)),
)

_CONVERSATION_OPTIONS: tuple[tuple[Any, str, Any, Any], ...] = (
    (vol.Required, CONF_LLM_HASS_API, LLM_API_ASSIST, None),
    (vol.Optional, CONF_PROMPT, DEFAULT_SYSTEM_PROMPT, TemplateSelector()),
    (vol.Required, CONF_TEMPERATURE, DEFAULT_TEMPERATURE, _number(0, 2, 0.1)),
    (
        vol.Required,
        CONF_MAX_OUTPUT_TOKENS,
        DEFAULT_MAX_OUTPUT_TOKENS,
        _number(1, 4096),
    ),
    (
        vol.Required,
        CONF_MAX_TOOL_ITERATIONS,
        DEFAULT_MAX_TOOL_ITERATIONS,
        _number(0, 10),
    ),
)

_ENGINE_DEFAULTS: dict[str, Any] = {
    key: default for _m, key, default, _s in _ENGINE_OPTIONS
}
_CONVERSATION_DEFAULTS: dict[str, Any] = {
    key: default for _m, key, default, _s in _CONVERSATION_OPTIONS
}

_CONNECTION_ERROR_KEYS: tuple[tuple[type[SaySoError], str], ...] = (
    (SaySoAuthError, "invalid_auth"),
    (SaySoConnectionError, "cannot_connect"),
    (SaySoTimeoutError, "cannot_connect"),
    (SaySoInvalidResponseError, "invalid_response"),
    (SaySoModelNotFoundError, "model_not_found"),
)


def _default_embedded_options() -> dict[str, Any]:
    return {
        CONF_MODEL: DEFAULT_MODEL_FILENAME,
        CONF_N_CTX: DEFAULT_N_CTX,
        CONF_N_THREADS: default_thread_count(),
        **_ENGINE_DEFAULTS,
    }


def _default_conversation_subentry(
    title: str, model: str | None = None
) -> ConfigSubentryData:
    data: dict[str, Any] = dict(_CONVERSATION_DEFAULTS)
    if model is not None:
        data[CONF_MODEL] = model
    return ConfigSubentryData(
        data=data,
        subentry_type=CONVERSATION_SUBENTRY_TYPE,
        title=title,
        unique_id=None,
    )


def _connection_error_key(error: SaySoError) -> str:
    for error_class, key in _CONNECTION_ERROR_KEYS:
        if isinstance(error, error_class):
            return key
    return "unknown"


def _entry_title(base_url: str, model: str) -> str:
    parsed = urlparse(normalize_base_url(base_url))
    host = parsed.netloc or parsed.path
    return f"{model} @ {host}"


def _suggest(
    options: Mapping[str, Any], key: str, default: Any = None
) -> dict[str, Any]:
    return {"suggested_value": options.get(key, default)}


def _select(values: list[str], *, custom: bool = False) -> SelectSelector:
    return SelectSelector(
        SelectSelectorConfig(
            options=[SelectOptionDict(label=value, value=value) for value in values],
            custom_value=custom,
        )
    )


class DownloadSteps(FlowHandler[ConfigFlowContext, ConfigFlowResult, str]):
    hass: HomeAssistant
    _model: str
    _replace: bool = False
    _download_task: asyncio.Task[None] | None = None
    _download_error: str = ""

    async def async_step_download(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if self._download_task is None:
            self._download_task = self.hass.async_create_task(
                async_provision_model(self.hass, self._model, replace=self._replace)
            )
        if not self._download_task.done():
            return self.async_show_progress(
                step_id="download",
                progress_action="download",
                progress_task=self._download_task,
            )
        try:
            await self._download_task
        except SaySoDependencyError:
            self._download_error = "install_failed"
        except SaySoError:
            self._download_error = "download_failed"
        else:
            return self.async_show_progress_done(next_step_id="finish")
        return self.async_show_progress_done(next_step_id="download_failed")

    async def async_step_download_failed(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is None:
            return self.async_show_form(
                step_id="download_failed",
                data_schema=vol.Schema({}),
                errors={"base": self._download_error},
            )
        self._download_task = None
        return await self.async_step_download()


def _model_fields(
    options: Mapping[str, Any], models: list[str], *, embedded: bool
) -> dict[Any, Any]:
    if not embedded:
        current = options.get(CONF_MODEL)
        choices = list(models)
        if isinstance(current, str) and current and current not in choices:
            choices.insert(0, current)
        return {
            vol.Required(
                CONF_MODEL, description=_suggest(options, CONF_MODEL)
            ): _select(choices, custom=True)
        }

    local_models = [path for path in models if Path(path).name not in EMBEDDED_MODELS]
    return {
        vol.Required(
            CONF_MODEL,
            description=_suggest(options, CONF_MODEL, DEFAULT_MODEL_FILENAME),
        ): SelectSelector(
            SelectSelectorConfig(
                options=[
                    *(
                        SelectOptionDict(label=model.label, value=filename)
                        for filename, model in EMBEDDED_MODELS.items()
                    ),
                    *(
                        SelectOptionDict(label=Path(path).name, value=path)
                        for path in local_models
                    ),
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


def _fields(
    options: Mapping[str, Any],
    spec: tuple[tuple[Any, str, Any, Any], ...],
    llm_apis: SelectSelector | None = None,
) -> dict[Any, Any]:
    return {
        marker(key, description=_suggest(options, key, default)): (selector or llm_apis)
        for marker, key, default, selector in spec
    }


def _engine_schema(
    options: Mapping[str, Any], models: list[str], *, embedded: bool
) -> vol.Schema:
    return vol.Schema(
        {
            **(_model_fields(options, models, embedded=True) if embedded else {}),
            **_fields(options, _ENGINE_OPTIONS),
        }
    )


def _conversation_schema(
    hass: HomeAssistant,
    suggested: Mapping[str, Any],
    models: list[str],
    *,
    external: bool,
) -> vol.Schema:
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
            vol.Required(CONF_NAME, description=_suggest(suggested, CONF_NAME)): (
                TextSelector()
            ),
            **(_model_fields(suggested, models, embedded=False) if external else {}),
            **_fields(suggested, _CONVERSATION_OPTIONS, llm_apis),
        }
    )


async def _async_list_models(hass: HomeAssistant, entry: ConfigEntry) -> list[str]:
    client = LlamaCppClient.from_hass(
        hass,
        entry.data[CONF_URL],
        api_key=entry.data.get(CONF_API_KEY),
        timeout=entry.options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
    )
    try:
        return await client.list_models()
    except SaySoError as err:
        _LOGGER.debug(
            "SaySo could not list models: %s",
            redact_api_key(str(err), entry.data.get(CONF_API_KEY)),
        )
        return []


def _subentry_payload(user_input: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    data = dict(user_input)
    return data.pop(CONF_NAME), data


class SaySoConfigFlow(DownloadSteps, ConfigFlow, domain=DOMAIN):
    VERSION = 2
    MINOR_VERSION = 1

    def __init__(self) -> None:
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
        return self.async_show_form(
            step_id=step_id,
            data_schema=self.add_suggested_values_to_schema(schema, user_input),
            errors=errors,
        )

    async def _async_fetch_models(self) -> dict[str, str]:
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
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        base_url = (user_input.get(CONF_URL) or "").strip()
        if not base_url:
            await self.async_set_unique_id(f"{DOMAIN}_embedded")
            self._abort_if_unique_id_configured()
            self._model = DEFAULT_MODEL_FILENAME
            return await self.async_step_download()

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

        await self.async_set_unique_id(self._base_url)
        self._abort_if_unique_id_configured()
        data: dict[str, Any] = {CONF_URL: self._base_url}
        if self._api_key:
            data[CONF_API_KEY] = self._api_key
        title = _entry_title(self._base_url, model)
        return self.async_create_entry(
            title=title,
            data=data,
            options=dict(_ENGINE_DEFAULTS),
            subentries=[_default_conversation_subentry(title, model)],
        )

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        title = "SaySo (local)"
        return self.async_create_entry(
            title=title,
            data={CONF_BACKEND: BACKEND_EMBEDDED},
            options=_default_embedded_options(),
            subentries=[_default_conversation_subentry(title)],
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> SaySoOptionsFlowHandler:
        return SaySoOptionsFlowHandler()

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        return {
            CONVERSATION_SUBENTRY_TYPE: SaySoConversationSubentryFlow,
            AI_TASK_SUBENTRY_TYPE: SaySoAITaskSubentryFlow,
        }


def _model_step_schema(models: list[str]) -> vol.Schema:
    return vol.Schema({vol.Required(CONF_MODEL): _select(models, custom=True)})


_AI_TASK_SCHEMA = vol.Schema({vol.Required(CONF_NAME): TextSelector()})


class SaySoConversationSubentryFlow(ConfigSubentryFlow):
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        entry = self._get_entry()
        if user_input is None:
            return await self._async_show_form(
                "user", entry, dict(_CONVERSATION_DEFAULTS)
            )
        title, data = _subentry_payload(user_input)
        return self.async_create_entry(title=title, data=data)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        entry = self._get_entry()
        subentry = self._get_reconfigure_subentry()
        if user_input is None:
            return await self._async_show_form(
                "reconfigure", entry, {CONF_NAME: subentry.title, **subentry.data}
            )
        title, data = _subentry_payload(user_input)
        return self.async_update_and_abort(entry, subentry, title=title, data=data)

    async def _async_show_form(
        self, step_id: str, entry: ConfigEntry, suggested: Mapping[str, Any]
    ) -> SubentryFlowResult:
        external = entry_backend(entry) == BACKEND_EXTERNAL
        models = await _async_list_models(self.hass, entry) if external else []
        return self.async_show_form(
            step_id=step_id,
            data_schema=_conversation_schema(
                self.hass, suggested, models, external=external
            ),
        )


class SaySoAITaskSubentryFlow(ConfigSubentryFlow):
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if user_input is None:
            return self.async_show_form(step_id="user", data_schema=_AI_TASK_SCHEMA)
        return self.async_create_entry(title=user_input[CONF_NAME], data={})

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        subentry = self._get_reconfigure_subentry()
        if user_input is None:
            return self.async_show_form(
                step_id="reconfigure",
                data_schema=self.add_suggested_values_to_schema(
                    _AI_TASK_SCHEMA, {CONF_NAME: subentry.title}
                ),
            )
        return self.async_update_and_abort(
            self._get_entry(), subentry, title=user_input[CONF_NAME]
        )


class SaySoOptionsFlowHandler(DownloadSteps, OptionsFlow):
    _pending: dict[str, Any]

    def _form(
        self,
        suggested: Mapping[str, Any],
        *,
        embedded: bool,
        models: list[str] | None = None,
    ) -> ConfigFlowResult:
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                _engine_schema(
                    self.config_entry.options, models or [], embedded=embedded
                ),
                suggested,
            ),
        )

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        entry = self.config_entry

        if entry_backend(entry) != BACKEND_EMBEDDED:
            if user_input is None:
                return self._form(entry.options, embedded=False)
            return self.async_create_entry(data=user_input)

        if user_input is not None:
            self._pending = {**entry.options, **user_input}
            self._model = user_input[CONF_MODEL]
            if await self._async_model_absent(self._model):
                return await self.async_step_download()
            return self.async_create_entry(data=self._pending)
        local_models = await self.hass.async_add_executor_job(
            list_local_models, self.hass
        )
        return self._form(entry.options, embedded=True, models=local_models)

    async def async_step_finish(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return self.async_create_entry(data=self._pending)

    async def _async_model_absent(self, model: str) -> bool:
        if model not in EMBEDDED_MODELS:
            return False
        path = models_dir(self.hass) / model
        return not await self.hass.async_add_executor_job(path.is_file)
