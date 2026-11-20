from __future__ import annotations

from homeassistant.components.repairs import (
    ConfirmRepairFlow,
    RepairsFlow,
    RepairsFlowResult,
)
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowContext
from homeassistant.helpers import issue_registry as ir

from .config_flow import DownloadSteps
from .const import DOMAIN

ISSUE_LLAMA_CPP_MISSING = "llama_cpp_missing"
ISSUE_MODEL_MISSING = "model_missing"
ISSUE_MODEL_CORRUPT = "model_corrupt"
ISSUE_MODEL_PATH_MISSING = "model_path_missing"
ISSUE_KEYS = (
    ISSUE_LLAMA_CPP_MISSING,
    ISSUE_MODEL_MISSING,
    ISSUE_MODEL_CORRUPT,
    ISSUE_MODEL_PATH_MISSING,
)

DISCARD_MODEL = "discard_model"


def _issue_id(key: str, entry_id: str) -> str:
    return f"{key}_{entry_id}"


@callback
def async_raise_issue(
    hass: HomeAssistant,
    entry_id: str,
    key: str,
    *,
    fixable: bool,
    discard_model: bool = False,
    placeholders: dict[str, str] | None = None,
) -> None:
    data: dict[str, str | int | float | None] = {"entry_id": entry_id}
    if discard_model:
        data[DISCARD_MODEL] = True
    ir.async_create_issue(
        hass,
        DOMAIN,
        _issue_id(key, entry_id),
        data=data,
        is_fixable=fixable,
        severity=ir.IssueSeverity.ERROR,
        translation_key=key,
        translation_placeholders=placeholders,
    )


@callback
def async_clear_issues(hass: HomeAssistant, entry_id: str) -> None:
    for key in ISSUE_KEYS:
        ir.async_delete_issue(hass, DOMAIN, _issue_id(key, entry_id))


class ModelRepairFlow(DownloadSteps[FlowContext, RepairsFlowResult], RepairsFlow):
    def __init__(self, entry_id: str, model: str, *, replace: bool) -> None:
        self._entry_id = entry_id
        self._model = model
        self._replace = replace

    async def async_step_init(
        self, user_input: dict | None = None
    ) -> RepairsFlowResult:
        return await self.async_step_download()

    async def async_step_finish(
        self, user_input: dict | None = None
    ) -> RepairsFlowResult:
        self.hass.config_entries.async_schedule_reload(self._entry_id)
        return self.async_create_entry(data={})


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
) -> RepairsFlow:
    data = data or {}
    entry_id = data.get("entry_id")
    if (
        not isinstance(entry_id, str)
        or (entry := hass.config_entries.async_get_entry(entry_id)) is None
    ):
        return ConfirmRepairFlow()
    return ModelRepairFlow(
        entry_id,
        entry.options[CONF_MODEL],
        replace=bool(data.get(DISCARD_MODEL)),
    )
