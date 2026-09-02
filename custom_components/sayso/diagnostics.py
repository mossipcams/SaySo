"""Diagnostics support for SaySo."""

from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_API_KEY, CONF_URL
from homeassistant.core import HomeAssistant

from . import SaySoConfigEntry
from .exceptions import SaySoError

TO_REDACT = {CONF_API_KEY}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: SaySoConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    connectivity: dict[str, Any] = {
        "api_key_configured": bool(entry.data.get(CONF_API_KEY)),
        "reachable": False,
        "models": [],
        "error": None,
    }

    if entry.runtime_data is not None:
        client = entry.runtime_data.client
        runtime = entry.runtime_data
        connectivity["base_url"] = client.base_url
        connectivity["chat_completions_url"] = client.chat_completions_url
        connectivity["models_url"] = client.models_url
        connectivity["timeout_seconds"] = client._timeout
        try:
            models = await client.list_models()
            connectivity["reachable"] = True
            connectivity["models"] = models
        except SaySoError as err:
            connectivity["error"] = type(err).__name__
        runtime_data: dict[str, Any] = {
            "loaded": True,
            "model": runtime.model,
            "llm_hass_api": runtime.llm_api,
            "temperature": runtime.temperature,
            "max_output_tokens": runtime.max_output_tokens,
            "max_tool_iterations": runtime.max_tool_iterations,
            "system_prompt_length": len(runtime.system_prompt),
        }
    else:
        connectivity["base_url"] = entry.data.get(CONF_URL)
        runtime_data = {"loaded": False}

    return {
        "entry": {
            "title": entry.title,
            "unique_id": entry.unique_id,
            "data": async_redact_data(entry.data, TO_REDACT),
            "options": dict(entry.options),
        },
        "runtime": runtime_data,
        "connectivity": connectivity,
    }
