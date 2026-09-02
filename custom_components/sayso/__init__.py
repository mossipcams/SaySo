"""The SaySo integration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_API_KEY,
    CONF_LLM_HASS_API,
    CONF_MODEL,
    CONF_URL,
    Platform,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import LLM_API_ASSIST

from .client import LlamaCppClient
from .const import (
    CONF_MAX_OUTPUT_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_TOOL_ITERATIONS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DOMAIN,
)
from .exceptions import SaySoError

PLATFORMS: list[Platform] = [Platform.CONVERSATION]

type SaySoConfigEntry = ConfigEntry[SaySoRuntimeData]


@dataclass
class SaySoRuntimeData:
    """Runtime data stored on the config entry."""

    client: LlamaCppClient
    model: str
    llm_api: str
    system_prompt: str
    temperature: float
    max_output_tokens: int
    max_tool_iterations: int


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the SaySo integration."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    """Set up SaySo from a config entry."""
    options = entry.options
    client = LlamaCppClient.from_hass(
        hass,
        entry.data[CONF_URL],
        api_key=entry.data.get(CONF_API_KEY),
        timeout=options.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
    )

    try:
        await client.validate_model(options[CONF_MODEL])
    except SaySoError:
        # Allow setup so options can be corrected without removing the entry.
        pass

    entry.runtime_data = SaySoRuntimeData(
        client=client,
        model=options[CONF_MODEL],
        llm_api=options.get(CONF_LLM_HASS_API, LLM_API_ASSIST),
        system_prompt=options.get(CONF_PROMPT, DEFAULT_SYSTEM_PROMPT),
        temperature=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
        max_output_tokens=options.get(CONF_MAX_OUTPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS),
        max_tool_iterations=options.get(
            CONF_MAX_TOOL_ITERATIONS, DEFAULT_MAX_TOOL_ITERATIONS
        ),
    )

    if PLATFORMS:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: SaySoConfigEntry) -> bool:
    """Unload a SaySo config entry."""
    if PLATFORMS:
        if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
            return False
    entry.runtime_data = None  # type: ignore[assignment]
    return True
