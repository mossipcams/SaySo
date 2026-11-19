from __future__ import annotations

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso import async_migrate_entry
from custom_components.sayso.const import (
    BACKEND_EMBEDDED,
    CONF_BACKEND,
    CONF_MODEL,
    CONF_MODEL_PATH,
    DEFAULT_MODEL_FILENAME,
)
from custom_components.sayso.model_store import models_dir


def _entry(
    hass: HomeAssistant, *, version: int = 1, minor_version: int = 1, options: dict
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain="sayso",
        version=version,
        minor_version=minor_version,
        data={CONF_BACKEND: BACKEND_EMBEDDED},
        options=options,
    )
    entry.add_to_hass(hass)
    return entry


async def test_migrates_legacy_model_path_in_catalog(hass: HomeAssistant) -> None:
    legacy_path = str(models_dir(hass) / DEFAULT_MODEL_FILENAME)
    entry = _entry(
        hass, options={CONF_MODEL_PATH: legacy_path}
    )

    assert await async_migrate_entry(hass, entry) is True

    assert entry.minor_version == 2
    assert entry.options[CONF_MODEL] == DEFAULT_MODEL_FILENAME
    assert entry.options[CONF_MODEL_PATH] == ""


async def test_migrates_legacy_absolute_model_path(hass: HomeAssistant) -> None:
    legacy_path = "/abs/models/custom.gguf"
    entry = _entry(hass, options={CONF_MODEL_PATH: legacy_path})

    assert await async_migrate_entry(hass, entry) is True

    assert entry.minor_version == 2
    assert entry.options[CONF_MODEL] == legacy_path
    assert entry.options[CONF_MODEL_PATH] == ""


async def test_minor_version_two_options_untouched(hass: HomeAssistant) -> None:
    options = {CONF_MODEL: DEFAULT_MODEL_FILENAME}
    entry = _entry(hass, minor_version=2, options=options)

    assert await async_migrate_entry(hass, entry) is True

    assert entry.minor_version == 2
    assert dict(entry.options) == options


async def test_future_version_fails_migration(hass: HomeAssistant) -> None:
    entry = _entry(hass, version=2, minor_version=1, options={})

    assert await async_migrate_entry(hass, entry) is False
