from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from homeassistant.const import CONF_MODEL, CONF_URL

from custom_components.sayso import _synced_embedded_options
from custom_components.sayso.const import (
    BACKEND_EMBEDDED,
    BACKEND_EXTERNAL,
    CONF_BACKEND,
    CONF_MODEL_PATH,
    CONF_N_CTX,
    DEFAULT_MODEL_FILENAME,
)

STALE_MODEL = "SaySo-LFM-v5-F16.gguf"


def _entry(data: dict[str, Any], options: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(data=data, options=options)


def test_stale_embedded_model_is_synced_to_default() -> None:
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        {CONF_MODEL: STALE_MODEL, CONF_MODEL_PATH: "", CONF_N_CTX: 4096},
    )

    assert _synced_embedded_options(entry) == {
        CONF_MODEL: DEFAULT_MODEL_FILENAME,
        CONF_MODEL_PATH: "",
        CONF_N_CTX: 4096,
    }


def test_current_embedded_model_is_not_updated() -> None:
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        {CONF_MODEL: DEFAULT_MODEL_FILENAME, CONF_MODEL_PATH: ""},
    )

    assert _synced_embedded_options(entry) is None


def test_embedded_custom_model_path_is_left_alone() -> None:
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        {CONF_MODEL: STALE_MODEL, CONF_MODEL_PATH: "/config/custom.gguf"},
    )

    assert _synced_embedded_options(entry) is None


def test_external_entry_is_left_alone() -> None:
    entry = _entry(
        {CONF_BACKEND: BACKEND_EXTERNAL, CONF_URL: "http://127.0.0.1:8080/v1"},
        {CONF_MODEL: "my-alias", CONF_MODEL_PATH: ""},
    )

    assert _synced_embedded_options(entry) is None
