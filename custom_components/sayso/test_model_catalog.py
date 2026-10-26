from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from homeassistant.const import CONF_MODEL

from custom_components.sayso import _async_build_engine, _synced_embedded_options
from custom_components.sayso.config_flow import _model_fields
from custom_components.sayso.const import (
    BACKEND_EMBEDDED,
    CONF_BACKEND,
    CONF_MODEL_PATH,
    CONF_N_CTX,
    CONF_N_THREADS,
    CONF_TIMEOUT,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_MODEL_SHA256,
    DEFAULT_MODEL_URL,
    EMBEDDED_MODELS,
)

RELEASE_PREFIX = "https://github.com/mossipcams/SaySo/releases/download/"


def _entry(data: dict[str, Any], options: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(data=data, options=options)


def test_default_model_is_the_default_catalog_entry() -> None:
    assert EMBEDDED_MODELS[DEFAULT_MODEL_FILENAME].url == DEFAULT_MODEL_URL
    assert EMBEDDED_MODELS[DEFAULT_MODEL_FILENAME].sha256 == DEFAULT_MODEL_SHA256


@pytest.mark.parametrize("filename", list(EMBEDDED_MODELS))
def test_catalog_entries_are_pinned_release_assets(filename: str) -> None:
    model = EMBEDDED_MODELS[filename]

    assert model.url.startswith(RELEASE_PREFIX)
    assert model.url.endswith(f"/{filename}")
    assert model.sha256 is not None
    assert re.fullmatch(r"[0-9a-f]{64}", model.sha256)


@pytest.mark.parametrize("filename", list(EMBEDDED_MODELS))
def test_sync_keeps_catalog_choice(filename: str) -> None:
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        {CONF_MODEL: filename, CONF_MODEL_PATH: ""},
    )

    assert _synced_embedded_options(entry) is None


def test_embedded_options_offer_catalog_and_suggest_stored_choice() -> None:
    fields = _model_fields(
        {CONF_MODEL: "SaySo-LFM-v7-F16.gguf", CONF_MODEL_PATH: ""},
        [],
        embedded=True,
    )
    marker = next(key for key in fields if key == CONF_MODEL)
    selector = fields[marker]

    assert marker.description == {"suggested_value": "SaySo-LFM-v7-F16.gguf"}
    assert [option["value"] for option in selector.config["options"]] == list(
        EMBEDDED_MODELS
    )
    assert [option["label"] for option in selector.config["options"]] == [
        model.label for model in EMBEDDED_MODELS.values()
    ]


def test_embedded_options_suggest_default_when_nothing_stored() -> None:
    fields = _model_fields({CONF_MODEL_PATH: ""}, [], embedded=True)
    marker = next(key for key in fields if key == CONF_MODEL)

    assert marker.description == {"suggested_value": DEFAULT_MODEL_FILENAME}


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    calls: dict[str, Any] = {"downloads": [], "engines": []}

    async def fake_ensure_llama_cpp(hass: object) -> None:
        return None

    async def fake_ensure_model(
        hass: object, *, url: str, filename: str, sha256: str | None
    ) -> Path:
        calls["downloads"].append((url, filename, sha256))
        return tmp_path / filename

    class FakeEngine:
        def __init__(self, model_path: Path, **kwargs: Any) -> None:
            calls["engines"].append(model_path)
            self.model_path = model_path

        async def async_start(self) -> None:
            return None

        async def async_shutdown(self) -> None:
            return None

    monkeypatch.setattr("custom_components.sayso.async_ensure_llama_cpp", fake_ensure_llama_cpp)
    monkeypatch.setattr("custom_components.sayso.async_ensure_model", fake_ensure_model)
    monkeypatch.setattr("custom_components.sayso.EmbeddedEngine", FakeEngine)
    return calls


@pytest.mark.parametrize("filename", list(EMBEDDED_MODELS))
async def test_selected_model_is_downloaded_and_loaded(
    fake_runtime: dict[str, Any], tmp_path: Path, filename: str
) -> None:
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        {
            CONF_MODEL: filename,
            CONF_MODEL_PATH: "",
            CONF_N_CTX: 4096,
            CONF_N_THREADS: 2,
            CONF_TIMEOUT: 30,
        },
    )

    engine, client = await _async_build_engine(SimpleNamespace(), entry)

    model = EMBEDDED_MODELS[filename]
    assert fake_runtime["downloads"] == [(model.url, filename, model.sha256)]
    assert fake_runtime["engines"] == [tmp_path / filename]
    assert engine.model_path == tmp_path / filename
    assert client is None


async def test_custom_model_path_skips_catalog_download(
    fake_runtime: dict[str, Any],
) -> None:
    custom = "/config/custom.gguf"
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        {
            CONF_MODEL: "SaySo-LFM-v7-F16.gguf",
            CONF_MODEL_PATH: custom,
            CONF_N_CTX: 4096,
            CONF_N_THREADS: 2,
            CONF_TIMEOUT: 30,
        },
    )

    await _async_build_engine(SimpleNamespace(), entry)

    assert fake_runtime["downloads"] == []
    assert fake_runtime["engines"] == [Path(custom)]

