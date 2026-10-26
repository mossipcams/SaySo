from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from homeassistant.const import CONF_MODEL
from homeassistant.exceptions import ConfigEntryNotReady

from custom_components.sayso import (
    _async_build_engine,
    _migrated_embedded_options,
    _synced_embedded_options,
)
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
V7_FILENAME = "SaySo-LFM-v7-F16.gguf"
MODELS_DIR = "/config/sayso/models"


class _FakeHass:
    config = SimpleNamespace(path=lambda subdir: f"/config/{subdir}")

    async def async_add_executor_job(self, func: Any, *args: Any) -> Any:
        return func(*args)


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
    entry = _entry({CONF_BACKEND: BACKEND_EMBEDDED}, {CONF_MODEL: filename})

    assert _synced_embedded_options(entry) is None


def test_sync_keeps_absolute_model_path() -> None:
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        {CONF_MODEL: f"{MODELS_DIR}/custom.gguf"},
    )

    assert _synced_embedded_options(entry) is None


def test_user_v7_entry_migrates_to_catalog_key() -> None:
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        {
            CONF_MODEL: V7_FILENAME,
            CONF_MODEL_PATH: f"{MODELS_DIR}/{V7_FILENAME}",
            CONF_N_CTX: 4096,
        },
    )

    migrated = _migrated_embedded_options(_FakeHass(), entry)

    assert migrated == {CONF_MODEL: V7_FILENAME, CONF_MODEL_PATH: "", CONF_N_CTX: 4096}
    stable = _entry({CONF_BACKEND: BACKEND_EMBEDDED}, migrated)
    assert _migrated_embedded_options(_FakeHass(), stable) is None
    assert _synced_embedded_options(stable) is None


def test_non_catalog_file_in_models_dir_migrates_to_path() -> None:
    path = f"{MODELS_DIR}/SaySo-LFM-v5-F16.gguf"
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        {CONF_MODEL: DEFAULT_MODEL_FILENAME, CONF_MODEL_PATH: path},
    )

    assert _migrated_embedded_options(_FakeHass(), entry) == {
        CONF_MODEL: path,
        CONF_MODEL_PATH: "",
    }


def test_path_outside_models_dir_migrates_to_path() -> None:
    path = f"/media/gguf/{V7_FILENAME}"
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        {CONF_MODEL: DEFAULT_MODEL_FILENAME, CONF_MODEL_PATH: path},
    )

    assert _migrated_embedded_options(_FakeHass(), entry) == {
        CONF_MODEL: path,
        CONF_MODEL_PATH: "",
    }


def test_entry_without_legacy_path_is_not_migrated() -> None:
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        {CONF_MODEL: V7_FILENAME, CONF_MODEL_PATH: ""},
    )

    assert _migrated_embedded_options(_FakeHass(), entry) is None


def test_embedded_form_has_one_model_field_with_catalog_first() -> None:
    local = [f"{MODELS_DIR}/{V7_FILENAME}", f"{MODELS_DIR}/custom.gguf"]
    fields = _model_fields({CONF_MODEL: DEFAULT_MODEL_FILENAME}, local, embedded=True)

    assert all(key != CONF_MODEL_PATH for key in fields)
    model_keys = [key for key in fields if key == CONF_MODEL]
    assert len(model_keys) == 1
    selector = fields[model_keys[0]]
    assert selector.config["custom_value"] is True
    options = selector.config["options"]
    assert [option["value"] for option in options] == [*EMBEDDED_MODELS, local[1]]
    assert [option["label"] for option in options[: len(EMBEDDED_MODELS)]] == [
        model.label for model in EMBEDDED_MODELS.values()
    ]
    assert options[-1]["label"] == "custom.gguf"


def test_embedded_form_suggests_stored_choice() -> None:
    fields = _model_fields({CONF_MODEL: V7_FILENAME}, [], embedded=True)
    marker = next(key for key in fields if key == CONF_MODEL)

    assert marker.description == {"suggested_value": V7_FILENAME}


def test_embedded_form_suggests_default_when_nothing_stored() -> None:
    fields = _model_fields({}, [], embedded=True)
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


def _runtime_options(model: str) -> dict[str, Any]:
    return {
        CONF_MODEL: model,
        CONF_N_CTX: 4096,
        CONF_N_THREADS: 2,
        CONF_TIMEOUT: 30,
    }


@pytest.mark.parametrize("filename", list(EMBEDDED_MODELS))
async def test_catalog_model_is_downloaded_and_loaded(
    fake_runtime: dict[str, Any], tmp_path: Path, filename: str
) -> None:
    entry = _entry({CONF_BACKEND: BACKEND_EMBEDDED}, _runtime_options(filename))

    engine, client = await _async_build_engine(_FakeHass(), entry)

    model = EMBEDDED_MODELS[filename]
    assert fake_runtime["downloads"] == [(model.url, filename, model.sha256)]
    assert fake_runtime["engines"] == [tmp_path / filename]
    assert engine.model_path == tmp_path / filename
    assert client is None


async def test_model_path_is_loaded_without_download(
    fake_runtime: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "custom.gguf"
    path.write_bytes(b"gguf")
    entry = _entry({CONF_BACKEND: BACKEND_EMBEDDED}, _runtime_options(str(path)))

    await _async_build_engine(_FakeHass(), entry)

    assert fake_runtime["downloads"] == []
    assert fake_runtime["engines"] == [path]


async def test_missing_model_path_is_not_ready(
    fake_runtime: dict[str, Any], tmp_path: Path
) -> None:
    entry = _entry(
        {CONF_BACKEND: BACKEND_EMBEDDED},
        _runtime_options(str(tmp_path / "missing.gguf")),
    )

    with pytest.raises(ConfigEntryNotReady):
        await _async_build_engine(_FakeHass(), entry)

    assert fake_runtime["engines"] == []
