from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.repairs.const import DOMAIN as REPAIRS_DOMAIN
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import issue_registry as ir
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso import repairs
from custom_components.sayso.config_flow import _default_embedded_options
from custom_components.sayso.const import (
    BACKEND_EMBEDDED,
    CONF_BACKEND,
    DEFAULT_MODEL_FILENAME,
    DOMAIN,
    EMBEDDED_MODELS,
)
from custom_components.sayso.core.exceptions import (
    SaySoDependencyError,
    SaySoModelLoadError,
    SaySoTimeoutError,
)
from custom_components.sayso.model_store import (
    SaySoModelMissingError,
    async_provision_model,
)

SMALL_MODEL = "SaySo-LFM-v11-F16.gguf"


@pytest.fixture
def no_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("custom_components.sayso.PLATFORMS", [])


@pytest.fixture
def models_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(
        "custom_components.sayso.config_flow.models_dir", lambda _hass: models
    )
    monkeypatch.setattr(
        "custom_components.sayso.model_store.models_dir", lambda _hass: models
    )
    return models


@pytest.fixture
def embedded_runtime(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    engine = MagicMock()
    engine.model_name = DEFAULT_MODEL_FILENAME
    engine.async_start = AsyncMock(return_value=None)
    engine.async_shutdown = AsyncMock(return_value=None)
    monkeypatch.setattr(
        "custom_components.sayso.async_ensure_llama_cpp", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "custom_components.sayso.async_verify_model",
        AsyncMock(return_value=Path("/models/model.gguf")),
    )
    monkeypatch.setattr(
        "custom_components.sayso.EmbeddedEngine", MagicMock(return_value=engine)
    )
    return engine


@pytest.fixture
def schedule_reload(hass: HomeAssistant) -> Iterator[MagicMock]:
    with patch.object(hass.config_entries, "async_schedule_reload") as mocked:
        yield mocked


@pytest.fixture
def provision() -> Iterator[AsyncMock]:
    gate = asyncio.Event()
    outcomes: list[Exception] = []

    async def provision_model(*_args: Any, **_kwargs: Any) -> None:
        await gate.wait()
        if outcomes:
            raise outcomes.pop(0)

    with patch(
        "custom_components.sayso.config_flow.async_provision_model",
        new=AsyncMock(side_effect=provision_model),
    ) as mocked:
        mocked.gate = gate
        mocked.outcomes = outcomes
        yield mocked


async def _finish_download(
    hass: HomeAssistant,
    provision: AsyncMock,
    configure: Callable[[str], Awaitable[Any]],
    flow_id: str,
    error: Exception | None = None,
) -> Any:
    if error is not None:
        provision.outcomes.append(error)
    provision.gate.set()
    await hass.async_block_till_done()
    return await configure(flow_id)


async def _repairs_manager(hass: HomeAssistant) -> Any:
    assert await async_setup_component(hass, DOMAIN, {})
    assert await async_setup_component(hass, REPAIRS_DOMAIN, {})
    return hass.data[REPAIRS_DOMAIN]["flow_manager"]


def _embedded_entry(
    hass: HomeAssistant, *, model: str = DEFAULT_MODEL_FILENAME
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        version=2,
        minor_version=1,
        data={CONF_BACKEND: BACKEND_EMBEDDED},
        options={**_default_embedded_options(), CONF_MODEL: model},
    )
    entry.add_to_hass(hass)
    return entry


def _issues(hass: HomeAssistant) -> dict[str, ir.IssueEntry]:
    return {
        issue.translation_key: issue
        for (domain, _issue_id), issue in ir.async_get(hass).issues.items()
        if domain == DOMAIN
    }


async def _assert_setup_fails(
    hass: HomeAssistant, entry: MockConfigEntry, key: str, *, fixable: bool
) -> None:
    assert await hass.config_entries.async_setup(entry.entry_id) is False
    assert entry.state is ConfigEntryState.SETUP_ERROR
    issue = _issues(hass)[key]
    assert issue.is_fixable is fixable
    assert issue.data is not None
    assert issue.data["entry_id"] == entry.entry_id


async def test_missing_wheel_raises_fixable_issue(
    hass: HomeAssistant, embedded_runtime: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "custom_components.sayso.async_ensure_llama_cpp",
        AsyncMock(side_effect=SaySoDependencyError("no prebuilt wheel")),
    )
    entry = _embedded_entry(hass)

    await _assert_setup_fails(hass, entry, "llama_cpp_missing", fixable=True)


async def test_missing_model_file_raises_fixable_issue(
    hass: HomeAssistant, embedded_runtime: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "custom_components.sayso.async_verify_model",
        AsyncMock(side_effect=SaySoModelMissingError("missing")),
    )
    entry = _embedded_entry(hass)

    await _assert_setup_fails(hass, entry, "model_missing", fixable=True)


async def test_checksum_mismatch_raises_fixable_issue(
    hass: HomeAssistant, embedded_runtime: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "custom_components.sayso.async_verify_model",
        AsyncMock(side_effect=SaySoModelLoadError("checksum mismatch")),
    )
    entry = _embedded_entry(hass)

    await _assert_setup_fails(hass, entry, "model_corrupt", fixable=True)


async def test_gguf_load_failure_raises_fixable_issue_that_discards_model(
    hass: HomeAssistant, embedded_runtime: MagicMock
) -> None:
    embedded_runtime.async_start.side_effect = SaySoModelLoadError("bad gguf")
    entry = _embedded_entry(hass)

    await _assert_setup_fails(hass, entry, "model_corrupt", fixable=True)
    assert _issues(hass)["model_corrupt"].data == {
        "entry_id": entry.entry_id,
        "discard_model": True,
    }


async def test_unreadable_custom_model_path_raises_unfixable_issue(
    hass: HomeAssistant, embedded_runtime: MagicMock, tmp_path: Path
) -> None:
    missing = tmp_path / "missing.gguf"
    entry = _embedded_entry(hass, model=str(missing))

    await _assert_setup_fails(hass, entry, "model_path_missing", fixable=False)
    assert _issues(hass)["model_path_missing"].translation_placeholders == {
        "path": str(missing)
    }


async def test_transient_load_failure_is_not_ready_and_raises_no_issue(
    hass: HomeAssistant, embedded_runtime: MagicMock
) -> None:
    embedded_runtime.async_start.side_effect = SaySoTimeoutError("slow")
    entry = _embedded_entry(hass)

    assert await hass.config_entries.async_setup(entry.entry_id) is False
    assert entry.state is ConfigEntryState.SETUP_RETRY
    assert _issues(hass) == {}


async def test_successful_setup_clears_issues(
    hass: HomeAssistant, embedded_runtime: MagicMock, no_platforms: None
) -> None:
    entry = _embedded_entry(hass)
    repairs.async_raise_issue(hass, entry.entry_id, "model_missing", fixable=True)

    assert await hass.config_entries.async_setup(entry.entry_id) is True
    assert entry.state is ConfigEntryState.LOADED
    assert _issues(hass) == {}


async def test_provision_installs_then_downloads_catalog_model(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    models_root: Path,
) -> None:
    install = AsyncMock(return_value=None)
    download = AsyncMock(return_value=models_root / DEFAULT_MODEL_FILENAME)
    monkeypatch.setattr(
        "custom_components.sayso.model_store.async_ensure_llama_cpp", install
    )
    monkeypatch.setattr(
        "custom_components.sayso.model_store.async_ensure_model", download
    )
    stale = models_root / DEFAULT_MODEL_FILENAME
    stale.write_bytes(b"stale")

    await async_provision_model(hass, DEFAULT_MODEL_FILENAME, replace=True)

    install.assert_awaited_once_with(hass)
    catalog = EMBEDDED_MODELS[DEFAULT_MODEL_FILENAME]
    download.assert_awaited_once_with(
        hass, url=catalog.url, filename=DEFAULT_MODEL_FILENAME, sha256=catalog.sha256
    )
    assert not stale.exists()


async def test_fix_flow_redownloads_model_and_clears_issue(
    hass: HomeAssistant,
    embedded_runtime: MagicMock,
    provision: AsyncMock,
    schedule_reload: MagicMock,
    models_root: Path,
) -> None:
    entry = _embedded_entry(hass)
    flow_manager = await _repairs_manager(hass)
    repairs.async_raise_issue(hass, entry.entry_id, "model_missing", fixable=True)
    issue_id = _issues(hass)["model_missing"].issue_id

    result = await flow_manager.async_init(DOMAIN, data={"issue_id": issue_id})
    assert result["type"] is FlowResultType.SHOW_PROGRESS
    assert result["progress_action"] == "download"
    result = await _finish_download(
        hass, provision, flow_manager.async_configure, result["flow_id"]
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    provision.assert_awaited_once_with(hass, DEFAULT_MODEL_FILENAME, replace=False)
    schedule_reload.assert_called_once_with(entry.entry_id)
    assert _issues(hass) == {}


async def test_fix_flow_discards_corrupt_model_before_download(
    hass: HomeAssistant,
    embedded_runtime: MagicMock,
    provision: AsyncMock,
    schedule_reload: MagicMock,
    models_root: Path,
) -> None:
    entry = _embedded_entry(hass)
    flow_manager = await _repairs_manager(hass)
    repairs.async_raise_issue(
        hass, entry.entry_id, "model_corrupt", fixable=True, discard_model=True
    )
    issue_id = _issues(hass)["model_corrupt"].issue_id

    result = await flow_manager.async_init(DOMAIN, data={"issue_id": issue_id})
    result = await _finish_download(
        hass, provision, flow_manager.async_configure, result["flow_id"]
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    provision.assert_awaited_once_with(hass, DEFAULT_MODEL_FILENAME, replace=True)
    assert _issues(hass) == {}


async def test_fix_flow_failure_keeps_issue_and_offers_retry(
    hass: HomeAssistant,
    embedded_runtime: MagicMock,
    provision: AsyncMock,
    schedule_reload: MagicMock,
    models_root: Path,
) -> None:
    entry = _embedded_entry(hass)
    flow_manager = await _repairs_manager(hass)
    repairs.async_raise_issue(hass, entry.entry_id, "model_missing", fixable=True)
    issue_id = _issues(hass)["model_missing"].issue_id

    result = await flow_manager.async_init(DOMAIN, data={"issue_id": issue_id})
    result = await _finish_download(
        hass,
        provision,
        flow_manager.async_configure,
        result["flow_id"],
        SaySoModelLoadError("network down"),
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "download_failed"
    assert result["errors"] == {"base": "download_failed"}
    assert "model_missing" in _issues(hass)
    schedule_reload.assert_not_called()


async def test_config_flow_downloads_before_creating_entry(
    hass: HomeAssistant,
    embedded_runtime: MagicMock,
    provision: AsyncMock,
    no_platforms: None,
    models_root: Path,
) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})

    assert result["type"] is FlowResultType.SHOW_PROGRESS
    assert result["progress_action"] == "download"
    assert hass.config_entries.async_entries(DOMAIN) == []

    result = await _finish_download(
        hass, provision, hass.config_entries.flow.async_configure, result["flow_id"]
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "SaySo (local)"
    provision.assert_awaited_once_with(hass, DEFAULT_MODEL_FILENAME, replace=False)


async def test_config_flow_failure_creates_no_entry_and_retries(
    hass: HomeAssistant,
    embedded_runtime: MagicMock,
    provision: AsyncMock,
    no_platforms: None,
    models_root: Path,
) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    result = await _finish_download(
        hass,
        provision,
        hass.config_entries.flow.async_configure,
        result["flow_id"],
        SaySoDependencyError("no wheel"),
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "download_failed"
    assert result["errors"] == {"base": "install_failed"}
    assert hass.config_entries.async_entries(DOMAIN) == []

    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1


async def test_options_flow_model_download_runs_progress_then_saves(
    hass: HomeAssistant,
    embedded_runtime: MagicMock,
    provision: AsyncMock,
    schedule_reload: MagicMock,
    models_root: Path,
) -> None:
    entry = _embedded_entry(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    user_input = {**entry.options, CONF_MODEL: SMALL_MODEL}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input
    )

    assert result["type"] is FlowResultType.SHOW_PROGRESS
    assert result["progress_action"] == "download"
    result = await _finish_download(
        hass, provision, hass.config_entries.options.async_configure, result["flow_id"]
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    provision.assert_awaited_once_with(hass, SMALL_MODEL, replace=False)
    assert entry.options[CONF_MODEL] == SMALL_MODEL


async def test_options_flow_failure_keeps_saved_model(
    hass: HomeAssistant,
    embedded_runtime: MagicMock,
    provision: AsyncMock,
    schedule_reload: MagicMock,
    models_root: Path,
) -> None:
    entry = _embedded_entry(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    user_input = {**entry.options, CONF_MODEL: SMALL_MODEL}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input
    )
    result = await _finish_download(
        hass,
        provision,
        hass.config_entries.options.async_configure,
        result["flow_id"],
        SaySoModelLoadError("download failed"),
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "download_failed"
    assert result["errors"] == {"base": "download_failed"}
    assert entry.options[CONF_MODEL] == DEFAULT_MODEL_FILENAME


async def test_options_flow_skips_progress_when_model_is_on_disk(
    hass: HomeAssistant,
    embedded_runtime: MagicMock,
    provision: AsyncMock,
    schedule_reload: MagicMock,
    models_root: Path,
) -> None:
    (models_root / SMALL_MODEL).write_bytes(b"gguf")
    entry = _embedded_entry(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    user_input = {**entry.options, CONF_MODEL: SMALL_MODEL}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], user_input
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    provision.assert_not_awaited()
    assert entry.options[CONF_MODEL] == SMALL_MODEL
