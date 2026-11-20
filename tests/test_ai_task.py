from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import voluptuous as vol
from homeassistant.components import ai_task
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.setup import async_setup_component

from custom_components.sayso.client import ChatCompletionResult, LlamaCppClient
from custom_components.sayso.const import AI_TASK_SUBENTRY_TYPE, DOMAIN
from custom_components.sayso.inference import ExternalEngine
from tests.test_config_flow import MODEL_ID
from tests.test_subentry_flows import _create_external_entry


@pytest.fixture(autouse=True)
async def setup_llm(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "llm", {})


@pytest.fixture
def mock_llama_client() -> Any:
    with (
        patch.object(
            LlamaCppClient,
            "list_models",
            new=AsyncMock(return_value=[MODEL_ID]),
        ),
        patch.object(
            LlamaCppClient,
            "validate_model",
            new=AsyncMock(return_value=None),
        ),
    ):
        yield


@pytest.fixture
def mock_chat_completion() -> Any:
    with patch.object(
        ExternalEngine,
        "async_chat_completion",
        new=AsyncMock(return_value=ChatCompletionResult(content="", tool_calls=[])),
    ) as mock:
        yield mock


async def _setup_entry_with_tasks(hass: HomeAssistant, names: list[str]) -> str:
    entry = await _create_external_entry(hass)
    for name in names:
        result = await hass.config_entries.subentries.async_init(
            (entry.entry_id, AI_TASK_SUBENTRY_TYPE),
            context={"source": "user"},
        )
        result = await hass.config_entries.subentries.async_configure(
            result["flow_id"], {CONF_NAME: name}
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()
    return entry.entry_id


def _task_subentry_ids(hass: HomeAssistant, entry_id: str) -> list[str]:
    entry = hass.config_entries.async_get_entry(entry_id)
    assert entry is not None
    return [
        subentry.subentry_id
        for subentry in entry.subentries.values()
        if subentry.subentry_type == AI_TASK_SUBENTRY_TYPE
    ]


def _ai_task_entity_id(hass: HomeAssistant, subentry_id: str) -> str:
    entity_id = er.async_get(hass).async_get_entity_id(
        ai_task.DOMAIN, DOMAIN, subentry_id
    )
    assert entity_id is not None
    return entity_id


def _reply(mock_chat_completion: Any, content: str) -> None:
    mock_chat_completion.return_value = ChatCompletionResult(
        content=content, tool_calls=[]
    )


@pytest.mark.usefixtures("mock_llama_client")
async def test_one_entity_per_ai_task_subentry(hass: HomeAssistant) -> None:
    entry_id = await _setup_entry_with_tasks(hass, ["Summaries", "Shopping"])

    subentry_ids = _task_subentry_ids(hass, entry_id)
    assert len(subentry_ids) == 2
    registry = er.async_get(hass)
    devices = dr.async_get(hass)
    for subentry_id in subentry_ids:
        row = registry.async_get(_ai_task_entity_id(hass, subentry_id))
        assert row is not None
        assert row.platform == DOMAIN
        assert row.unique_id == subentry_id
        assert row.config_entry_id == entry_id
        assert row.config_subentry_id == subentry_id
        device = devices.async_get_device(identifiers={(DOMAIN, subentry_id)})
        assert device is not None
        assert row.device_id == device.id
        assert device.entry_type == dr.DeviceEntryType.SERVICE


@pytest.mark.usefixtures("mock_llama_client")
async def test_text_generation_returns_model_text(
    hass: HomeAssistant, mock_chat_completion: Any
) -> None:
    entry_id = await _setup_entry_with_tasks(hass, ["Summaries"])
    entity_id = _ai_task_entity_id(hass, _task_subentry_ids(hass, entry_id)[0])
    _reply(mock_chat_completion, "Two lights are on.")

    result = await ai_task.async_generate_data(
        hass,
        task_name="summarize",
        entity_id=entity_id,
        instructions="Summarize the lights.",
    )

    assert result.data == "Two lights are on."
    assert result.conversation_id


@pytest.mark.usefixtures("mock_llama_client")
async def test_structured_output_is_validated_and_returned(
    hass: HomeAssistant, mock_chat_completion: Any
) -> None:
    entry_id = await _setup_entry_with_tasks(hass, ["Summaries"])
    entity_id = _ai_task_entity_id(hass, _task_subentry_ids(hass, entry_id)[0])
    _reply(mock_chat_completion, '{"answer": "yes"}')

    result = await ai_task.async_generate_data(
        hass,
        task_name="answer",
        entity_id=entity_id,
        instructions="Are the lights on?",
        structure=vol.Schema({vol.Required("answer"): str}),
    )

    assert result.data == {"answer": "yes"}


@pytest.mark.usefixtures("mock_llama_client")
async def test_invalid_json_raises_home_assistant_error(
    hass: HomeAssistant, mock_chat_completion: Any
) -> None:
    entry_id = await _setup_entry_with_tasks(hass, ["Summaries"])
    entity_id = _ai_task_entity_id(hass, _task_subentry_ids(hass, entry_id)[0])
    _reply(mock_chat_completion, "Yes, the lights are on.")

    with pytest.raises(HomeAssistantError, match="not valid JSON"):
        await ai_task.async_generate_data(
            hass,
            task_name="answer",
            entity_id=entity_id,
            instructions="Are the lights on?",
            structure=vol.Schema({vol.Required("answer"): str}),
        )


@pytest.mark.usefixtures("mock_llama_client")
async def test_schema_failure_raises_home_assistant_error(
    hass: HomeAssistant, mock_chat_completion: Any
) -> None:
    entry_id = await _setup_entry_with_tasks(hass, ["Summaries"])
    entity_id = _ai_task_entity_id(hass, _task_subentry_ids(hass, entry_id)[0])
    _reply(mock_chat_completion, '{"answer": 3}')

    with pytest.raises(
        HomeAssistantError, match="does not match the requested structure"
    ):
        await ai_task.async_generate_data(
            hass,
            task_name="answer",
            entity_id=entity_id,
            instructions="Are the lights on?",
            structure=vol.Schema({vol.Required("answer"): str}),
        )


@pytest.mark.usefixtures("mock_llama_client")
async def test_generation_uses_shared_engine_without_tools(
    hass: HomeAssistant, mock_chat_completion: Any
) -> None:
    entry_id = await _setup_entry_with_tasks(hass, ["Summaries"])
    entity_id = _ai_task_entity_id(hass, _task_subentry_ids(hass, entry_id)[0])
    _reply(mock_chat_completion, '{"answer": "yes"}')

    await ai_task.async_generate_data(
        hass,
        task_name="answer",
        entity_id=entity_id,
        instructions="Are the lights on?",
        structure=vol.Schema({vol.Required("answer"): str}),
    )

    mock_chat_completion.assert_awaited_once()
    args, kwargs = mock_chat_completion.await_args
    assert kwargs.get("tools") is None
    assert kwargs.get("grammar") is None
    messages = args[0]
    assert messages[-2] == {"role": "user", "content": "Are the lights on?"}
    assert messages[-1]["role"] == "system"
    assert "JSON" in messages[-1]["content"]
