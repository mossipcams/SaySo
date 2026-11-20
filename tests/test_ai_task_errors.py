from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voluptuous as vol
from homeassistant.components import ai_task
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.setup import async_setup_component

from custom_components.sayso.client import ChatCompletionResult, LlamaCppClient
from custom_components.sayso.const import DOMAIN
from custom_components.sayso.core.exceptions import (
    SaySoConnectionError,
    SaySoTimeoutError,
)
from custom_components.sayso.inference import ExternalEngine
from tests.test_ai_task import (
    _ai_task_entity_id,
    _setup_entry_with_tasks,
    _task_subentry_ids,
)
from tests.test_config_flow import MODEL_ID


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


async def _task_entity_id(hass: HomeAssistant) -> str:
    entry_id = await _setup_entry_with_tasks(hass, ["Shopping"])
    return _ai_task_entity_id(hass, _task_subentry_ids(hass, entry_id)[0])


@pytest.mark.usefixtures("mock_llama_client")
async def test_json_array_reply_is_returned_as_data(hass: HomeAssistant) -> None:
    entity_id = await _task_entity_id(hass)
    client_reply = AsyncMock(
        return_value=ChatCompletionResult(content='["milk", "eggs"]', tool_calls=[])
    )

    with patch.object(LlamaCppClient, "chat_completion", new=client_reply):
        result = await ai_task.async_generate_data(
            hass,
            task_name="shopping",
            entity_id=entity_id,
            instructions="List the items to buy.",
            structure=vol.Schema([str]),
        )

    assert result.data == ["milk", "eggs"]
    assert client_reply.await_args.kwargs.get("tools") is None


@pytest.mark.usefixtures("mock_llama_client")
async def test_engine_timeout_surfaces_as_translated_error(
    hass: HomeAssistant,
) -> None:
    entity_id = await _task_entity_id(hass)

    with (
        patch.object(
            ExternalEngine,
            "async_chat_completion",
            new=AsyncMock(side_effect=SaySoTimeoutError("Request timed out")),
        ),
        pytest.raises(HomeAssistantError) as raised,
    ):
        await ai_task.async_generate_data(
            hass,
            task_name="summarize",
            entity_id=entity_id,
            instructions="Summarize the lights.",
        )

    assert raised.value.translation_domain == DOMAIN
    assert raised.value.translation_key == "ai_task_failed"
    assert raised.value.translation_placeholders == {"error": "Request timed out"}
    assert isinstance(raised.value.__cause__, SaySoTimeoutError)


@pytest.mark.usefixtures("mock_llama_client")
async def test_engine_connection_error_surfaces_as_translated_error(
    hass: HomeAssistant,
) -> None:
    entity_id = await _task_entity_id(hass)

    with (
        patch.object(
            ExternalEngine,
            "async_chat_completion",
            new=AsyncMock(side_effect=SaySoConnectionError("Connection refused")),
        ),
        pytest.raises(HomeAssistantError) as raised,
    ):
        await ai_task.async_generate_data(
            hass,
            task_name="summarize",
            entity_id=entity_id,
            instructions="Summarize the lights.",
        )

    assert raised.value.translation_key == "ai_task_failed"
    assert raised.value.translation_placeholders == {"error": "Connection refused"}
    assert isinstance(raised.value.__cause__, SaySoConnectionError)


async def test_lfm_content_call_is_extracted_when_tools_are_offered() -> None:
    client = MagicMock()
    client.chat_completion = AsyncMock(
        return_value=ChatCompletionResult(
            content="[HassTurnOn(name='lamp')]", tool_calls=[]
        )
    )
    engine = ExternalEngine(client, "test-model")
    tools = [
        {
            "type": "function",
            "function": {"name": "HassTurnOn", "parameters": {"type": "object"}},
        }
    ]

    result = await engine.async_chat_completion(
        [{"role": "user", "content": "turn on the lamp"}], tools=tools
    )

    assert result.content is None
    assert [(call.name, call.arguments) for call in result.tool_calls] == [
        ("HassTurnOn", {"name": "lamp"})
    ]
