from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, CONF_MODEL, CONF_NAME
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.llm import LLM_API_ASSIST
from homeassistant.setup import async_setup_component

from custom_components.sayso.client import ChatCompletionResult, LlamaCppClient
from custom_components.sayso.config_flow import SaySoConfigFlow
from custom_components.sayso.const import (
    AI_TASK_SUBENTRY_TYPE,
    BACKEND_EMBEDDED,
    CONF_BACKEND,
    CONF_DUCK_VOLUME,
    CONF_GRAMMAR_DECODING,
    CONF_MAX_OUTPUT_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    CONF_TRACE_MAX_INTERACTIONS,
    CONF_TRACE_RETENTION_DAYS,
    CONF_TRACE_STORE_UTTERANCES,
    CONVERSATION_SUBENTRY_TYPE,
    DEFAULT_TEMPERATURE,
    DOMAIN,
)
from custom_components.sayso.core.exceptions import SaySoDependencyError
from custom_components.sayso.diagnostics import async_get_config_entry_diagnostics
from tests.test_config_flow import (
    BASE_URL,
    MODEL_ID,
    _complete_model_step,
    _complete_user_step,
    _start_user_step,
)

SECOND_MODEL = "second-model"


@pytest.fixture(autouse=True)
async def setup_llm(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "llm", {})


@pytest.fixture
def mock_llama_client() -> Any:
    with (
        patch.object(
            LlamaCppClient,
            "list_models",
            new=AsyncMock(return_value=[MODEL_ID, SECOND_MODEL]),
        ),
        patch.object(
            LlamaCppClient,
            "validate_model",
            new=AsyncMock(return_value=None),
        ),
    ):
        yield


async def _create_external_entry(hass: HomeAssistant) -> ConfigEntry:
    result = await _start_user_step(hass)
    result = await _complete_user_step(hass, result["flow_id"])
    result = await _complete_model_step(hass, result["flow_id"])
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    return hass.config_entries.async_entries(DOMAIN)[0]


async def _create_embedded_entry(hass: HomeAssistant) -> ConfigEntry:
    with (
        patch(
            "custom_components.sayso.config_flow.async_provision_model",
            new=AsyncMock(),
        ),
        patch(
            "custom_components.sayso.async_ensure_llama_cpp",
            new=AsyncMock(side_effect=SaySoDependencyError("runtime unavailable")),
        ),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": "user"}
        )
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
        assert result["type"] == FlowResultType.CREATE_ENTRY
        await hass.async_block_till_done()
    return hass.config_entries.async_entries(DOMAIN)[0]


def _conversation_entity_id(hass: HomeAssistant, subentry_id: str) -> str | None:
    return er.async_get(hass).async_get_entity_id(
        conversation.DOMAIN, DOMAIN, subentry_id
    )


def _conversation_form_input(name: str, model: str) -> dict[str, Any]:
    return {
        CONF_NAME: name,
        CONF_MODEL: model,
        CONF_LLM_HASS_API: LLM_API_ASSIST,
        CONF_PROMPT: "Keep answers short.",
        CONF_TEMPERATURE: 0.3,
        CONF_MAX_OUTPUT_TOKENS: 200,
        CONF_MAX_TOOL_ITERATIONS: 3,
    }


async def test_supported_subentry_types(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)

    supported = SaySoConfigFlow.async_get_supported_subentry_types(entry)

    assert set(supported) == {CONVERSATION_SUBENTRY_TYPE, AI_TASK_SUBENTRY_TYPE}


async def test_external_install_creates_default_conversation_subentry(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)

    [subentry] = entry.subentries.values()
    assert subentry.subentry_type == CONVERSATION_SUBENTRY_TYPE
    assert subentry.title == entry.title
    assert subentry.unique_id is None
    assert subentry.data[CONF_MODEL] == MODEL_ID
    assert subentry.data[CONF_LLM_HASS_API] == LLM_API_ASSIST
    assert subentry.data[CONF_TEMPERATURE] == DEFAULT_TEMPERATURE
    assert CONF_MODEL not in entry.options
    assert CONF_TEMPERATURE not in entry.options


async def test_embedded_install_creates_default_conversation_subentry(
    hass: HomeAssistant,
) -> None:
    entry = await _create_embedded_entry(hass)

    assert entry.data[CONF_BACKEND] == BACKEND_EMBEDDED
    [subentry] = entry.subentries.values()
    assert subentry.subentry_type == CONVERSATION_SUBENTRY_TYPE
    assert subentry.title == "SaySo (local)"
    assert CONF_MODEL not in subentry.data
    assert subentry.data[CONF_LLM_HASS_API] == LLM_API_ASSIST


async def test_add_conversation_subentry_creates_entity(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)
    [default] = entry.subentries.values()

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": "user"},
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], _conversation_form_input("Kitchen", SECOND_MODEL)
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    assert len(entry.subentries) == 2
    [kitchen] = [s for s in entry.subentries.values() if s.title == "Kitchen"]
    assert kitchen.subentry_type == CONVERSATION_SUBENTRY_TYPE
    assert kitchen.data[CONF_MODEL] == SECOND_MODEL
    assert kitchen.data[CONF_PROMPT] == "Keep answers short."
    assert kitchen.data[CONF_MAX_TOOL_ITERATIONS] == 3
    assert kitchen.subentry_id != default.subentry_id

    registry = er.async_get(hass)
    entity_id = _conversation_entity_id(hass, kitchen.subentry_id)
    assert entity_id is not None
    assert registry.async_get(entity_id).config_subentry_id == kitchen.subentry_id


async def test_reconfigure_conversation_subentry_updates_in_place(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)
    [subentry] = entry.subentries.values()
    entity_id = _conversation_entity_id(hass, subentry.subentry_id)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": "reconfigure", "subentry_id": subentry.subentry_id},
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "reconfigure"

    form = _conversation_form_input("Renamed", SECOND_MODEL)
    form[CONF_TEMPERATURE] = 0.7
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], form
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    await hass.async_block_till_done()

    updated = entry.subentries[subentry.subentry_id]
    assert updated.title == "Renamed"
    assert updated.data[CONF_MODEL] == SECOND_MODEL
    assert updated.data[CONF_TEMPERATURE] == 0.7
    assert _conversation_entity_id(hass, subentry.subentry_id) == entity_id


async def test_reconfigured_temperature_and_max_tokens_reach_the_engine(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)
    [subentry] = entry.subentries.values()
    entity_id = _conversation_entity_id(hass, subentry.subentry_id)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": "reconfigure", "subentry_id": subentry.subentry_id},
    )
    form = _conversation_form_input("Default", MODEL_ID)
    form[CONF_TEMPERATURE] = 0.7
    form[CONF_MAX_OUTPUT_TOKENS] = 321
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], form
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    await hass.async_block_till_done()

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(return_value=ChatCompletionResult(content="Hi.", tool_calls=[])),
    ) as mock_chat:
        await conversation.async_converse(
            hass, "Say hi", None, Context(), agent_id=entity_id
        )

    assert mock_chat.await_count == 1
    assert mock_chat.await_args.kwargs["temperature"] == 0.7
    assert mock_chat.await_args.kwargs["max_tokens"] == 321


async def test_one_entity_per_subentry_with_stable_unique_id(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": "user"},
    )
    await hass.config_entries.subentries.async_configure(
        result["flow_id"], _conversation_form_input("Office", SECOND_MODEL)
    )
    await hass.async_block_till_done()

    registry = er.async_get(hass)
    entities = er.async_entries_for_config_entry(registry, entry.entry_id)
    conversation_entities = [e for e in entities if e.domain == conversation.DOMAIN]
    assert len(conversation_entities) == 2
    assert {e.unique_id for e in conversation_entities} == set(entry.subentries)
    before = {e.unique_id: e.entity_id for e in conversation_entities}

    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done()

    after = {
        e.unique_id: e.entity_id
        for e in er.async_entries_for_config_entry(registry, entry.entry_id)
        if e.domain == conversation.DOMAIN
    }
    assert after == before


async def test_ai_task_subentry_flow_creates_and_renames(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, AI_TASK_SUBENTRY_TYPE),
        context={"source": "user"},
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_NAME: "Summaries"}
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()

    [task] = [
        s for s in entry.subentries.values() if s.subentry_type == AI_TASK_SUBENTRY_TYPE
    ]
    assert task.title == "Summaries"
    assert dict(task.data) == {}
    registry = er.async_get(hass)
    assert not [
        e
        for e in er.async_entries_for_config_entry(registry, entry.entry_id)
        if e.config_subentry_id == task.subentry_id
    ]

    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, AI_TASK_SUBENTRY_TYPE),
        context={"source": "reconfigure", "subentry_id": task.subentry_id},
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_NAME: "Daily summaries"}
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.subentries[task.subentry_id].title == "Daily summaries"


async def test_model_is_passed_per_request_on_external(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)
    [default] = entry.subentries.values()
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": "user"},
    )
    await hass.config_entries.subentries.async_configure(
        result["flow_id"], _conversation_form_input("Office", SECOND_MODEL)
    )
    await hass.async_block_till_done()
    [office] = [s for s in entry.subentries.values() if s.title == "Office"]

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(return_value=ChatCompletionResult(content="Hi.", tool_calls=[])),
    ) as mock_chat:
        for subentry in (default, office):
            await conversation.async_converse(
                hass,
                "Say hi",
                None,
                Context(),
                agent_id=_conversation_entity_id(hass, subentry.subentry_id),
            )

    assert [call.kwargs["model"] for call in mock_chat.await_args_list] == [
        MODEL_ID,
        SECOND_MODEL,
    ]


async def test_external_options_flow_only_offers_engine_options(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.FORM
    fields = {str(key) for key in result["data_schema"].schema}
    assert {
        CONF_TIMEOUT,
        CONF_TRACE_RETENTION_DAYS,
        CONF_TRACE_MAX_INTERACTIONS,
        CONF_TRACE_STORE_UTTERANCES,
        CONF_GRAMMAR_DECODING,
        CONF_DUCK_VOLUME,
    } <= fields
    assert not fields & {
        CONF_MODEL,
        CONF_LLM_HASS_API,
        CONF_PROMPT,
        CONF_TEMPERATURE,
        CONF_MAX_OUTPUT_TOKENS,
        CONF_MAX_TOOL_ITERATIONS,
    }

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_TIMEOUT: 45,
            CONF_TRACE_RETENTION_DAYS: 30,
            CONF_TRACE_MAX_INTERACTIONS: 500,
            CONF_TRACE_STORE_UTTERANCES: False,
            CONF_GRAMMAR_DECODING: False,
            CONF_DUCK_VOLUME: 0.02,
        },
    )
    assert result["type"] == FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    assert entry.options[CONF_TIMEOUT] == 45
    assert CONF_TEMPERATURE not in entry.options
    assert CONF_MODEL not in entry.options


async def test_diagnostics_reports_each_subentry(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": "user"},
    )
    await hass.config_entries.subentries.async_configure(
        result["flow_id"], _conversation_form_input("Office", SECOND_MODEL)
    )
    await hass.async_block_till_done()

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    subentries = diagnostics["entry"]["subentries"]
    assert {s["title"] for s in subentries} == {entry.title, "Office"}
    office = next(s for s in subentries if s["title"] == "Office")
    assert office["subentry_type"] == CONVERSATION_SUBENTRY_TYPE
    assert office["data"][CONF_MODEL] == SECOND_MODEL
    assert diagnostics["entry"]["options"][CONF_TIMEOUT] == entry.options[CONF_TIMEOUT]


async def test_entry_id_alias_reaches_primary_subentry(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": "user"},
    )
    await hass.config_entries.subentries.async_configure(
        result["flow_id"], _conversation_form_input("Office", SECOND_MODEL)
    )
    await hass.async_block_till_done()

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(return_value=ChatCompletionResult(content="Hi.", tool_calls=[])),
    ) as mock_chat:
        await conversation.async_converse(
            hass, "Say hi", None, Context(), agent_id=entry.entry_id
        )

    assert mock_chat.await_count == 1
    assert mock_chat.await_args.kwargs["model"] == MODEL_ID


async def test_second_subentry_is_reachable_by_entity_id_only(
    hass: HomeAssistant,
    mock_llama_client: None,
) -> None:
    entry = await _create_external_entry(hass)
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": "user"},
    )
    await hass.config_entries.subentries.async_configure(
        result["flow_id"], _conversation_form_input("Office", SECOND_MODEL)
    )
    await hass.async_block_till_done()
    [office] = [s for s in entry.subentries.values() if s.title == "Office"]
    office_entity_id = _conversation_entity_id(hass, office.subentry_id)
    assert office_entity_id is not None

    with patch.object(
        LlamaCppClient,
        "chat_completion",
        new=AsyncMock(return_value=ChatCompletionResult(content="Hi.", tool_calls=[])),
    ) as mock_chat:
        await conversation.async_converse(
            hass, "Say hi", None, Context(), agent_id=office_entity_id
        )

    assert mock_chat.await_args.kwargs["model"] == SECOND_MODEL
    assert conversation.async_get_agent(hass, entry.entry_id) is not None
