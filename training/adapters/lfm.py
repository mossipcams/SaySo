"""LFM2.5-230M training helpers using SaySo's OpenAI tool envelope."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schema import (
    CHATML_TOOL_CALL_MARKERS,
    TrainingExample,
    assert_tools_subset_of_v1,
    contains_chatml_tool_call_markers,
    v1_openai_tools,
)

LFM_BASE_MODEL = "LiquidAI/LFM2.5-230M"
LFM_BASE_TRAINING_MODEL = "LiquidAI/LFM2.5-230M-Base"
_TRAINING_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_TOKENIZER_DIR = _TRAINING_ROOT / "artifacts" / "lfm-base-tokenizer"


def lfm_tool_catalog() -> list[dict[str, Any]]:
    """Canonical v1 tools for LFM training examples."""
    return v1_openai_tools()


def prepare_lfm_example(example: TrainingExample) -> TrainingExample:
    """Ensure a converted example uses the locked v1 catalog and runtime envelope."""
    assert_tools_subset_of_v1(example.tools)
    for message in example.messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and contains_chatml_tool_call_markers(content):
            raise ValueError("assistant content must not use ChatML <tool_call> labels")
        for call in message.get("tool_calls") or []:
            if call.get("type") != "function":
                raise ValueError("tool_calls must use type:function")
            fn = call.get("function") or {}
            if not isinstance(fn.get("name"), str):
                raise ValueError("tool_calls must declare function.name")
            args = fn.get("arguments")
            if not isinstance(args, str):
                raise ValueError("LFM/SaySo labels must keep function.arguments as JSON strings")
            if contains_chatml_tool_call_markers(args):
                raise ValueError("tool call arguments must not use ChatML <tool_call> labels")
    return TrainingExample(
        messages=example.messages,
        tools=lfm_tool_catalog(),
        metadata=example.metadata,
    )


def lfm_jsonl_line(example: TrainingExample) -> str:
    """Serialize one LFM training record (SaySo runtime envelope, not ChatML tool-call labels)."""
    prepared = prepare_lfm_example(example)
    line = prepared.to_jsonl_line(view="lfm")
    if contains_chatml_tool_call_markers(line):
        raise ValueError("serialized example must not contain ChatML tool-call labels")
    return line


def forbidden_chatml_tool_call_patterns() -> frozenset[str]:
    """Markers that must never appear in LFM fine-tuning labels."""
    return CHATML_TOOL_CALL_MARKERS


def validate_lfm_config_text(config_text: str) -> None:
    """Reject Axolotl configs that embed ChatML tool-call rendering."""
    for marker in CHATML_TOOL_CALL_MARKERS:
        if marker in config_text:
            raise ValueError(f"LFM config must not render ChatML tool-call labels ({marker})")


def _call_with_parsed_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``call`` with JSON-string ``arguments`` parsed for the chat template."""
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
        return call
    try:
        arguments = json.loads(function["arguments"])
    except json.JSONDecodeError:
        return call
    return {**call, "function": {**function, "arguments": arguments}}


def default_chat_template_model() -> str:
    """Prefer the checked-in Base tokenizer when present (offline tests)."""
    if (_LOCAL_TOKENIZER_DIR / "tokenizer.json").is_file():
        return str(_LOCAL_TOKENIZER_DIR)
    return LFM_BASE_TRAINING_MODEL


@lru_cache(maxsize=4)
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _openai_tools_for_template(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function", tool)
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                },
            }
        )
    return normalized


def _message_for_template(message: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
    if message.get("tool_calls"):
        entry["tool_calls"] = [_call_with_parsed_arguments(call) for call in message["tool_calls"]]
    if message.get("tool_call_id"):
        entry["tool_call_id"] = message["tool_call_id"]
    return entry


def assert_no_chatml_tool_call_markers(text: str, *, field: str) -> None:
    if contains_chatml_tool_call_markers(text):
        raise ValueError(f"{field} must not contain ChatML <tool_call> labels")


def render_supervised_turn(
    row: dict[str, Any],
    supervised_index: int,
    *,
    model_name: str | None = None,
) -> dict[str, str]:
    """Render one alpaca instruction/output pair for a supervised assistant turn."""
    messages = row.get("messages") or []
    message = messages[supervised_index]
    if message.get("role") != "assistant" or not message.get("train_on_turn"):
        raise ValueError("supervised_index must point at train_on_turn assistant message")

    tokenizer = _tokenizer(model_name or default_chat_template_model())
    tools = _openai_tools_for_template(row.get("tools") or [])
    prefix = [_message_for_template(item) for item in messages[:supervised_index]]
    target = _message_for_template(message)
    full_messages = prefix + [target]

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt = tokenizer.apply_chat_template(
        prefix,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not full_text.startswith(prompt):
        raise ValueError("rendered completion does not extend rendered prompt")
    completion = full_text[len(prompt) :]
    assert_no_chatml_tool_call_markers(prompt, field="instruction")
    assert_no_chatml_tool_call_markers(completion, field="output")
    return {"instruction": prompt, "output": completion}


def expand_canonical_row_to_views(
    row: dict[str, Any],
    *,
    model_name: str | None = None,
    source_row_index: int | None = None,
) -> list[dict[str, Any]]:
    """Expand one canonical row into one derived view per supervised assistant turn."""
    views: list[dict[str, Any]] = []
    for index, message in enumerate(row.get("messages") or []):
        if message.get("role") != "assistant" or not message.get("train_on_turn"):
            continue
        rendered = render_supervised_turn(row, index, model_name=model_name)
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "supervised_turn_index": index,
                "source_row_index": source_row_index,
            }
        )
        views.append({**rendered, "metadata": metadata})
    return views


def dataset_info_fragment(
    dataset_name: str,
    file_name: str,
) -> dict[str, dict[str, Any]]:
    """Build one LlamaFactory ``dataset_info.json`` entry (alpaca instruction/output)."""
    return {
        dataset_name: {
            "file_name": file_name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "response": "output",
            },
        }
    }


def validate_llamafactory_full_config(cfg: dict[str, Any]) -> None:
    """Fail closed on LoRA/quantization/ChatML markers in the v5 full recipe."""
    if cfg.get("finetuning_type") != "full":
        raise ValueError("v5 recipe must use finetuning_type: full")
    for key in ("lora", "quantization_bit", "adapter_name_or_path"):
        if cfg.get(key):
            raise ValueError(f"v5 recipe must not set {key}")
    if cfg.get("template") != "empty":
        raise ValueError("v5 recipe must use template: empty")
    if cfg.get("cutoff_len") != 8192:
        raise ValueError("v5 recipe must use cutoff_len: 8192")
    if cfg.get("train_on_prompt") is not False:
        raise ValueError("v5 recipe must set train_on_prompt: false")
    if cfg.get("mask_history") is not False:
        raise ValueError("v5 recipe must set mask_history: false")
    if cfg.get("learning_rate") != 2.0e-5:
        raise ValueError("v5 recipe must use learning_rate: 2.0e-5")
    if cfg.get("per_device_train_batch_size") != 1:
        raise ValueError("v5 recipe must use per_device_train_batch_size: 1")
    if cfg.get("gradient_accumulation_steps") != 32:
        raise ValueError("v5 recipe must use gradient_accumulation_steps: 32")
    if cfg.get("model_name_or_path") != "/srv/llm/models/LFM2.5-230M-Base":
        raise ValueError("v5 recipe must point at host Base weights path")
    if cfg.get("dataset_dir") != "/srv/llm/datasets":
        raise ValueError("v5 recipe must use dataset_dir: /srv/llm/datasets")
    if cfg.get("bf16") is not True:
        raise ValueError("v5 recipe must use bf16: true after HIP bf16 smoke")
    if cfg.get("fp16") is not False:
        raise ValueError("v5 recipe must set fp16: false when using bf16")
    validate_lfm_config_text(json.dumps(cfg))


