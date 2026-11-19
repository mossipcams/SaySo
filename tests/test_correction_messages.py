from __future__ import annotations

import json

from custom_components.sayso.core.completion import ToolCall
from custom_components.sayso.core.correction import (
    build_correction_messages as core_build_correction_messages,
)
from custom_components.sayso.core.tool_contract import (
    ToolArgumentFailureCode,
    ToolArgumentValidationError,
)
from custom_components.sayso.transcript import build_correction_messages

BASE_MESSAGES = [
    {"role": "system", "content": "SaySo voice assistant."},
    {"role": "user", "content": "Turn on the living room light"},
]

FAILED_CALL = ToolCall(
    id="call_bad", name="HassTurnOn", arguments={"name": "Living Room"}
)

VALIDATION_ERROR = ToolArgumentValidationError(
    code=ToolArgumentFailureCode.INVALID_ARGUMENTS,
    message="name must be a string",
    tool_name="HassTurnOn",
)

ALLOWED_TOOLS = {"HassGetState", "HassTurnOn"}

FINGERPRINT = "fp_123"


def test_correction_messages_match_across_modules() -> None:
    failures = [(FAILED_CALL, VALIDATION_ERROR)]
    transcript_messages = build_correction_messages(
        BASE_MESSAGES, failures, ALLOWED_TOOLS, FINGERPRINT
    )
    core_messages = core_build_correction_messages(
        BASE_MESSAGES, failures, ALLOWED_TOOLS, FINGERPRINT
    )

    expected = [
        *BASE_MESSAGES,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {
                        "name": "HassTurnOn",
                        "arguments": json.dumps({"name": "Living Room"}, sort_keys=True),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_bad",
            "content": json.dumps(
                {
                    "error": {
                        "code": ToolArgumentFailureCode.INVALID_ARGUMENTS,
                        "message": "name must be a string",
                        "allowed_tools": ["HassGetState", "HassTurnOn"],
                        "schema_fingerprint": FINGERPRINT,
                    }
                }
            ),
        },
    ]

    assert transcript_messages == core_messages
    assert transcript_messages == expected
