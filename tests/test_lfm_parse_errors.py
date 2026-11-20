from __future__ import annotations

import pytest

from custom_components.sayso.core.completion import extract_tool_calls
from custom_components.sayso.core.exceptions import SaySoInvalidToolCallsError
from custom_components.sayso.core.lfm_parse import (
    LfmPythonParseError,
    parse_lfm_python_tool_call,
    parse_lfm_python_tool_calls,
)


class TestBareMinusIsParseError:
    @pytest.mark.parametrize(
        "text",
        [
            "[SetTemp(level=-)]",
            "[SetTemp(level=--)]",
            "[SetTemp(level=1-)]",
            "[SetTemp(level=--1)]",
            "[SetTemp(level=1-2)]",
        ],
    )
    def test_invalid_integer_raises_parse_error(self, text: str) -> None:
        with pytest.raises(LfmPythonParseError):
            parse_lfm_python_tool_calls(text)

    def test_bare_minus_is_not_a_plain_value_error(self) -> None:
        with pytest.raises(LfmPythonParseError) as caught:
            parse_lfm_python_tool_call("SetTemp(level=-)")
        assert type(caught.value) is LfmPythonParseError

    def test_bare_minus_through_extract_raises_invalid_tool_calls(self) -> None:
        with pytest.raises(SaySoInvalidToolCallsError):
            extract_tool_calls("<|tool_call_start|>[SetTemp(level=-)]<|tool_call_end|>")

    def test_bare_minus_bare_bracket_through_extract_raises_invalid_tool_calls(
        self,
    ) -> None:
        with pytest.raises(SaySoInvalidToolCallsError):
            extract_tool_calls("[SetTemp(level=-)]")


class TestAcceptedIntegersUnchanged:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("SetTemp(level=0)", 0),
            ("SetTemp(level=21)", 21),
            ("SetTemp(level=-5)", -5),
            ("SetTemp(level=-0)", 0),
        ],
    )
    def test_valid_integer_still_parses(self, text: str, value: int) -> None:
        assert parse_lfm_python_tool_call(text) == {
            "name": "SetTemp",
            "arguments": {"level": value},
        }
