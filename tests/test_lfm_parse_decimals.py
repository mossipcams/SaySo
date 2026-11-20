from __future__ import annotations

import pytest

from custom_components.sayso.core.lfm_parse import (
    LfmPythonParseError,
    parse_lfm_python_tool_call,
    parse_lfm_python_tool_calls,
)


class TestDecimalLiteralsParse:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("HassClimateSetTemperature(temperature=21.5)", 21.5),
            ("HassClimateSetTemperature(temperature=-0.5)", -0.5),
            ("HassClimateSetTemperature(temperature=0.25)", 0.25),
        ],
    )
    def test_decimal_parses_to_float(self, text: str, value: float) -> None:
        arguments = parse_lfm_python_tool_call(text)["arguments"]

        assert arguments == {"temperature": value}
        assert type(arguments["temperature"]) is float

    def test_decimal_in_list_parses(self) -> None:
        assert parse_lfm_python_tool_call("SetLevels(levels=[0.5, -1.25, 3])") == {
            "name": "SetLevels",
            "arguments": {"levels": [0.5, -1.25, 3]},
        }

    def test_decimal_tool_call_list_parses(self) -> None:
        assert parse_lfm_python_tool_calls("[SetTemp(level=21.5)]") == [
            {"name": "SetTemp", "arguments": {"level": 21.5}}
        ]


class TestIntegersStillInt:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("SetTemp(level=21)", 21),
            ("SetTemp(level=-5)", -5),
            ("SetTemp(level=0)", 0),
        ],
    )
    def test_integer_stays_int(self, text: str, value: int) -> None:
        level = parse_lfm_python_tool_call(text)["arguments"]["level"]

        assert level == value
        assert type(level) is int


class TestMalformedNumbersRaise:
    @pytest.mark.parametrize(
        "text",
        [
            "[SetTemp(level=21.)]",
            "[SetTemp(level=.5)]",
            "[SetTemp(level=1e3)]",
            "[SetTemp(level=-)]",
            "[SetTemp(level=1.5.2)]",
            "[SetTemp(level=-.5)]",
        ],
    )
    def test_malformed_number_raises_parse_error(self, text: str) -> None:
        with pytest.raises(LfmPythonParseError):
            parse_lfm_python_tool_calls(text)
