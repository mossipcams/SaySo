from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from custom_components.sayso.core.completion import extract_tool_calls
from custom_components.sayso.core.grammar import GrammarUnsupported, build_grammar
from custom_components.sayso.core.lfm_parse import parse_lfm_python_tool_calls

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = {
    "ha": REPO_ROOT / "schemas" / "ha-2026.11.1-tools.json",
    "sayso": REPO_ROOT / "schemas" / "sayso-tool-schema-v2.json",
}
MARKER_START = "<|tool_call_start|>"
MARKER_END = "<|tool_call_end|>"


def _fixture_tools(key: str) -> list[dict[str, Any]]:
    return json.loads(FIXTURES[key].read_text())["tools"]


def _tool(name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "parameters": parameters},
    }


def _object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }


class _Gbnf:

    def __init__(self, expression: str) -> None:
        self.src = expression
        self.pos = 0

    def parse(self) -> tuple[Any, ...]:
        node = self.alternatives()
        if self.pos != len(self.src):
            raise ValueError(f"trailing GBNF at {self.pos}: {self.src[self.pos:]!r}")
        return node

    def peek(self) -> str:
        return self.src[self.pos] if self.pos < len(self.src) else ""

    def alternatives(self) -> tuple[Any, ...]:
        branches = [self.sequence()]
        while self.peek() == "|":
            self.pos += 1
            branches.append(self.sequence())
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def sequence(self) -> tuple[Any, ...]:
        items: list[Any] = []
        self.skip_spaces()
        while self.peek() not in ("", "|", ")"):
            items.append(self.postfix())
            self.skip_spaces()
        return ("seq", items)

    def postfix(self) -> tuple[Any, ...]:
        node = self.atom()
        while self.peek() in ("?", "*", "+"):
            marker = self.peek()
            self.pos += 1
            node = {"?": ("opt", node), "*": ("star", node), "+": ("plus", node)}[marker]
        return node

    def atom(self) -> tuple[Any, ...]:
        char = self.peek()
        if char == '"':
            self.pos += 1
            return ("lit", self.quoted('"'))
        if char == "[":
            self.pos += 1
            return self.char_class()
        if char == "(":
            self.pos += 1
            node = self.alternatives()
            if self.peek() != ")":
                raise ValueError("unclosed group")
            self.pos += 1
            return node
        match = re.compile(r"[A-Za-z0-9_-]+").match(self.src, self.pos)
        if not match:
            raise ValueError(f"unexpected {char!r} at {self.pos}")
        self.pos = match.end()
        return ("ref", match.group(0))

    def skip_spaces(self) -> None:
        while self.peek() == " ":
            self.pos += 1

    def char(self) -> str:
        if self.peek() != "\\":
            value = self.peek()
            self.pos += 1
            return value
        escape = self.src[self.pos + 1]
        self.pos += 2
        return {"n": "\n", "r": "\r", "t": "\t"}.get(escape, escape)

    def quoted(self, terminator: str) -> str:
        chars: list[str] = []
        while self.peek() != terminator:
            if not self.peek():
                raise ValueError("unterminated literal")
            chars.append(self.char())
        self.pos += 1
        return "".join(chars)

    def char_class(self) -> tuple[Any, ...]:
        negated = self.peek() == "^"
        if negated:
            self.pos += 1
        ranges: list[tuple[str, str]] = []
        while self.peek() != "]":
            if not self.peek():
                raise ValueError("unterminated character class")
            low = self.char()
            high = low
            if self.peek() == "-" and self.src[self.pos + 1] != "]":
                self.pos += 1
                high = self.char()
            ranges.append((low, high))
        self.pos += 1
        return ("cls", negated, ranges)


def _parse_rules(grammar: str) -> dict[str, tuple[Any, ...]]:
    rules: dict[str, tuple[Any, ...]] = {}
    for line in grammar.splitlines():
        if not line.strip():
            continue
        head, separator, body = line.partition(" ::= ")
        if not separator:
            raise ValueError(f"rule without ::=: {line!r}")
        if not re.fullmatch(r"[a-z0-9-]+", head):
            raise ValueError(f"rule name not llama.cpp compatible: {head!r}")
        if head in rules:
            raise ValueError(f"duplicate rule {head!r}")
        rules[head] = _Gbnf(body).parse()
    return rules


def _undefined_refs(node: Any, rules: dict[str, Any]) -> set[str]:
    if isinstance(node, tuple) and node and node[0] == "ref":
        return {node[1]} - set(rules)
    if isinstance(node, tuple):
        found: set[str] = set()
        for child in node:
            if isinstance(child, (tuple, list)):
                found |= _undefined_refs(child, rules)
        return found
    if isinstance(node, list):
        found = set()
        for child in node:
            found |= _undefined_refs(child, rules)
        return found
    return set()


@lru_cache(maxsize=16)
def _rules_of(grammar: str) -> dict[str, tuple[Any, ...]]:
    rules = _parse_rules(grammar)
    assert "root" in rules
    dangling = set()
    for node in rules.values():
        dangling |= _undefined_refs(node, rules)
    assert not dangling, f"undefined rules: {sorted(dangling)}"
    return rules


class _Matcher:

    def __init__(self, rules: dict[str, tuple[Any, ...]], text: str) -> None:
        self.rules = rules
        self.text = text
        self.memo: dict[tuple[str, int], set[int]] = {}

    def ends(self, node: tuple[Any, ...], pos: int) -> set[int]:
        kind = node[0]
        if kind == "lit":
            literal = node[1]
            return {pos + len(literal)} if self.text.startswith(literal, pos) else set()
        if kind == "cls":
            if pos < len(self.text) and self.in_class(node, self.text[pos]):
                return {pos + 1}
            return set()
        if kind == "seq":
            positions = {pos}
            for item in node[1]:
                positions = set().union(*(self.ends(item, p) for p in positions))
                if not positions:
                    break
            return positions
        if kind == "alt":
            return set().union(*(self.ends(branch, pos) for branch in node[1]))
        if kind == "opt":
            return {pos} | self.ends(node[1], pos)
        if kind == "star":
            return self.closure(node[1], {pos})
        if kind == "plus":
            return self.closure(node[1], self.ends(node[1], pos))
        key = (node[1], pos)
        if key not in self.memo:
            self.memo[key] = self.ends(self.rules[node[1]], pos)
        return self.memo[key]

    def closure(self, inner: tuple[Any, ...], starts: set[int]) -> set[int]:
        result = set(starts)
        frontier = set(starts)
        while frontier:
            fresh: set[int] = set()
            for pos in frontier:
                for end in self.ends(inner, pos):
                    if end not in result:
                        result.add(end)
                        fresh.add(end)
            frontier = fresh
        return result

    @staticmethod
    def in_class(node: tuple[Any, ...], char: str) -> bool:
        _, negated, ranges = node
        hit = any(low <= char <= high for low, high in ranges)
        return hit != negated


def _accepts(grammar: str, text: str) -> bool:
    rules = _rules_of(grammar)
    return len(text) in _Matcher(rules, text).ends(("ref", "root"), 0)


def _sample_value(node: dict[str, Any]) -> Any:
    if "anyOf" in node:
        return _sample_value(node["anyOf"][0])
    if "enum" in node:
        return node["enum"][0]
    kind = node["type"]
    if kind == "string":
        return "Mom's lamp"
    if kind in ("integer", "number"):
        return 21
    if kind == "boolean":
        return True
    return [_sample_value(node["items"])]


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return f"'{value}'"
    return "[" + ", ".join(_render(item) for item in value) + "]"


def _sample_call(tool: dict[str, Any], *, reverse: bool = False) -> tuple[str, dict[str, Any]]:
    name = tool["function"]["name"]
    properties = tool["function"]["parameters"].get("properties", {})
    arguments = {key: _sample_value(node) for key, node in properties.items()}
    keys = list(arguments)
    if reverse:
        keys.reverse()
    body = ", ".join(f"{key}={_render(arguments[key])}" for key in keys)
    return f"{name}({body})", arguments


def _wrap(call: str, *, markers: bool) -> str:
    if markers:
        return f"{MARKER_START}[{call}]{MARKER_END}"
    return f"[{call}]"


def _extracted(text: str) -> list[tuple[str, dict[str, Any]]]:
    _, calls = extract_tool_calls(text)
    return [(call.name, call.arguments) for call in calls]


GRAMMAR_CASES = [(key, markers) for key in FIXTURES for markers in (True, False)]


class TestBuildGrammarFixtures:

    @pytest.mark.parametrize(("key", "markers"), GRAMMAR_CASES)
    def test_builds_and_is_well_formed(self, key: str, markers: bool) -> None:
        grammar = build_grammar(_fixture_tools(key), markers=markers)

        rules = _rules_of(grammar)

        assert grammar.startswith("root ::= block | text\n")
        assert "block" in rules
        assert "text" in rules

    @pytest.mark.parametrize(("key", "markers"), GRAMMAR_CASES)
    def test_round_trip_every_tool(self, key: str, markers: bool) -> None:
        grammar = build_grammar(_fixture_tools(key), markers=markers)

        for tool in _fixture_tools(key):
            call, arguments = _sample_call(tool)
            text = _wrap(call, markers=markers)

            assert _accepts(grammar, text), text
            assert parse_lfm_python_tool_calls(text.replace(MARKER_START, "").replace(MARKER_END, "")) == [
                {"name": tool["function"]["name"], "arguments": arguments}
            ]
            assert _extracted(text) == [(tool["function"]["name"], arguments)]

    @pytest.mark.parametrize(("key", "markers"), GRAMMAR_CASES)
    def test_round_trip_reversed_key_order(self, key: str, markers: bool) -> None:
        grammar = build_grammar(_fixture_tools(key), markers=markers)

        for tool in _fixture_tools(key):
            call, arguments = _sample_call(tool, reverse=True)
            text = _wrap(call, markers=markers)

            assert _accepts(grammar, text), text
            assert _extracted(text) == [(tool["function"]["name"], arguments)]

    def test_multi_call_group_round_trips(self) -> None:
        tools = _fixture_tools("ha")
        first_call, first_args = _sample_call(tools[0])
        second_call, second_args = _sample_call(tools[1])
        call = f"{first_call}, {second_call}"
        grammar = build_grammar(tools, markers=True)

        text = f"{MARKER_START}[{call}]{MARKER_END}"

        assert _accepts(grammar, text)
        assert _extracted(text) == [
            (tools[0]["function"]["name"], first_args),
            (tools[1]["function"]["name"], second_args),
        ]

    def test_empty_argument_call_round_trips(self) -> None:
        tools = _fixture_tools("sayso")
        grammar = build_grammar(tools, markers=False)

        assert _accepts(grammar, "[GetDateTime()]")
        assert _extracted("[GetDateTime()]") == [("GetDateTime", {})]

    def test_marker_mode_excludes_bare_block(self) -> None:
        grammar = build_grammar(_fixture_tools("sayso"), markers=True)

        assert not _accepts(grammar, "[GetDateTime()]")
        assert _accepts(grammar, f"{MARKER_START}[GetDateTime()]{MARKER_END}")

    def test_bare_mode_excludes_marker_block(self) -> None:
        grammar = build_grammar(_fixture_tools("sayso"), markers=False)

        assert not _accepts(grammar, f"{MARKER_START}[GetDateTime()]{MARKER_END}")
        assert _accepts(grammar, "[GetDateTime()]")


class TestRepresentableValues:

    def test_apostrophe_inside_string_is_representable(self) -> None:
        tools = [_tool("Say", _object({"message": {"type": "string"}}))]
        grammar = build_grammar(tools, markers=False)

        assert _accepts(grammar, "[Say(message='Mom's lamp')]")
        assert _extracted("[Say(message='Mom's lamp')]") == [
            ("Say", {"message": "Mom's lamp"})
        ]

    def test_string_closing_quote_before_comma_is_representable(self) -> None:
        tools = [_tool("Say", _object({"a": {"type": "string"}, "b": {"type": "string"}}))]
        grammar = build_grammar(tools, markers=False)

        assert _accepts(grammar, "[Say(a='x', b='y')]")

    def test_string_with_parenthesis_is_not_representable(self) -> None:
        tools = [_tool("Say", _object({"message": {"type": "string"}}))]
        grammar = build_grammar(tools, markers=False)

        assert not _accepts(grammar, "[Say(message='a(b)')]")
        assert not _accepts(grammar, "[Say(message='a)b')]")

    def test_array_of_enum_values(self) -> None:
        tools = [
            _tool(
                "Pick",
                _object(
                    {
                        "domain": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["light", "fan"]},
                        }
                    }
                ),
            )
        ]
        grammar = build_grammar(tools, markers=False)

        assert _accepts(grammar, "[Pick(domain=['light', 'fan'])]")
        assert _accepts(grammar, "[Pick(domain=[])]")
        assert not _accepts(grammar, "[Pick(domain=['lock'])]")

    def test_property_union_accepts_each_branch(self) -> None:
        tools = [
            _tool(
                "Step",
                _object(
                    {
                        "volume_step": {
                            "anyOf": [
                                {"enum": ["up", "down"], "type": "string"},
                                {"maximum": 100, "minimum": -100, "type": "integer"},
                            ]
                        }
                    }
                ),
            )
        ]
        grammar = build_grammar(tools, markers=False)

        assert _accepts(grammar, "[Step(volume_step='up')]")
        assert _accepts(grammar, "[Step(volume_step=-10)]")
        assert not _accepts(grammar, "[Step(volume_step='sideways')]")

    def test_object_level_required_any_of_is_ignored(self) -> None:
        parameters = _object({"hours": {"type": "integer"}, "minutes": {"type": "integer"}})
        parameters["anyOf"] = [{"required": ["hours"]}, {"required": ["minutes"]}]
        grammar = build_grammar([_tool("Timer", parameters)], markers=False)

        assert _accepts(grammar, "[Timer(minutes=5)]")


class TestNegativeCalls:

    @pytest.fixture
    def ha_bare(self) -> str:
        return build_grammar(_fixture_tools("ha"), markers=False)

    @pytest.fixture
    def ha_markers(self) -> str:
        return build_grammar(_fixture_tools("ha"), markers=True)

    @pytest.mark.parametrize(
        "text",
        [
            "[HassTurnOn(name=)]",
            "[intent__HassTurnOn(name=)]",
            "[intent__HassTurnOn(name=-)]",
            "[climate__HassClimateSetTemperature(temperature=-)]",
            "[climate__HassClimateSetTemperature(temperature=21.5)]",
            "[NoSuchTool(name='lamp')]",
            "[intent__HassTurnOn(bogus='lamp')]",
            "[intent__HassTurnOn(name='lamp')] [intent__HassTurnOff(name='fan')]",
            "[intent__HassTurnOn({\"name\": \"lamp\"})]",
            "[intent__HassTurnOn(name='lamp'),]",
            "[intent__HassTurnOn(name='lamp' )]",
        ],
    )
    def test_outside_full_language_bare(self, ha_bare: str, text: str) -> None:
        assert not _accepts(ha_bare, text)

    @pytest.mark.parametrize(
        "text",
        [
            f"{MARKER_START}[intent__HassTurnOn(name=)]{MARKER_END}",
            f"{MARKER_START}[climate__HassClimateSetTemperature(temperature=-)]{MARKER_END}",
            f"{MARKER_START}[climate__HassClimateSetTemperature(temperature=21.5)]{MARKER_END}",
            f"{MARKER_START}[NoSuchTool(name='lamp')]{MARKER_END}",
            f"{MARKER_START}[intent__HassTurnOn(bogus='lamp')]{MARKER_END}",
            f"{MARKER_START}[intent__HassTurnOn(name='lamp')]{MARKER_END}{MARKER_START}[intent__HassTurnOff(name='fan')]{MARKER_END}",
        ],
    )
    def test_outside_full_language_markers(self, ha_markers: str, text: str) -> None:
        assert not _accepts(ha_markers, text)

    def test_malformed_call_is_outside_whole_language(self, ha_bare: str) -> None:
        assert not _accepts(ha_bare, "[HassTurnOn(name=)]")

    def test_bare_minus_integer_is_not_in_language(self) -> None:
        tools = [_tool("SetTemp", _object({"level": {"type": "integer"}}))]
        grammar = build_grammar(tools, markers=False)

        assert not _accepts(grammar, "[SetTemp(level=-)]")
        assert _accepts(grammar, "[SetTemp(level=-5)]")
        assert _accepts(grammar, "[SetTemp(level=21)]")

    def test_float_for_number_is_not_in_language(self) -> None:
        tools = [_tool("SetTemp", _object({"level": {"type": "number"}}))]
        grammar = build_grammar(tools, markers=False)

        assert not _accepts(grammar, "[SetTemp(level=21.5)]")
        assert _accepts(grammar, "[SetTemp(level=21)]")


class TestTextBranch:

    @pytest.mark.parametrize("markers", [True, False])
    def test_plain_prose_is_accepted(self, markers: bool) -> None:
        grammar = build_grammar(_fixture_tools("sayso"), markers=markers)

        assert _accepts(grammar, "The lamp is on.")

    @pytest.mark.parametrize("markers", [True, False])
    @pytest.mark.parametrize(
        "text",
        [
            "",
            " Turning on the lamp.",
            "[HassTurnOn(name=)]",
            "<|tool_call_start|>",
            "Sure. <|tool_call_start|>[GetDateTime()]<|tool_call_end|>",
        ],
    )
    def test_text_never_swallows_calls(self, markers: bool, text: str) -> None:
        grammar = build_grammar(_fixture_tools("sayso"), markers=markers)

        assert not _accepts(grammar, text)


class TestUnsupportedSchemas:

    @pytest.mark.parametrize(
        ("parameters", "reason"),
        [
            (_object({"level": {"type": ["integer", "null"]}}), "unsupported_type"),
            (_object({"level": {"type": "null"}}), "unsupported_type"),
            (_object({"level": {"type": "string", "nullable": True}}), "nullable"),
            (_object({"level": {"description": "no type"}}), "missing_type"),
            (_object({"level": {"type": "object", "properties": {}}}), "nested_object"),
            (
                _object({"level": {"anyOf": [{"type": "object", "properties": {}}, {"type": "string"}]}}),
                "nested_object",
            ),
            (_object({"level": {"type": "array"}}), "array_items"),
            (_object({"level": {"type": "string", "enum": ["it's"]}}), "enum_value"),
            (_object({"level": {"type": "string", "enum": ["a(b"]}}), "enum_value"),
            (_object({"level": {"type": "integer", "enum": [1, 2]}}), "enum_type"),
            (_object({"level": {"anyOf": [{"type": "string"}], "type": "string"}}), "any_of"),
            (
                {"type": "object", "properties": {}, "oneOf": [{"required": ["a"]}]},
                "object_constraint",
            ),
        ],
    )
    def test_unsupported_node_raises_with_reason(
        self, parameters: dict[str, Any], reason: str
    ) -> None:
        with pytest.raises(GrammarUnsupported) as caught:
            build_grammar([_tool("Thing", parameters)], markers=False)

        assert caught.value.reason == reason

    def test_hyphenated_tool_name_is_unsupported(self) -> None:
        with pytest.raises(GrammarUnsupported) as caught:
            build_grammar([_tool("bad-name", _object({}))], markers=False)

        assert caught.value.reason == "tool_name"

    def test_argument_name_must_be_identifier(self) -> None:
        with pytest.raises(GrammarUnsupported) as caught:
            build_grammar(
                [_tool("Thing", _object({"bad-key": {"type": "string"}}))], markers=False
            )

        assert caught.value.reason == "argument_name"

    def test_empty_catalog_is_unsupported(self) -> None:
        with pytest.raises(GrammarUnsupported) as caught:
            build_grammar([], markers=False)

        assert caught.value.reason == "no_tools"


class TestCache:

    def test_same_catalog_returns_same_object(self) -> None:
        tools = _fixture_tools("ha")

        first = build_grammar(tools, markers=True)
        second = build_grammar(copy.deepcopy(tools), markers=True)

        assert first is second

    def test_key_order_of_catalog_does_not_change_grammar(self) -> None:
        tools = _fixture_tools("sayso")

        first = build_grammar(tools, markers=False)
        second = build_grammar(list(reversed(tools)), markers=False)

        assert first is second

    def test_marker_mode_is_cached_separately(self) -> None:
        tools = _fixture_tools("sayso")

        markers = build_grammar(tools, markers=True)
        bare = build_grammar(tools, markers=False)

        assert markers != bare
        assert build_grammar(tools, markers=True) is markers
