from __future__ import annotations

import functools
import json
import re
from collections.abc import Sequence
from typing import Any

from .tool_schema import emit_canonical_json

_TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_ARG_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ENUM_VALUE = re.compile(r"[^'()\"\\\x00-\x1f]*")

_SHARED_RULES = r"""
ws ::= [ \t\r\n]*
v-body ::= ( [^'()] | "'" [^,)\]}'()] )*
v-str ::= "'" v-body "'"
v-int ::= "-"? [0-9]+
v-bool ::= "True" | "False"
text ::= [^ \t\r\n\[<] [^<]*
""".strip().splitlines()

_MARKER_BLOCK = (
    'block ::= "<|tool_call_start|>" ws "[" ws call-list ws "]" ws "<|tool_call_end|>"'
)
_BARE_BLOCK = 'block ::= ws "[" ws call-list ws "]" ws'


class GrammarUnsupported(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Rules:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, name: str, expression: str) -> None:
        self.lines.append(f"{name} ::= {expression}")

    def value(self, node: Any) -> str:
        if not isinstance(node, dict):
            raise GrammarUnsupported("schema_node")
        if node.get("nullable"):
            raise GrammarUnsupported("nullable")
        if "anyOf" in node:
            return self._any_of(node)
        kind = node.get("type")
        if "enum" in node:
            return self._enum(node, kind)
        if kind == "string":
            return "v-str"
        if kind in ("integer", "number"):
            return "v-int"
        if kind == "boolean":
            return "v-bool"
        if kind == "array":
            return self._array(node)
        if kind is None:
            raise GrammarUnsupported("missing_type")
        if kind == "object":
            raise GrammarUnsupported("nested_object")
        raise GrammarUnsupported("unsupported_type")

    def _any_of(self, node: dict[str, Any]) -> str:
        branches = node["anyOf"]
        if "type" in node or "enum" in node:
            raise GrammarUnsupported("any_of")
        if not isinstance(branches, list) or not branches:
            raise GrammarUnsupported("any_of")
        return "(" + " | ".join(self.value(branch) for branch in branches) + ")"

    def _enum(self, node: dict[str, Any], kind: Any) -> str:
        values = node["enum"]
        if kind != "string" or not isinstance(values, list) or not values:
            raise GrammarUnsupported("enum_type")
        literals: list[str] = []
        for value in values:
            if not isinstance(value, str) or not _ENUM_VALUE.fullmatch(value):
                raise GrammarUnsupported("enum_value")
            literals.append("\"'" + value + "'\"")
        return "(" + " | ".join(literals) + ")"

    def _array(self, node: dict[str, Any]) -> str:
        items = node.get("items")
        if not isinstance(items, dict):
            raise GrammarUnsupported("array_items")
        index = len(self.lines)
        self.lines.append("")
        item = self.value(items)
        self.lines[index] = f'arr-{index} ::= "[" ws ( {item} ("," ws {item})* )? "]"'
        return f"arr-{index}"


def _arguments(parameters: Any) -> list[tuple[str, Any]]:
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise GrammarUnsupported("parameters")
    for key in ("oneOf", "allOf", "not", "patternProperties"):
        if key in parameters:
            raise GrammarUnsupported("object_constraint")
    branches = parameters.get("anyOf", [])
    if not isinstance(branches, list) or any(
        not isinstance(branch, dict) or set(branch) - {"required"}
        for branch in branches
    ):
        raise GrammarUnsupported("object_constraint")
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise GrammarUnsupported("parameters")
    return list(properties.items())


def _compile(tools: Sequence[dict[str, Any]], *, markers: bool) -> str:
    if not tools:
        raise GrammarUnsupported("no_tools")
    rules = _Rules()
    call_names: list[str] = []
    for index, tool in enumerate(tools):
        function = tool["function"]
        name = function["name"]
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
            raise GrammarUnsupported("tool_name")
        call_rule = f"call-{index}"
        call_names.append(call_rule)
        arguments = _arguments(function.get("parameters"))
        if not arguments:
            rules.add(call_rule, f'"{name}(" ws ")"')
            continue
        argument_rules: list[str] = []
        for arg_index, (arg, node) in enumerate(arguments):
            if not _ARG_NAME.fullmatch(arg):
                raise GrammarUnsupported("argument_name")
            argument_rule = f"arg-{index}-{arg_index}"
            rules.add(argument_rule, f'"{arg}=" {rules.value(node)}')
            argument_rules.append(argument_rule)
        rules.add(f"kv-{index}", " | ".join(argument_rules))
        rules.add(
            call_rule,
            f'"{name}(" ws ( kv-{index} ("," ws kv-{index})* )? ")"',
        )

    head = [
        "root ::= block | text",
        _MARKER_BLOCK if markers else _BARE_BLOCK,
        'call-list ::= call (ws "," ws call)*',
        "call ::= " + " | ".join(call_names),
    ]
    return "\n".join(head + _SHARED_RULES + rules.lines) + "\n"


@functools.lru_cache(maxsize=32)
def _build_cached(canonical: str, markers: bool) -> str:
    return _compile(json.loads(canonical), markers=markers)


def build_grammar(tools: Sequence[dict[str, Any]], *, markers: bool) -> str:
    canonical = emit_canonical_json(list(tools)).decode("utf-8")
    return _build_cached(canonical, markers)
