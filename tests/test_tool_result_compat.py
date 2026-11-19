from __future__ import annotations

from custom_components.sayso.boundary import tool_result_data


class _OldShapeResult:
    def __init__(self) -> None:
        self.tool_result = {"success": True, "data": {"ok": 1}}


class _NewToolResult:
    def __init__(self, data: dict, error: bool) -> None:
        self.data = data
        self.error = error


class _NewShapeContent:
    def __init__(self) -> None:
        self.result = _NewToolResult(data={"success": True}, error=False)

    @property
    def tool_result(self):
        raise AttributeError("deprecated tool_result must not be read")


def test_tool_result_data_supports_both_shapes() -> None:
    assert tool_result_data(_OldShapeResult()) == {"success": True, "data": {"ok": 1}}
    assert tool_result_data(_NewShapeContent()) == {"success": True}
