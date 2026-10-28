from homeassistant.components.intent.timers import StartTimerIntentHandler
from homeassistant.helpers import llm

from .schema import ToolArgumentFailureCode, validate_tool_arguments


def _timer_tool() -> llm.IntentTool:
    return llm.IntentTool("HassStartTimer", StartTimerIntentHandler())


def test_timer_accepts_any_wrapped_required_duration_name():
    for arguments in ({"minutes": 5}, {"hours": 1}, {"seconds": 30, "name": "pasta"}):
        normalized, error = validate_tool_arguments(_timer_tool(), arguments)
        assert error is None
        assert normalized is not None


def test_timer_rejects_missing_required_duration():
    normalized, error = validate_tool_arguments(_timer_tool(), {})
    assert normalized is None
    assert error is not None
    assert error.code is ToolArgumentFailureCode.SCHEMA_MISMATCH
    assert "required" in error.message.lower()


def test_timer_rejects_unexpected_argument():
    normalized, error = validate_tool_arguments(
        _timer_tool(), {"minutes": 5, "area": "Kitchen"}
    )
    assert normalized is None
    assert error is not None
    assert error.code is ToolArgumentFailureCode.SCHEMA_MISMATCH
    assert "Unexpected argument" in error.message
