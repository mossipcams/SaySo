"""Tests for SaySo Home Assistant integration constants."""

from custom_components.sayso import const


def test_conversation_message_types_are_registered() -> None:
    assert const.MSG_CONVERSATION_REQUEST == "conversation_request"
    assert const.MSG_CONVERSATION_RESPONSE == "conversation_response"
