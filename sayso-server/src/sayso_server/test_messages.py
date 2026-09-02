"""SaySo API v1 message type registry tests."""

from sayso_server.messages import MESSAGE_TYPES_V1, MessageType


def test_message_types_are_stable_strings() -> None:
    assert MessageType.HELLO.value == "hello"
    assert MessageType.PING.value == "ping"
    assert set(MESSAGE_TYPES_V1) == {member.value for member in MessageType}


def test_home_graph_message_types_are_registered() -> None:
    for graph_type in ("graph_snapshot", "state_delta", "registry_delta"):
        assert graph_type in MESSAGE_TYPES_V1


def test_action_message_types_are_registered() -> None:
    for action_type in ("action_request", "action_result"):
        assert action_type in MESSAGE_TYPES_V1


def test_conversation_message_types_are_registered() -> None:
    for conversation_type in ("conversation_request", "conversation_response"):
        assert conversation_type in MESSAGE_TYPES_V1


def test_prepare_message_types_are_registered() -> None:
    for prepare_type in ("prepare", "prepare_response"):
        assert prepare_type in MESSAGE_TYPES_V1
