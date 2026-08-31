"""SaySo API v1 message type registry tests."""

from sayso_server.messages import MESSAGE_TYPES_V1, MessageType


def test_message_types_are_stable_strings() -> None:
    assert MessageType.HELLO.value == "hello"
    assert MessageType.PING.value == "ping"
    assert set(MESSAGE_TYPES_V1) == {member.value for member in MessageType}


def test_home_graph_message_types_are_registered() -> None:
    for graph_type in ("graph_snapshot", "state_delta", "registry_delta"):
        assert graph_type in MESSAGE_TYPES_V1


def test_action_message_types_are_not_registered_yet() -> None:
    for future_type in ("action_request", "action_result"):
        assert future_type not in MESSAGE_TYPES_V1
