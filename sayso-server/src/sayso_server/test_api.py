"""SaySo API v1 surface constants."""

from sayso_server.api import API_VERSION, PROTOCOL_NAME


def test_api_version_is_one() -> None:
    assert API_VERSION == 1


def test_protocol_name_is_stable() -> None:
    assert PROTOCOL_NAME == "sayso-api"
