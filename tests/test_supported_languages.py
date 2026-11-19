from __future__ import annotations

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.conversation import SaySoConversationEntity


def test_supported_languages_is_english_only() -> None:
    agent = SaySoConversationEntity(MockConfigEntry(domain="sayso"))
    assert agent.supported_languages == ["en"]
