from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.sayso.conversation import SaySoConversationEntity
from custom_components.sayso.core.held import HeldState


def _held(expires: float) -> HeldState:
    return HeldState(calls=[], index=0, options=(), expires=expires)


def _store(
    agent: SaySoConversationEntity, conversation_id: str, held: HeldState
) -> None:
    agent._respond(
        SimpleNamespace(
            chat_log=SimpleNamespace(conversation_id=conversation_id),
            text=lambda speech: speech,
        ),
        SimpleNamespace(
            correction_used=False,
            held=held,
            failure=None,
            speech="Which one?",
        ),
    )


def test_store_evicts_expired_held_for_other_conversations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = SaySoConversationEntity(
        MockConfigEntry(domain="sayso", entry_id="eviction-store")
    )
    monkeypatch.setattr(time, "monotonic", lambda: 1_000.0)
    _store(agent, "conv-a", _held(expires=1_100.0))
    _store(agent, "conv-b", _held(expires=1_100.0))
    _store(agent, "conv-keep", _held(expires=5_000.0))

    monkeypatch.setattr(time, "monotonic", lambda: 1_200.0)
    _store(agent, "conv-c", _held(expires=1_500.0))

    assert "conv-a" not in agent._held
    assert "conv-b" not in agent._held
    assert "conv-keep" in agent._held
    assert "conv-c" in agent._held
