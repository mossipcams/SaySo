"""Conversation state TTL and per-satellite referent resolution tests."""

from sayso_server.conversation import (
    ConversationReferent,
    ConversationStore,
    LastIntent,
    LastTarget,
    ReferentKind,
)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_active_last_target_resolves_for_same_satellite() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=60.0, clock=clock)
    target = LastTarget(entity_ids=["light.living_room"])
    referent = store.record_last_target("macbook", target)

    resolved = store.resolve_last_target(referent, satellite_id="macbook")

    assert resolved == target
    assert referent.satellite_id == "macbook"
    assert referent.kind == ReferentKind.LAST_TARGET


def test_active_last_intent_resolves_for_same_satellite() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=60.0, clock=clock)
    intent = LastIntent(intent="turn off the lights", outcome="action")
    referent = store.record_last_intent("macbook", intent)

    resolved = store.resolve_last_intent(referent, satellite_id="macbook")

    assert resolved == intent
    assert referent.kind == ReferentKind.LAST_INTENT


def test_expired_last_target_does_not_resolve() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=30.0, clock=clock)
    referent = store.record_last_target("macbook", LastTarget(entity_ids=["light.kitchen"]))
    clock.advance(31.0)

    assert store.resolve_last_target(referent, satellite_id="macbook") is None


def test_expired_last_intent_does_not_resolve() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=30.0, clock=clock)
    referent = store.record_last_intent(
        "macbook",
        LastIntent(intent="turn on the lamp", outcome="action"),
    )
    clock.advance(30.1)

    assert store.resolve_last_intent(referent, satellite_id="macbook") is None


def test_cross_satellite_last_target_does_not_resolve() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=60.0, clock=clock)
    referent = store.record_last_target("satellite-a", LastTarget(entity_ids=["light.office"]))

    assert store.resolve_last_target(referent, satellite_id="satellite-b") is None


def test_cross_satellite_last_intent_does_not_resolve() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=60.0, clock=clock)
    referent = store.record_last_intent(
        "satellite-a",
        LastIntent(intent="dim the lights", outcome="action"),
    )

    assert store.resolve_last_intent(referent, satellite_id="satellite-b") is None


def test_superseded_referent_does_not_resolve() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=60.0, clock=clock)
    first = store.record_last_target("macbook", LastTarget(entity_ids=["light.one"]))
    store.record_last_target("macbook", LastTarget(entity_ids=["light.two"]))

    assert store.resolve_last_target(first, satellite_id="macbook") is None


def test_per_satellite_state_is_isolated() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=60.0, clock=clock)
    store.record_last_target("satellite-a", LastTarget(entity_ids=["light.a"]))
    ref_b = store.record_last_target("satellite-b", LastTarget(entity_ids=["light.b"]))

    assert store.resolve_last_target(ref_b, satellite_id="satellite-b") == LastTarget(
        entity_ids=["light.b"]
    )
    state_a = store.get_state("satellite-a")
    assert state_a.last_target == LastTarget(entity_ids=["light.a"])


def test_conversation_referent_round_trips() -> None:
    referent = ConversationReferent(
        satellite_id="macbook",
        kind=ReferentKind.LAST_TARGET,
        recorded_at=42.0,
        generation=1,
    )
    dumped = referent.model_dump(mode="json")
    assert ConversationReferent.model_validate(dumped) == referent
