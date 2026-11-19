from custom_components.sayso.core.grounding import GroundingEntity
from custom_components.sayso.core.held import (
    HeldState,
    is_cancel,
    is_expired,
    resolve_held_option,
)


def _entity(entity_id: str, name: str, area: str) -> GroundingEntity:
    return GroundingEntity(
        entity_id=entity_id,
        name=name,
        aliases=(),
        domain="light",
        area=area,
        device_id=None,
    )


KITCHEN = _entity("light.kitchen_ceiling", "Kitchen Ceiling Light", "kitchen")
BEDROOM = _entity("light.bedroom_ceiling", "Bedroom Ceiling Light", "bedroom")


def _held(*options: GroundingEntity, expires: float = 100.0) -> HeldState:
    return HeldState(
        calls=[],
        index=0,
        options=tuple(options),
        expires=expires,
    )


def test_is_cancel_true_for_no_answers() -> None:
    assert is_cancel("no")
    assert is_cancel("cancel")
    assert is_cancel("no thanks")
    assert is_cancel("never mind")


def test_is_cancel_false_for_yes_or_mixed_or_empty() -> None:
    assert not is_cancel("yes")
    assert not is_cancel("no, yes")
    assert not is_cancel("the first one")
    assert not is_cancel("")


def test_is_expired_boundary_is_expired_at_expires() -> None:
    held = _held(expires=10.0)
    assert not is_expired(held, 9.99)
    assert is_expired(held, 10.0)
    assert is_expired(held, 10.01)


def test_resolve_single_option_yes_returns_it() -> None:
    held = _held(KITCHEN)
    assert resolve_held_option(held, "yes") is KITCHEN


def test_resolve_two_options_yes_is_ambiguous() -> None:
    held = _held(KITCHEN, BEDROOM)
    assert resolve_held_option(held, "yes") is None


def test_resolve_two_options_by_name() -> None:
    held = _held(KITCHEN, BEDROOM)
    assert resolve_held_option(held, "the bedroom ceiling light") is BEDROOM
    assert resolve_held_option(held, "kitchen ceiling light") is KITCHEN


def test_resolve_two_options_unrelated_answer_returns_none() -> None:
    held = _held(KITCHEN, BEDROOM)
    assert resolve_held_option(held, "what is the weather") is None
