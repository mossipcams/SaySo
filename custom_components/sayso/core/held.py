from __future__ import annotations

from dataclasses import dataclass

from .completion import ToolCall
from .grounding import GroundingEntity, support, tokens

YES_ANSWER = {"yes", "yeah", "yep", "yup", "sure", "correct", "ok", "okay"}
NO_ANSWER = {
    "no",
    "nope",
    "nah",
    "cancel",
    "neither",
    "never",
    "mind",
    "stop",
}
FILLER = {"the", "it", "that", "one", "please", "do", "thanks", "thank", "you"}


@dataclass(frozen=True)
class HeldState:
    calls: list[ToolCall]
    index: int
    options: tuple[GroundingEntity, ...]
    expires: float


def is_cancel(text: str) -> bool:
    words = tokens(text)
    return (
        bool(words)
        and all(word in YES_ANSWER | NO_ANSWER | FILLER for word in words)
        and any(word in NO_ANSWER for word in words)
        and not any(word in YES_ANSWER for word in words)
    )


def is_expired(held: HeldState, now: float) -> bool:
    return not now < held.expires


def resolve_held_option(held: HeldState, text: str) -> GroundingEntity | None:
    words = tokens(text)
    if (
        len(held.options) == 1
        and any(word in YES_ANSWER for word in words)
        and all(word in YES_ANSWER | FILLER for word in words)
    ):
        return held.options[0]
    scored = [
        (
            max(
                support(label.casefold(), text)
                for label in (option.name, *option.aliases)
            ),
            option,
        )
        for option in held.options
    ]
    best_score, best_option = max(scored, key=lambda pair: pair[0])
    if best_score >= 0.5 and all(
        best_score > score for score, option in scored if option is not best_option
    ):
        return best_option
    return None
