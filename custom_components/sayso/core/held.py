from __future__ import annotations

from dataclasses import dataclass

from .completion import ToolCall
from .grounding import GroundingEntity, support, tokens
from .vocabulary import VOCAB_EN, Vocabulary


@dataclass(frozen=True)
class HeldState:
    calls: list[ToolCall]
    index: int
    options: tuple[GroundingEntity, ...]
    expires: float


def is_cancel(text: str, vocab: Vocabulary = VOCAB_EN) -> bool:
    words = tokens(text)
    return (
        bool(words)
        and all(word in vocab.yes | vocab.no | vocab.filler for word in words)
        and any(word in vocab.no for word in words)
        and not any(word in vocab.yes for word in words)
    )


def is_expired(held: HeldState, now: float) -> bool:
    return not now < held.expires


def resolve_held_option(
    held: HeldState, text: str, vocab: Vocabulary = VOCAB_EN
) -> GroundingEntity | None:
    words = tokens(text)
    if (
        len(held.options) == 1
        and any(word in vocab.yes for word in words)
        and all(word in vocab.yes | vocab.filler for word in words)
    ):
        return held.options[0]
    scored = [
        (
            max(
                support(label.casefold(), text, vocab=vocab)
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
