"""STT noise transformations with protected slots."""

from __future__ import annotations

import random
import re
from typing import Any

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_WORD_TO_INT: dict[str, int] = {word: idx for idx, word in enumerate(_ONES)}
for tens_idx, tens_word in enumerate(_TENS[2:], start=2):
    if not tens_word:
        continue
    _WORD_TO_INT[tens_word] = tens_idx * 10
    for ones_idx, ones_word in enumerate(_ONES[1:10], start=1):
        _WORD_TO_INT[f"{tens_word} {ones_word}"] = tens_idx * 10 + ones_idx


def _int_to_words(value: int) -> str | None:
    if value == 100:
        return "one hundred"
    if 0 <= value < len(_ONES):
        return _ONES[value]
    if 10 <= value < 100 and value % 10 == 0:
        return _TENS[value // 10]
    if 10 < value < 100:
        tens, ones = divmod(value, 10)
        return f"{_TENS[tens]} {_ONES[ones]}"
    return None


def _number_variant(match: re.Match[str]) -> str:
    token = match.group(1)
    value = int(token)
    words = _int_to_words(value)
    if words:
        return words
    return token


def _word_number_variant(match: re.Match[str]) -> str:
    token = match.group(0).casefold()
    value = _WORD_TO_INT.get(token)
    if value is not None:
        return str(value)
    return match.group(0)


def _corruption_pool() -> tuple[tuple[re.Pattern[str], Any, str], ...]:
    return (
        (re.compile(r"\blight\b", re.I), "lite", "homophone_light"),
        (re.compile(r"\bfan\b", re.I), "van", "consonant_fan"),
        (re.compile(r"\boutlet\b", re.I), "out let", "word_boundary_outlet"),
        (re.compile(r"\bblinds\b", re.I), "blends", "vowel_blinds"),
        (re.compile(r"\bgarage\b", re.I), "garaj", "phonetic_garage"),
        (re.compile(r"\block\b", re.I), "lok", "phonetic_lock"),
        (re.compile(r"\bthe\b", re.I), "", "dropped_article"),
        (re.compile(r"\b(\d{1,3})\b"), _number_variant, "number_variant"),
        (
            re.compile(
                r"\b("
                + "|".join(sorted(_WORD_TO_INT.keys(), key=len, reverse=True))
                + r")\b",
                re.I,
            ),
            _word_number_variant,
            "number_variant",
        ),
    )


_CORRUPTIONS = _corruption_pool()


def apply_stt_noise(
    utterance: str,
    rng: random.Random,
    *,
    target_names: list[str] | None = None,
    force_transform: bool = False,
) -> tuple[str, str | None]:
    """Return corrupted utterance and corruption kind, protecting action words."""
    order = list(_CORRUPTIONS)
    rng.shuffle(order)
    for pattern, replacement, kind in order:
        corrupted = pattern.sub(replacement if callable(replacement) else replacement, utterance)
        if corrupted == utterance:
            continue
        if target_names:
            protected = True
            for name in target_names:
                fragment = name.split()[0].casefold()
                if fragment and fragment not in corrupted.casefold() and fragment not in utterance.casefold():
                    protected = False
                    break
            if not protected:
                continue
        return corrupted.strip(), kind
    if force_transform and utterance:
        return utterance.replace(" the ", " ", 1).strip(), "forced_article_drop"
    return utterance, None
