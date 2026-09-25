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

# Whisper substitutions observed on the live SaySo pipeline (librarian TV) plus
# the household's reported living→ribbon class. Keys are what STT wrote.
_ASR_TOKEN_MAP = {
    "lite": "light",
    "van": "fan",
    "blends": "blinds",
    "lok": "lock",
    "garaj": "garage",
    "teevee": "tv",
}


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


def _stem(token: str) -> str:
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def fold_stt_text(text: str) -> str:
    """Collapse log-style STT spelling so gold can still see the intended name."""
    folded = text.casefold()
    folded = re.sub(r"\blibrarian\b", "living room", folded)
    folded = re.sub(r"\bribbon\b", "living", folded)
    folded = re.sub(r"t\s*\.?\s*v", "tv", folded)
    tokens = re.findall(r"[a-z0-9]+", folded)
    mapped = [_stem(_ASR_TOKEN_MAP.get(token, token)) for token in tokens]
    return " ".join(mapped)


def utterance_contains_target(text: str, name: str) -> bool:
    """Whether ``text`` still refers to registry name ``name`` after STT-style splits."""
    folded = fold_stt_text(text)
    canonical = fold_stt_text(name)
    if not canonical:
        return True
    if canonical in folded or canonical.replace(" ", "") in folded.replace(" ", ""):
        return True
    name_tokens = [token for token in canonical.split() if token not in {"the", "a", "an"}]
    text_tokens = set(folded.split())
    if name_tokens and all(token in text_tokens for token in name_tokens):
        return True
    tokens = name.casefold().strip().split()
    if len(tokens) == 1 and tokens[0].isalpha() and len(tokens[0]) >= 2:
        letters = list(tokens[0])
        spaced = r"\b" + r"\s*\.?\s*".join(re.escape(letter) for letter in letters) + r"\b"
        if re.search(spaced, text.casefold()):
            return True
        if tokens[0] == "tv" and re.search(r"\bteevee\b", text.casefold()):
            return True
    return False


def _corruption_pool() -> tuple[tuple[re.Pattern[str], Any, str], ...]:
    return (
        (re.compile(r"\blight\b", re.I), "lite", "homophone_light"),
        (re.compile(r"\bfan\b", re.I), "van", "consonant_fan"),
        (re.compile(r"\boutlet\b", re.I), "out let", "word_boundary_outlet"),
        (re.compile(r"\bTV\b", re.I), "T V", "letter_spaced_tv"),
        (re.compile(r"\bTV\b", re.I), "T.V.", "punctuated_tv"),
        (re.compile(r"\bTV\b", re.I), "teevee", "phonetic_tv"),
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

_LIVING_ROOM = re.compile(r"\bliving room\b", re.I)
_FILLER = re.compile(
    r"\b(the|a|an|uh|um|please|just|like|could you|can you|for me)\b",
    re.I,
)


def _protected(utterance: str, corrupted: str, target_names: list[str] | None) -> bool:
    if not target_names:
        return True
    for name in target_names:
        if utterance_contains_target(utterance, name) and not utterance_contains_target(
            corrupted, name
        ):
            return False
    return True


def apply_log_stt_noise(
    utterance: str,
    rng: random.Random,
    *,
    target_names: list[str] | None = None,
) -> tuple[str, str | None]:
    """Apply one live-Whisper-shaped corruption. Gold must stay canonical."""
    candidates: list[tuple[str, str]] = []
    if _LIVING_ROOM.search(utterance):
        candidates.append((_LIVING_ROOM.sub("ribbon room", utterance, count=1), "log_living_ribbon"))
        candidates.append((_LIVING_ROOM.sub("librarian", utterance, count=1), "log_living_librarian"))
    dropped = _FILLER.sub("", utterance, count=1)
    if dropped != utterance:
        candidates.append((dropped, "log_dropped_filler"))
    order = list(_CORRUPTIONS[:9])  # homophones / TV / name splits, not numbers
    rng.shuffle(order)
    for pattern, replacement, kind in order:
        corrupted = pattern.sub(replacement if callable(replacement) else replacement, utterance, count=1)
        if corrupted != utterance:
            candidates.append((corrupted, f"log_{kind}"))
            break
    punct = utterance.rstrip()
    if not punct.endswith((".", "?", "!")):
        candidates.append((punct + ".", "log_trailing_period"))
    if utterance and not utterance.isupper():
        candidates.append((utterance.upper() if rng.random() < 0.5 else utterance.lower(), "log_casing"))
    rng.shuffle(candidates)
    for corrupted, kind in candidates:
        cleaned = " ".join(corrupted.split())
        if cleaned == " ".join(utterance.split()):
            continue
        if not _protected(utterance, cleaned, target_names):
            continue
        return cleaned, kind
    return utterance, None


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
        if not _protected(utterance, corrupted, target_names):
            continue
        # A dropped word must not leave a double space the model can key on.
        return " ".join(corrupted.split()), kind
    if force_transform and utterance:
        changed = utterance.replace(" the ", " ", 1).strip()
        return changed, "forced_article_drop" if changed != utterance else None
    return utterance, None
