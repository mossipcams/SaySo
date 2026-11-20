from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_LANGUAGE = "en"


@dataclass(frozen=True, slots=True)
class Speech:
    not_understood: str = "Sorry, I didn't understand that."
    did_you_mean: str = "Did you mean the {name}?"
    choose: str = "The {first} or the {second}?"
    area_has_no_domain: str = "The {area} has no {domains} available."
    model_unavailable: str = "The local model is unavailable."
    model_not_loaded: str = "The local model is not loaded."
    request_timeout: str = "That request took too long."
    empty_response: str = "I didn't get a response from the local model."
    action_failed: str = "I couldn't complete that action."
    no_timers: str = "There are no timers running."
    name_not_found: str = "I couldn't find {name}."
    media_not_playing: str = "The {name} isn't playing."
    media_not_paused: str = "The {name} isn't paused."
    tool_iteration_limit: str = "I couldn't complete that action."
    cancelled: str = "Okay."
    done: str = "Done."


SPEECH_EN = Speech()

DOMAIN_PLURALS = {
    "light": "lights",
    "fan": "fans",
    "switch": "switches",
    "media_player": "media players",
    "cover": "covers",
    "climate": "thermostats",
    "lock": "locks",
    "vacuum": "vacuums",
    "humidifier": "humidifiers",
}

GENERIC = frozenset(
    {
        "the",
        "tv",
        "light",
        "lights",
        "lamp",
        "switch",
        "plug",
        "fan",
        "speaker",
        "volume",
    }
)

NOUN = {
    "light": "light",
    "lights": "light",
    "lamp": "light",
    "lamps": "light",
    "tv": "media_player",
    "fan": "fan",
    "volume": "media_player",
}

DOMAIN_WORDS = {
    **NOUN,
    "television": "media_player",
    "vacuum": "vacuum",
    "vacuums": "vacuum",
}

STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "my",
        "our",
        "your",
        "this",
        "that",
        "its",
        "on",
        "off",
        "up",
        "down",
        "to",
        "in",
        "at",
        "of",
        "for",
        "and",
        "bit",
        "little",
        "back",
        "is",
        "it",
        "can",
        "you",
        "please",
        "turn",
        "switch",
        "set",
        "dim",
        "mute",
        "pause",
        "unpause",
        "resume",
        "start",
        "send",
        "dock",
        "have",
        "run",
        "clean",
        "brighten",
        "make",
        "lower",
        "raise",
        "open",
        "close",
        "lock",
        "unlock",
        "play",
        "stop",
        "go",
        "return",
        "put",
        "change",
        "volume",
        "percent",
        "degrees",
        "some",
        "all",
        "any",
    }
)

YES_ANSWER = frozenset({"yes", "yeah", "yep", "yup", "sure", "correct", "ok", "okay"})
NO_ANSWER = frozenset(
    {
        "no",
        "nope",
        "nah",
        "cancel",
        "neither",
        "never",
        "mind",
        "stop",
    }
)
FILLER = frozenset(
    {"the", "it", "that", "one", "please", "do", "thanks", "thank", "you"}
)


@dataclass(frozen=True, slots=True)
class Vocabulary:
    generic: frozenset[str]
    noun: Mapping[str, str]
    domain_words: Mapping[str, str]
    stop_words: frozenset[str]
    yes: frozenset[str]
    no: frozenset[str]
    filler: frozenset[str]
    domain_plurals: Mapping[str, str]
    possessive: str = "'s"


VOCAB_EN = Vocabulary(
    generic=GENERIC,
    noun=NOUN,
    domain_words=DOMAIN_WORDS,
    stop_words=STOP_WORDS,
    yes=YES_ANSWER,
    no=NO_ANSWER,
    filler=FILLER,
    domain_plurals=DOMAIN_PLURALS,
)

VOCABULARY: Mapping[str, Vocabulary] = {DEFAULT_LANGUAGE: VOCAB_EN}


def language_code(language: str) -> str:
    base = language.split("-")[0].casefold()
    return base if base in VOCABULARY else DEFAULT_LANGUAGE


def vocabulary_for(language: str) -> Vocabulary:
    return VOCABULARY[language_code(language)]


ERROR_NOT_UNDERSTOOD = SPEECH_EN.not_understood
ERROR_DID_YOU_MEAN = SPEECH_EN.did_you_mean
ERROR_CHOOSE = SPEECH_EN.choose
ERROR_AREA_HAS_NO_DOMAIN = SPEECH_EN.area_has_no_domain
ERROR_MODEL_UNAVAILABLE = SPEECH_EN.model_unavailable
ERROR_MODEL_NOT_LOADED = SPEECH_EN.model_not_loaded
ERROR_REQUEST_TIMEOUT = SPEECH_EN.request_timeout
ERROR_EMPTY_RESPONSE = SPEECH_EN.empty_response
ERROR_ACTION_FAILED = SPEECH_EN.action_failed
ERROR_NO_TIMERS = SPEECH_EN.no_timers
ERROR_NAME_NOT_FOUND = SPEECH_EN.name_not_found
ERROR_MEDIA_NOT_PLAYING = SPEECH_EN.media_not_playing
ERROR_MEDIA_NOT_PAUSED = SPEECH_EN.media_not_paused
ERROR_TOOL_ITERATION_LIMIT = SPEECH_EN.tool_iteration_limit
ERROR_CANCELLED = SPEECH_EN.cancelled
