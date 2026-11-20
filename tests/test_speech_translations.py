from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import translation

from custom_components.sayso.const import DOMAIN
from custom_components.sayso.conversation import _turn_speech, execution_failure_speech
from custom_components.sayso.core.completion import ToolCall
from custom_components.sayso.core.held import HeldState, is_cancel
from custom_components.sayso.core.vocabulary import SPEECH_EN, VOCAB_EN, vocabulary_for
from tests.test_post_model_outcome import (
    DESK_LAMP,
    FLOOR_LAMP,
    _call,
    _home,
    _Model,
    _reply,
    _run,
    _turn,
)

INTEGRATION_ROOT = Path(__file__).resolve().parents[1] / "custom_components" / DOMAIN


def _fake_translations(
    monkeypatch: Any, strings: dict[str, str]
) -> list[tuple[str, str, list[str]]]:
    calls: list[tuple[str, str, list[str]]] = []

    async def fake(
        hass: HomeAssistant,
        language: str,
        category: str,
        integrations: Any = None,
        config_flow: Any = None,
    ) -> dict[str, str]:
        calls.append((language, category, list(integrations)))
        return {
            f"component.{DOMAIN}.{category}.{key}": value
            for key, value in strings.items()
        }

    monkeypatch.setattr(translation, "async_get_translations", fake)
    return calls


def test_en_speech_strings_match_defaults_in_both_files() -> None:
    en = json.loads((INTEGRATION_ROOT / "translations" / "en.json").read_text())
    strings = json.loads((INTEGRATION_ROOT / "strings.json").read_text())

    assert en["speech"] == asdict(SPEECH_EN)
    assert strings["speech"] == en["speech"]


async def test_spoken_strings_resolve_through_translations(hass: HomeAssistant) -> None:
    strings = await translation.async_get_translations(hass, "en", "speech", [DOMAIN])

    assert strings[f"component.{DOMAIN}.speech.done"] == "Done."
    assert await _turn_speech(hass, "en-US") == SPEECH_EN


async def test_turn_speech_asks_translations_for_the_base_language(
    hass: HomeAssistant, monkeypatch: Any
) -> None:
    calls = _fake_translations(monkeypatch, {"not_understood": "Pardon?"})

    speech = await _turn_speech(hass, "en-US")

    assert calls == [("en", "speech", [DOMAIN])]
    assert speech.not_understood == "Pardon?"


async def test_missing_translation_key_falls_back_to_english(
    hass: HomeAssistant, monkeypatch: Any
) -> None:
    _fake_translations(monkeypatch, {"done": "Fine."})

    speech = await _turn_speech(hass, "en")

    assert speech.done == "Fine."
    assert speech.not_understood == SPEECH_EN.not_understood


async def test_unknown_language_uses_english_speech(hass: HomeAssistant) -> None:
    assert await _turn_speech(hass, "xx") == SPEECH_EN


def test_vocabulary_falls_back_to_english() -> None:
    assert vocabulary_for("xx") is VOCAB_EN
    assert vocabulary_for("en-GB") is VOCAB_EN
    assert is_cancel("no thanks", vocabulary_for("xx"))


async def test_speech_table_reaches_done_text() -> None:
    speech = replace(SPEECH_EN, done="Fine.")
    model = _Model(_reply(_call("intent__HassTurnOn", {"name": "Desk Lamp"})))

    outcome, _ = await _run(_turn(home=lambda: _home(DESK_LAMP), speech=speech), model)

    assert outcome.speech == "Fine."
    assert outcome.done_shortcut is True


async def test_speech_table_reaches_clarify_text() -> None:
    speech = replace(SPEECH_EN, not_understood="Pardon?")
    model = _Model(_reply(_call("intent__HassTurnOn", {"name": "Ghost Lamp"})))

    outcome, executed = await _run(
        _turn(
            user_text="turn on the ghost lamp",
            home=lambda: _home(DESK_LAMP, FLOOR_LAMP),
            speech=speech,
        ),
        model,
    )

    assert outcome.clarified is True
    assert outcome.speech == "Pardon?"
    assert executed == []


async def test_speech_table_reaches_cancel_text() -> None:
    speech = replace(SPEECH_EN, cancelled="Fine.")
    held = HeldState(
        [_call("intent__HassTurnOn", {"name": "Desk Lamp"})],
        0,
        (DESK_LAMP, FLOOR_LAMP),
        10.0,
    )

    outcome, _ = await _run(
        _turn(
            user_text="no",
            home=lambda: _home(DESK_LAMP, FLOOR_LAMP),
            held=held,
            speech=speech,
        ),
        _Model(),
    )

    assert outcome.speech == "Fine."


async def test_unknown_turn_language_cancels_with_english_vocabulary() -> None:
    held = HeldState(
        [_call("intent__HassTurnOn", {"name": "Desk Lamp"})],
        0,
        (DESK_LAMP, FLOOR_LAMP),
        10.0,
    )

    outcome, _ = await _run(
        _turn(
            user_text="no",
            language="xx",
            home=lambda: _home(DESK_LAMP, FLOOR_LAMP),
            held=held,
        ),
        _Model(),
    )

    assert outcome.speech == "Okay."


async def test_execution_failure_speech_reads_speech_table(hass: HomeAssistant) -> None:
    speech = replace(SPEECH_EN, action_failed="Fine.")
    calls = [
        (ToolCall(id="a", name="intent__HassTurnOn", arguments={}), {}),
        (ToolCall(id="b", name="intent__HassTurnOn", arguments={}), {}),
    ]

    assert execution_failure_speech(hass, None, calls, None, speech=speech) == "Fine."
