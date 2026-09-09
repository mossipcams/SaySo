"""Linguistic validation must retain action values, including STT number words."""

import pytest

from generators.validate import validate_utterance


@pytest.mark.parametrize("text", ["set Reading Lamp brightness", "set Reading Lamp to 350 percent"])
def test_reject_missing_brightness(text):
    spec = {"utterance": text, "expected": {"kind": "action", "calls": [
        {"name": "HassLightSet", "arguments": {"name": "Reading Lamp", "brightness": 35}}
    ]}, "target_names": ["Reading Lamp"]}
    assert validate_utterance(spec) == "missing_expected_value"


@pytest.mark.parametrize("value", ["35", "thirty five"])
def test_accept_numeric_or_spoken_brightness(value):
    spec = {"utterance": f"set Reading Lamp to {value} percent", "expected": {
        "kind": "action", "calls": [{"name": "HassLightSet", "arguments": {
            "name": "Reading Lamp", "brightness": 35}}]}, "target_names": ["Reading Lamp"]}
    assert validate_utterance(spec) is None


def test_reject_dropped_timer_duration_component():
    spec = {"utterance": "start a 2 minute timer", "expected": {"kind": "action", "calls": [
        {"name": "HassStartTimer", "arguments": {"hours": 1, "minutes": 2, "seconds": 3}}
    ]}}
    assert validate_utterance(spec) == "missing_expected_value"
