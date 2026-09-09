"""Both public rendering paths preserve authoritative request constraints."""
import sys
import random
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generators.utterances import expand_utterance
from generators.utterances import vary_training_utterance
from scripts.build_synthetic_dataset import expand_utterance as legacy_expand


@pytest.mark.parametrize("render", [expand_utterance, legacy_expand])
@pytest.mark.parametrize("calls,targets,required", [
    ([{"name": "HassTurnOn", "arguments": {"name": "Desk"}},
      {"name": "HassTurnOff", "arguments": {"name": "Bed"}}], ["Desk", "Bed"], ["reading lamp", "Bed", "Window"]),
    ([{"name": "HassTurnOn", "arguments": {"floor": "Upstairs", "domain": ["light"]}}], [], ["Upstairs", "light"]),
    ([{"name": "HassLightSet", "arguments": {"name": "Desk", "brightness": 35, "color": "red"}}], ["Desk"], ["reading lamp", "35", "red"]),
    ([{"name": "HassStartTimer", "arguments": {"hours": 1, "minutes": 20, "seconds": 15, "name": "pasta"}}], [], ["1", "20", "15", "pasta"]),
    ([{"name": "HassLightSet", "arguments": {"name": "Desk", "area": "Study", "floor": "Upstairs", "brightness": 35, "color": "red"}}], ["Desk"], ["reading lamp", "Study", "Upstairs", "35", "red"]),
    ([{"name": "HassStartTimer", "arguments": {"minutes": 20, "name": "pasta", "area": "Kitchen"}}], [], ["20", "pasta", "Kitchen"]),
])
def test_renderers_preserve_constraints(render, calls, targets, required):
    spec = {"candidate_id": "grammar-regression", "category": "clean_direct",
            "expected": {"kind": "action", "calls": calls},
            "target_names": targets, "spoken_targets": {"Desk": "reading lamp"},
            "excluded_names": ["Window"] if len(calls) > 1 else []}
    text = render(spec)
    assert all(value in text for value in required), text


def test_renderers_share_deterministic_grammar():
    spec = {"candidate_id": "same-seed", "category": "clean_direct",
            "expected": {"kind": "action", "calls": [{"name": "HassTurnOn", "arguments": {"name": "Desk", "domain": ["light"]}}]},
            "target_names": ["Desk"], "spoken_targets": {}, "excluded_names": []}
    assert legacy_expand(spec) == expand_utterance(spec)


def test_temperature_style_does_not_duplicate_units():
    for seed in range(20):
        text = vary_training_utterance("set Desk color temperature to 2700 kelvin", random.Random(seed))
        assert text.count("kelvin") == 1, text
