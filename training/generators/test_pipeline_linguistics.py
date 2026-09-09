"""Pipeline must preserve the selected grammar and spoken target."""

import random

from generators import pipeline
from generators.config import GeneratorConfig
from generators.duplicates import DuplicateTracker
from generators.scenarios import build_scenario
from generators.utterances import vary_training_utterance


def test_question_framing_and_kelvin_unit_are_not_duplicated():
    for seed in range(30):
        query = vary_training_utterance("how much time is left on the pasta timer", random.Random(seed))
        assert not any(prefix in query for prefix in ("please how", "could you how", "can you how"))
        setting = vary_training_utterance("set Desk Lamp color temperature to 2700 kelvin", random.Random(seed))
        assert setting.count("kelvin") == 1


def test_generated_rows_record_selected_grammar():
    result = pipeline.run_generation(GeneratorConfig(count=30, seed=31, stt_noise_rate=0))
    grammar = [entry for row in result["rows"] for entry in row["metadata"].get("linguistics", [])
               if entry.get("source", "").startswith("sentences/en/")]
    assert grammar
    assert all(entry["revision"] and entry["template"] for entry in grammar)
    assert all(row["metadata"]["linguistics"] for row in result["rows"]
               if any(message.get("tool_calls") for message in row["messages"]))
    assert result["stats"]["quality_audit"]["ohf_rows"] > 0
    assert result["stats"]["quality_audit"]["ohf_source_count"] > 0


def test_pipeline_preserves_alias_grammar_and_category(monkeypatch):
    scenario = next(
        scenario for index in range(100)
        if (scenario := build_scenario(index=index, seed=19, capability="lights",
            operation="turn_on", home_size=16, robustness="alias_distractor"))
        .get("spoken_targets")
    )
    spoken = next(iter(scenario["spoken_targets"].values()))
    categories = []

    def render(spec):
        categories.append(spec["category"])
        return f"switch {spoken} on"

    def unexpected_fallback(spec):
        raise AssertionError("valid alias grammar was replaced by fallback")

    monkeypatch.setattr(pipeline, "build_scenario", lambda **kwargs: scenario)
    monkeypatch.setattr(pipeline, "expand_utterance", render)
    monkeypatch.setattr(pipeline, "request_seed_from_spec", unexpected_fallback)
    row, reason = pipeline.generate_row(
        {"index": 0, "capability": "lights", "operation": "turn_on", "home_size": 16},
        GeneratorConfig(), random.Random(19), excluded=set(), dup_tracker=DuplicateTracker(),
    )
    assert reason is None
    assert row is not None
    assert categories == ["alias_distractor"]
    assert spoken.casefold() in row["messages"][1]["content"].casefold()
