from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import wake_librispeech_mine as mine
from sayso.wake.detection import Detection


def test_classify() -> None:
    assert mine.classify(["AND IF YOU SAY SO THEN"]) == ("excluded_say_so", ["SAY SO"])
    assert mine.classify(["WE SAY SOMETHING", "IT IS SAFE"]) == ("hard_negative", ["SAFE", "SAY SOMETHING"])
    assert mine.classify(["ESSAY SOLUTIONS SOMETIMES"]) == ("false_positive", [])


class _FakeProvider:
    def __init__(self, fire_at: set[int]) -> None:
        self.fire_at = fire_at

    def reset(self) -> None:
        pass

    def start(self) -> None:
        pass

    def predict_window(self, window, sample_index=None):
        if sample_index in self.fire_at:
            return Detection("SaySo", 0.9, 0.0, sample_index)
        return None


def test_mine_chapter_saves_activation_with_context(tmp_path: Path, monkeypatch) -> None:
    sr = mine.SAMPLE_RATE
    spans = [("1-2-0000", 0, 3 * sr, "WHY DO YOU SAY SO"), ("1-2-0001", 3 * sr, 12 * sr, "THE SAME OLD STORY")]
    monkeypatch.setattr(mine, "load_chapter", lambda _d: (np.zeros(12 * sr, dtype=np.int16), spans))
    fire = mine.WINDOW_SAMPLES + mine.HOP_SAMPLES * 30  # window lies wholly in utterance 2
    monkeypatch.setattr(mine, "_provider", _FakeProvider({fire}))
    (tmp_path / "chapters").mkdir()

    summary = mine.mine_chapter(tmp_path / "1" / "2", tmp_path)

    assert summary["activations"] == {"false_positive": 0, "hard_negative": 1, "excluded_say_so": 0}
    assert summary["excluded_say_so_seconds"] == 3.0
    rec_dir = next((tmp_path / "activations" / "hard_negative").iterdir())
    record = json.loads((rec_dir / "record.json").read_text())
    assert record["hard_terms"] == ["SAME"] and record["confidence"] == 0.9
    assert record["context_end"] - record["context_start"] == mine.WINDOW_SAMPLES + 2 * mine.CONTEXT_MS * sr // 1000
    s = mine.summarize(tmp_path, model=Path(mine.__file__), threshold=0.25, refractory=2.0)
    assert s["false_activations_per_hour"] == round(1 / (9 / 3600), 4)
