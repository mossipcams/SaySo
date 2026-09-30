
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.wake_mine_check import classify, phrase_spans


def _words(*items):
    return [{"word": w, "start": s, "end": e} for w, s, e in items]


def test_phrase_match_is_exact_word():
    words = _words(("At", 0.6, 0.7), ("last,", 0.7, 1.0), ("Atlas.", 1.9, 2.3))
    assert phrase_spans(words, "Atlas") == [(1.9, 2.3)]
    assert phrase_spans(_words(("atlases", 1.0, 1.5)), "Atlas") == []


def test_classify_by_fire_and_phrase():
    assert classify(True, True) == "wake"
    assert classify(False, True) == "missed_wake"
    assert classify(True, False) == "false_positive"
    assert classify(False, False) == "near_miss"
