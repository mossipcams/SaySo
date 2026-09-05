# Re-export metrics tests for discoverability at evals/test_metrics.py
from pathlib import Path
import sys

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from tests.test_metrics import *  # noqa: F403

from evals.metrics import score_expected_vs_actual  # noqa: E402


def test_args_match_when_entity_name_differs_only_by_curly_apostrophe() -> None:
    ascii_name = "Joe's Guest Room Door Lock"
    curly_name = "Joe\u2019s Guest Room Door Lock"
    expected = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "HassTurnOn",
                        "arguments": f'{{"name": "{ascii_name}", "domain": ["lock"]}}',
                    }
                }
            ],
        }
    ]
    actual = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "HassTurnOn",
                        "arguments": f'{{"name": "{curly_name}", "domain": ["lock"]}}',
                    }
                }
            ],
        }
    ]
    _name_ok, args_ok, multi_ok, category = score_expected_vs_actual(expected, actual)
    assert _name_ok
    assert args_ok
    assert multi_ok
    assert category is None


def test_args_fail_on_real_entity_name_mismatch() -> None:
    expected = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "HassTurnOn",
                        "arguments": '{"name": "Joe\'s Guest Room Door Lock", "domain": ["lock"]}',
                    }
                }
            ],
        }
    ]
    actual = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "HassTurnOn",
                        "arguments": '{"name": "Joe\'s Kitchen Light", "domain": ["lock"]}',
                    }
                }
            ],
        }
    ]
    name_ok, args_ok, multi_ok, category = score_expected_vs_actual(expected, actual)
    assert name_ok
    assert not args_ok
    assert not multi_ok
    assert category == "args_mismatch"
