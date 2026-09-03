# Re-export adapter tests for discoverability at adapters/test_home_llm_v2.py
from pathlib import Path
import sys

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from tests.test_adapter import *  # noqa: F403
