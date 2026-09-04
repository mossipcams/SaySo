# Re-export harness tests for discoverability at evals/test_harness.py
from pathlib import Path
import sys

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from tests.test_harness import *  # noqa: F403
