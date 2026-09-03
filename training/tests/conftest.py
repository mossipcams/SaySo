"""Pytest path setup for training package tests."""

from __future__ import annotations

import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent

for path in (str(TRAINING_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
