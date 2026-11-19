from __future__ import annotations

import sys
from pathlib import Path

_SAYSO_PACKAGE_DIR = str(Path(__file__).resolve().parent / "custom_components" / "sayso")
if _SAYSO_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _SAYSO_PACKAGE_DIR)

from core import (
    area_context,
    completion,
    exceptions,
    history,
    tool_contract,
    tool_schema,
)
