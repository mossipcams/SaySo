"""Compatibility shim: the LFM2 tool-call parser now ships in the integration.

The runtime embedded backend and the training scorer must parse LFM2 output
identically, so there is exactly one implementation and it lives with the code
that serves users. Import it from here or from
``custom_components.sayso.lfm_parse`` — they are the same objects.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from custom_components.sayso.lfm_parse import (  # noqa: E402
    LfmPythonParseError,
    parse_lfm_python_tool_call,
    parse_lfm_python_tool_calls,
)

__all__ = [
    "LfmPythonParseError",
    "parse_lfm_python_tool_call",
    "parse_lfm_python_tool_calls",
]
