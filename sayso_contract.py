
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

_PACKAGE = "_sayso_pure"
if _PACKAGE not in sys.modules:
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parent / "custom_components" / "sayso")]
    sys.modules[_PACKAGE] = package

area_context = importlib.import_module(f"{_PACKAGE}.area_context")
completion = importlib.import_module(f"{_PACKAGE}.completion")
exceptions = importlib.import_module(f"{_PACKAGE}.exceptions")
tool_contract = importlib.import_module(f"{_PACKAGE}.tool_contract")
tool_schema = importlib.import_module(f"{_PACKAGE}.tool_schema")
