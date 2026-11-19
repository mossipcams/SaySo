from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_PACKAGE = "_sayso_core"
_CORE_DIR = Path(__file__).resolve().parent / "custom_components" / "sayso" / "core"

if _PACKAGE not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _PACKAGE,
        _CORE_DIR / "__init__.py",
        submodule_search_locations=[str(_CORE_DIR)],
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_PACKAGE] = _module
    _spec.loader.exec_module(_module)

area_context = importlib.import_module(f"{_PACKAGE}.area_context")
completion = importlib.import_module(f"{_PACKAGE}.completion")
correction = importlib.import_module(f"{_PACKAGE}.correction")
exceptions = importlib.import_module(f"{_PACKAGE}.exceptions")
grammar = importlib.import_module(f"{_PACKAGE}.grammar")
grounding = importlib.import_module(f"{_PACKAGE}.grounding")
history = importlib.import_module(f"{_PACKAGE}.history")
post_model = importlib.import_module(f"{_PACKAGE}.post_model")
tool_contract = importlib.import_module(f"{_PACKAGE}.tool_contract")
tool_schema = importlib.import_module(f"{_PACKAGE}.tool_schema")
