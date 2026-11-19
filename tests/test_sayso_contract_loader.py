from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import sayso_contract

SAYSO_DIR = (Path(__file__).resolve().parents[1] / "custom_components" / "sayso").resolve()


def _inside_sayso(path: str | None) -> bool:
    return path is not None and Path(path).resolve().is_relative_to(SAYSO_DIR)


def test_sys_path_does_not_include_integration_dir() -> None:
    for entry in sys.path:
        assert Path(entry or ".").resolve() != SAYSO_DIR


def test_integration_modules_do_not_shadow_top_level_names() -> None:
    for name in ("schema", "client", "inference", "const"):
        spec = importlib.util.find_spec(name)
        assert spec is None or not _inside_sayso(spec.origin), name


def test_core_is_loaded_as_private_package() -> None:
    assert sayso_contract.tool_schema.__name__ == "_sayso_core.tool_schema"
