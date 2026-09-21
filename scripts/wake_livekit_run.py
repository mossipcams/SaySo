"""Thin wrapper around livekit.wakeword CLI stages for SaySo training.

Stages: setup, generate, augment, train, export, eval, run.

``run`` always executes the full LiveKit pipeline in order and never skips
generate. ``data_dir`` and ``output_dir`` are rewritten under ``--work-dir``
so shipped living2 artifacts stay untouched.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "satellite" / "models" / "sayso.yaml"

STAGES = ("setup", "generate", "augment", "train", "export", "eval")
RUN_STAGES = STAGES


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def rewrite_config_for_work_dir(config: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    """Return a copy with data/output paths rooted under work_dir."""
    out = dict(config)
    data_dir = work_dir / "data"
    output_dir = work_dir / "output"
    out["data_dir"] = str(data_dir)
    out["output_dir"] = str(output_dir)

    augmentation = dict(out.get("augmentation") or {})
    augmentation["background_paths"] = [str(data_dir / "backgrounds")]
    augmentation["rir_paths"] = [str(data_dir / "rirs")]
    out["augmentation"] = augmentation
    return out


def write_work_config(source: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    rewritten = rewrite_config_for_work_dir(_load_yaml(source), work_dir)
    dest = work_dir / "recipe.yaml"
    dest.write_text(yaml.safe_dump(rewritten, sort_keys=False), encoding="utf-8")
    return dest


def run_stage(
    stage: str,
    config_path: Path,
    *,
    runner: Any | None = None,
) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    invoke = runner or _default_runner
    invoke(stage, config_path)


def _livekit_argv(stage: str, config_path: Path) -> list[str]:
    argv = [sys.executable, "-m", "livekit.wakeword", stage]
    if stage == "setup":
        argv.extend(["--config", str(config_path)])
    else:
        argv.append(str(config_path))
    return argv


def _default_runner(stage: str, config_path: Path) -> None:
    proc = subprocess.run(
        _livekit_argv(stage, config_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"livekit.wakeword {stage} failed: {detail}")


def run_pipeline(
    config_path: Path,
    work_dir: Path,
    *,
    runner: Any | None = None,
) -> list[str]:
    """Execute setup → generate → augment → train → export → eval."""
    recipe = write_work_config(config_path, work_dir)
    executed: list[str] = []
    for stage in RUN_STAGES:
        run_stage(stage, recipe, runner=runner)
        executed.append(stage)
    return executed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage",
        choices=[*STAGES, "run"],
        help="livekit.wakeword stage, or run for the full pipeline",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"recipe YAML (default: {_DEFAULT_CONFIG.relative_to(_REPO_ROOT)})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="isolated host output root (data/ and output/ are rewritten here)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = args.config.resolve()
    work_dir = args.work_dir.resolve()
    if not config_path.is_file():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 2

    recipe = write_work_config(config_path, work_dir)
    if args.stage == "run":
        run_pipeline(config_path, work_dir)
    else:
        run_stage(args.stage, recipe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
