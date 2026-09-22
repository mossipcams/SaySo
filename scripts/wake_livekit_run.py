"""Thin wrapper around livekit.wakeword CLI stages for SaySo training.

Stages: setup, generate, augment, train, export, eval, run.

``run`` always executes the full LiveKit pipeline in order and never skips
generate. An optional verified hard-negative overlay appends model-mined ACAV
rows to ``negative_features_train.npy`` after augment and before train.

``data_dir`` and ``output_dir`` are rewritten under ``--work-dir`` so shipped
living2 artifacts stay untouched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import wake_hard_negative_mine as mine  # noqa: E402
_DEFAULT_CONFIG = _REPO_ROOT / "satellite" / "models" / "sayso.yaml"

STAGES = ("setup", "generate", "augment", "train", "export", "eval")
OVERLAY_STAGE = "overlay"
RUN_STAGES = STAGES

OVERLAY_MANIFEST_NAME = "hard_negative_overlay_manifest.json"
DEFAULT_OVERLAY_CHUNK_ROWS = 256
ACAV_ROW_SHAPE = mine.ACAV_ROW_SHAPE


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


def pipeline_stages(*, with_overlay: bool) -> tuple[str, ...]:
    if not with_overlay:
        return STAGES
    return ("setup", "generate", "augment", OVERLAY_STAGE, "train", "export", "eval")


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


def verify_hard_negative_provenance(
    provenance_path: Path,
    features_path: Path,
) -> dict[str, Any]:
    """Validate miner provenance against the overlay feature file on disk."""
    if not provenance_path.is_file():
        raise FileNotFoundError(f"provenance not found: {provenance_path}")
    if not features_path.is_file():
        raise FileNotFoundError(f"hard-negative features not found: {features_path}")

    try:
        doc = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid provenance JSON: {provenance_path}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"provenance must be a JSON object: {provenance_path}")

    output = doc.get("output")
    if not isinstance(output, dict):
        raise ValueError("provenance missing output object")
    selection = doc.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("provenance missing selection object")
    source = doc.get("source")
    if not isinstance(source, dict):
        raise ValueError("provenance missing source object")

    source_shape = source.get("shape")
    if not isinstance(source_shape, list) or len(source_shape) != 3:
        raise ValueError("provenance source.shape must be [N, 16, 96]")
    if tuple(source_shape[1:]) != ACAV_ROW_SHAPE:
        raise ValueError("provenance source.shape must be (N, 16, 96)")

    features = np.load(features_path, mmap_mode="r")
    mine.validate_acav_shape(features)

    expected_shape = output.get("shape")
    if not isinstance(expected_shape, list) or list(features.shape) != expected_shape:
        raise ValueError(
            f"overlay features shape {tuple(features.shape)} does not match provenance "
            f"output.shape {expected_shape}"
        )

    expected_dtype = output.get("dtype")
    if expected_dtype is not None and str(features.dtype) != str(expected_dtype):
        raise ValueError(
            f"overlay features dtype {features.dtype} does not match provenance "
            f"output.dtype {expected_dtype}"
        )

    selected_count = selection.get("selected_count")
    if not isinstance(selected_count, int) or selected_count != int(features.shape[0]):
        raise ValueError(
            f"provenance selected_count {selected_count} does not match overlay rows "
            f"{features.shape[0]}"
        )

    expected_sha = output.get("sha256")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise ValueError("provenance output.sha256 is required")
    actual_sha = mine._sha256_file(features_path)
    if actual_sha != expected_sha:
        raise ValueError("overlay features sha256 does not match provenance output.sha256")

    return doc


def append_overlay_to_train_negatives(
    train_negatives_path: Path,
    overlay_features_path: Path,
    *,
    chunk_rows: int = DEFAULT_OVERLAY_CHUNK_ROWS,
) -> tuple[int, int]:
    """Append overlay rows to train negatives via chunked copy and atomic replace."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if not train_negatives_path.is_file():
        raise FileNotFoundError(f"train negatives not found: {train_negatives_path}")

    existing = np.load(train_negatives_path, mmap_mode="r")
    mine.validate_acav_shape(existing)
    overlay = np.load(overlay_features_path, mmap_mode="r")
    mine.validate_acav_shape(overlay)

    before_rows = int(existing.shape[0])
    overlay_rows = int(overlay.shape[0])
    after_rows = before_rows + overlay_rows
    dtype = existing.dtype
    if overlay.dtype != dtype:
        raise ValueError(
            f"overlay dtype {overlay.dtype} does not match train negatives dtype {dtype}"
        )

    tmp_path = train_negatives_path.with_name(f"{train_negatives_path.name}.overlay.tmp")
    new_array = np.lib.format.open_memmap(
        str(tmp_path),
        mode="w+",
        dtype=dtype,
        shape=(after_rows, *ACAV_ROW_SHAPE),
    )
    replaced = False
    try:
        write_offset = 0
        for start in range(0, before_rows, chunk_rows):
            end = min(start + chunk_rows, before_rows)
            new_array[write_offset : write_offset + (end - start)] = np.asarray(
                existing[start:end],
                dtype=dtype,
            )
            write_offset += end - start
        for start in range(0, overlay_rows, chunk_rows):
            end = min(start + chunk_rows, overlay_rows)
            new_array[write_offset : write_offset + (end - start)] = np.asarray(
                overlay[start:end],
                dtype=dtype,
            )
            write_offset += end - start
        new_array.flush()
        del new_array
        tmp_path.replace(train_negatives_path)
        replaced = True
    except Exception:
        try:
            del new_array
        except NameError:
            pass
        raise
    finally:
        if not replaced and tmp_path.exists():
            tmp_path.unlink()

    return before_rows, overlay_rows


def apply_hard_negative_overlay(
    recipe_path: Path,
    work_dir: Path,
    features_path: Path,
    provenance_path: Path,
    *,
    chunk_rows: int = DEFAULT_OVERLAY_CHUNK_ROWS,
) -> Path:
    """Verify overlay provenance and append only to negative_features_train.npy."""
    provenance = verify_hard_negative_provenance(provenance_path, features_path)
    cfg = _load_yaml(recipe_path)
    model_name = cfg.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("recipe model_name is required")
    output_dir = Path(str(cfg["output_dir"]))
    model_dir = output_dir / model_name
    train_path = model_dir / "negative_features_train.npy"

    before_hash = mine._sha256_file(train_path)
    before_rows, overlay_rows = append_overlay_to_train_negatives(
        train_path,
        features_path,
        chunk_rows=chunk_rows,
    )
    after_hash = mine._sha256_file(train_path)
    after_rows = before_rows + overlay_rows

    manifest: dict[str, Any] = {
        "overlay_features_path": str(features_path.resolve()),
        "provenance_path": str(provenance_path.resolve()),
        "train_negatives_path": str(train_path.resolve()),
        "provenance_version": provenance.get("version"),
        "counts": {
            "before": before_rows,
            "overlay": overlay_rows,
            "after": after_rows,
        },
        "sha256": {
            "before": before_hash,
            "overlay": provenance["output"]["sha256"],
            "after": after_hash,
        },
    }
    manifest_path = work_dir / OVERLAY_MANIFEST_NAME
    mine.atomic_write_json(manifest_path, manifest)
    return manifest_path


def run_pipeline(
    config_path: Path,
    work_dir: Path,
    *,
    runner: Any | None = None,
    hard_negative_features: Path | None = None,
    hard_negative_provenance: Path | None = None,
    overlay_chunk_rows: int = DEFAULT_OVERLAY_CHUNK_ROWS,
) -> list[str]:
    """Execute setup → generate → augment → [overlay] → train → export → eval."""
    if (hard_negative_features is None) ^ (hard_negative_provenance is None):
        raise ValueError(
            "hard-negative overlay requires both --hard-negative-features and "
            "--hard-negative-provenance"
        )
    with_overlay = hard_negative_features is not None
    recipe = write_work_config(config_path, work_dir)
    executed: list[str] = []
    for stage in pipeline_stages(with_overlay=with_overlay):
        if stage == OVERLAY_STAGE:
            apply_hard_negative_overlay(
                recipe,
                work_dir,
                hard_negative_features.resolve(),
                hard_negative_provenance.resolve(),
                chunk_rows=overlay_chunk_rows,
            )
        else:
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
    parser.add_argument(
        "--hard-negative-features",
        type=Path,
        default=None,
        help="verified (K, 16, 96) hard-negative features from wake_hard_negative_mine.py",
    )
    parser.add_argument(
        "--hard-negative-provenance",
        type=Path,
        default=None,
        help="provenance JSON paired with --hard-negative-features",
    )
    parser.add_argument(
        "--overlay-chunk-rows",
        type=int,
        default=DEFAULT_OVERLAY_CHUNK_ROWS,
        help="row chunk size for bounded-memory train-negative append",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = args.config.resolve()
    work_dir = args.work_dir.resolve()
    if not config_path.is_file():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 2

    if (args.hard_negative_features is None) ^ (args.hard_negative_provenance is None):
        print(
            "hard-negative overlay requires both --hard-negative-features and "
            "--hard-negative-provenance",
            file=sys.stderr,
        )
        return 2
    if args.overlay_chunk_rows <= 0:
        print("overlay-chunk-rows must be positive", file=sys.stderr)
        return 2

    recipe = write_work_config(config_path, work_dir)
    if args.stage == "run":
        run_pipeline(
            config_path,
            work_dir,
            hard_negative_features=args.hard_negative_features,
            hard_negative_provenance=args.hard_negative_provenance,
            overlay_chunk_rows=args.overlay_chunk_rows,
        )
    else:
        if args.hard_negative_features is not None:
            print(
                "hard-negative overlay is only applied during the run stage",
                file=sys.stderr,
            )
            return 2
        run_stage(args.stage, recipe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
