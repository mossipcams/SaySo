"""Run recorded-audio wake-word evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from satellite.sayso.wake.eval import run_wake_eval, satellite_eval_root


def _detect_hardware() -> str:
    machine = platform.machine().lower()
    if machine.startswith("arm") or machine.startswith("aarch"):
        return "pi"
    return platform.platform()


def _load_wake_defaults() -> tuple[float, float]:
    """Return (threshold, refractory_seconds) from satellite config when available."""
    try:
        from satellite.sayso.config import load_config

        cfg = load_config().wake_word
        return float(cfg.threshold), float(cfg.refractory_seconds)
    except Exception:
        return 0.65, 2.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SaySo satellite wake-word recorded-audio eval")
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to LiveKit-exported ONNX wake model (default: from config if available)",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="Eval corpus root (default: satellite/eval)",
    )
    parser.add_argument(
        "--phrase",
        default="SaySo",
        help="Wake phrase configured on the satellite",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Wake detection threshold (default: config or 0.65)",
    )
    parser.add_argument(
        "--refractory-seconds",
        type=float,
        default=None,
        help="Production cooldown between activations (default: 0 dev, config value in --strict)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing/skipped required cases; use production refractory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_root = args.eval_root or satellite_eval_root()
    config_threshold, config_refractory = _load_wake_defaults()
    threshold = args.threshold if args.threshold is not None else config_threshold
    if args.refractory_seconds is not None:
        refractory_seconds = args.refractory_seconds
    elif args.strict:
        refractory_seconds = config_refractory
    else:
        refractory_seconds = 0.0

    model_path = args.model
    if model_path is None:
        try:
            from satellite.sayso.config import load_config

            model_path = load_config().wake_word.model
        except Exception:
            print("model path required when config is unavailable", file=sys.stderr)
            return 2

    if not model_path.is_file():
        print(f"wake model missing: {model_path}", file=sys.stderr)
        return 2

    report = run_wake_eval(
        model_path=model_path,
        eval_root=eval_root,
        phrase=args.phrase,
        threshold=threshold,
        refractory_seconds=refractory_seconds,
        strict=args.strict,
        hardware=_detect_hardware(),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")

    summary = report.get("summary", {})
    if int(summary.get("failed", 0)) > 0 or int(summary.get("errors", 0)) > 0:
        return 1
    if args.strict and int(summary.get("skipped", 0)) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
