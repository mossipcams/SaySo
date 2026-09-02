"""CLI entry point for SaySo evaluation benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evals.comparison_report import (
    ComparisonReportError,
    build_comparison_report,
    default_comparison_report_path,
    write_comparison_report,
)
from evals.config import BenchmarkConfig
from evals.corpus import (
    load_comparison_corpus,
    load_core_corpus,
    load_followup_corpus,
    load_language_noise_corpus,
    load_safety_corpus,
)
from evals.mlx_executor import controller_mlx_executor, resolve_eval_executor
from evals.gate import expansion_allowed
from evals.report import (
    build_gate_inputs_from_benchmark_output,
    build_report_from_benchmark_output,
    default_report_path,
    write_eval_report,
)
from evals.runner import run_benchmark
from evals.schema import EvalCase

CORPUS_CHOICES = ("core", "safety", "language_noise", "followup", "comparison", "all")


def load_corpus_cases(corpus: str) -> list[EvalCase]:
    if corpus == "core":
        return load_core_corpus()
    if corpus == "safety":
        return load_safety_corpus()
    if corpus == "language_noise":
        return load_language_noise_corpus()
    if corpus == "followup":
        return load_followup_corpus()
    if corpus == "comparison":
        return load_comparison_corpus()
    cases: list[EvalCase] = []
    cases.extend(load_core_corpus())
    cases.extend(load_safety_corpus())
    cases.extend(load_language_noise_corpus())
    cases.extend(load_followup_corpus())
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Run SaySo evaluation benchmarks",
    )
    parser.add_argument(
        "--corpus",
        choices=CORPUS_CHOICES,
        help="Eval corpus to run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL output path for benchmark records",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow live Home Assistant actuation when entity allowlist matches",
    )
    parser.add_argument(
        "--allowlist",
        default="",
        help="Comma-separated entity IDs permitted for live actuation with --execute",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write evals/reports/<corpus>.report.json after the benchmark run",
    )
    parser.add_argument(
        "--check-gate",
        action="store_true",
        help="Check expansion gate on existing benchmark output; exit 1 when blocked",
    )
    parser.add_argument(
        "--comparison-report",
        action="store_true",
        help="Write evals/reports/comparison.report.json from two JSONL benchmark runs",
    )
    parser.add_argument(
        "--compare-sayso",
        type=Path,
        help="SaySo 230M JSONL output for --comparison-report",
    )
    parser.add_argument(
        "--compare-baseline",
        type=Path,
        help="Home-LLM 270M JSONL output for --comparison-report",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup executor runs before scoring (default: 0)",
    )
    return parser


def parse_allowlist(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def check_expansion_gate(cases: list[EvalCase], output_path: Path) -> int:
    score, ledger_summary, latency = build_gate_inputs_from_benchmark_output(
        cases,
        output_path,
    )
    allowed, reasons = expansion_allowed(score, ledger_summary, latency)
    if allowed:
        print("expansion_gate=allowed")
        return 0
    print("expansion_gate=blocked")
    for reason in reasons:
        print(f"  - {reason}")
    return 1


def build_comparison_report_from_paths(
    sayso_output: Path,
    baseline_output: Path,
) -> dict[str, object]:
    cases = load_comparison_corpus()
    return build_comparison_report(sayso_output, baseline_output, cases=cases)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.comparison_report:
        if args.compare_sayso is None or args.compare_baseline is None:
            print(
                "error: --comparison-report requires --compare-sayso and --compare-baseline",
                file=sys.stderr,
            )
            return 2
        try:
            report = build_comparison_report_from_paths(args.compare_sayso, args.compare_baseline)
        except ComparisonReportError as exc:
            print(f"comparison_report=blocked: {exc}", file=sys.stderr)
            return 1
        report_path = write_comparison_report(default_comparison_report_path(), report)
        print(f"comparison_report={report_path}")
        return 0

    if args.corpus is None or args.output is None:
        print("error: benchmark run requires --corpus and --output", file=sys.stderr)
        return 2

    cases = load_corpus_cases(args.corpus)

    if args.check_gate:
        return check_expansion_gate(cases, args.output)

    allowlist = parse_allowlist(args.allowlist)

    executor = resolve_eval_executor()
    runtime = "mlx" if executor is controller_mlx_executor else "fake"
    config = BenchmarkConfig(runtime=runtime, warmup_count=args.warmup)

    result = run_benchmark(
        cases,
        args.output,
        executor=executor,
        config=config,
        seed=config.seed,
        warmup_count=config.warmup_count,
        execute=args.execute,
        entity_allowlist=allowlist,
    )

    print(
        f"scored={result.scored} skipped={result.skipped} "
        f"warmup_runs={result.warmup_runs} errors={result.errors}",
    )

    if args.report:
        report = build_report_from_benchmark_output(cases, args.output)
        report_path = write_eval_report(default_report_path(args.corpus), report)
        print(f"report={report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
