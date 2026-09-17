"""The only CLI entry point for SaySo model evaluation.

    python -m evals.cli run --suite smoke --adapter endpoint --server http://127.0.0.1:8080
    python -m evals.cli run --suite promotion --adapter endpoint --server http://127.0.0.1:8080
    python -m evals.cli run --category ambiguity --adapter endpoint --server http://127.0.0.1:8080
    python -m evals.cli run --tag grounding --adapter endpoint --server http://127.0.0.1:8080
    python -m evals.cli run --case-id <id> --adapter endpoint --server http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import os
import sys

from evals.adapters.endpoint import EndpointAdapter
from evals.cases import CaseError, load_gates, select_cases
from evals.runner import evaluate, write_run
from evals.scorer import check_gates, summarize


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run the shared evaluation pipeline")
    run.add_argument("--suite", choices=("smoke", "promotion"))
    run.add_argument("--category")
    run.add_argument("--tag")
    run.add_argument("--case-id")
    run.add_argument("--adapter", required=True, choices=("endpoint", "in_memory"))
    run.add_argument("--server", help="OpenAI-compatible base URL (required for --adapter endpoint)")
    run.add_argument("--model", default="sayso")
    run.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "run":
        raise SystemExit(2)
    if not any((args.suite, args.category, args.tag, args.case_id)):
        print("error: specify --suite, --category, --tag, or --case-id", file=sys.stderr)
        return 2
    if args.adapter == "in_memory":
        print("error: --adapter in_memory is the training API, not a CLI backend", file=sys.stderr)
        return 2
    if not args.server:
        print("error: --adapter endpoint requires --server", file=sys.stderr)
        return 2
    try:
        cases = select_cases(suite=args.suite, category=args.category, tag=args.tag, case_id=args.case_id)
    except CaseError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    adapter = EndpointAdapter(
        args.server,
        model=args.model,
        timeout=args.timeout,
        api_key=os.environ.get("LLAMA_API_KEY", ""),
    )
    results = evaluate(cases, adapter)
    dest = write_run(cases=cases, results=results, adapter=adapter, suite=args.suite)
    summary = summarize(results)
    gates = load_gates(args.suite) if args.suite else None
    failures = check_gates(summary, gates, expected_count=len(cases)) if gates else []
    print(f"{summary['passed']}/{summary['total']} passed -> {dest}")
    for failure in failures:
        print(f"  gate: {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
