#!/usr/bin/env python3
"""Export canonical SaySo JSONL into LlamaFactory alpaca instruction/output views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.lfm import (
    dataset_info_fragment,
    default_chat_template_model,
    expand_canonical_row_to_views,
)


def export_canonical_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    model_name: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    view_count = 0
    with input_path.open(encoding="utf-8") as handle, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        for source_row_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            views = expand_canonical_row_to_views(
                row,
                model_name=model_name,
                source_row_index=source_row_index,
            )
            for view in views:
                out.write(json.dumps(view, ensure_ascii=False) + "\n")
                view_count += 1
    return view_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Canonical OpenAI JSONL")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Rendered alpaca JSONL (instruction/output)",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="LlamaFactory dataset_info.json key",
    )
    parser.add_argument(
        "--dataset-info-out",
        type=Path,
        help="Write a single-entry dataset_info.json fragment",
    )
    parser.add_argument(
        "--model",
        default=default_chat_template_model(),
        help="Tokenizer/chat-template source (default: local artifact or HF Base id)",
    )
    args = parser.parse_args()

    views = export_canonical_jsonl(args.input, args.output, model_name=args.model)
    print(f"Wrote {views} rendered views -> {args.output}")

    if args.dataset_info_out is not None:
        fragment = dataset_info_fragment(args.dataset_name, args.output.name)
        args.dataset_info_out.parent.mkdir(parents=True, exist_ok=True)
        args.dataset_info_out.write_text(
            json.dumps(fragment, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote dataset_info fragment -> {args.dataset_info_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
