#!/usr/bin/env python3
"""Score a checkpoint on a v3 quality eval set by parsing raw llama.cpp completions.

Mirrors the host-only `rawparse` scorer (`/srv/training-runs/eval_gold_raw.py`),
but version-controlled, so a scorer change is attributable to a commit. It is a
*different file* from that scorer: calibrate before trusting a new number.

    # calibrate — must reproduce the recorded rawparse result for this checkpoint
    python3 training/scripts/eval_v3_rawparse.py \
      --eval-set training/datasets/sayso_quality_eval_v3_gold.jsonl \
      --tokenizer /srv/models/SaySo-LFM2.5-230M-v3-40k-ep1-merged \
      --out /srv/training-runs/eval_v3_gold_ep1_repoparse.json    # expect 16/30

    # then the baseline
    python3 training/scripts/eval_v3_rawparse.py \
      --eval-set training/datasets/sayso_quality_eval_v3_gold.jsonl \
      --tokenizer /srv/models/LFM2.5-230M-Base \
      --out /srv/training-runs/eval_v3_gold_base.json

Requires a llama.cpp server hosting the checkpoint (`--server`, default
http://127.0.0.1:8080) and `transformers` for the chat template.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evals.harness import load_eval_jsonl  # noqa: E402
from evals.lfm_python_parse import LfmPythonParseError, parse_lfm_python_tool_calls  # noqa: E402
from evals.v3_quality import score_quality_gold  # noqa: E402

STOP = ["<|im_end|>", "<|im_start|>"]


def prompt_for(example: dict[str, Any], tokenizer: Any) -> str:
    """Render everything up to the assistant turn the model must produce."""
    messages = []
    for message in example["messages"]:
        if message.get("role") == "assistant":
            break
        messages.append({key: value for key, value in message.items() if key != "train_on_turn"})
    return tokenizer.apply_chat_template(
        messages,
        tools=example.get("tools"),
        tokenize=False,
        add_generation_prompt=True,
    )


def complete(server: str, prompt: str, n_predict: int, timeout: float, api_key: str = "") -> str:
    payload = json.dumps(
        {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "stop": STOP}
    ).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{server.rstrip('/')}/completion",
        data=payload,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return str(json.loads(response.read()).get("content", ""))


def to_tool_calls(text: str) -> tuple[list[dict[str, Any]], str | None]:
    """Parse Python-style calls out of raw text. No calls is a valid prediction."""
    stripped = text.strip()
    if not stripped:
        return [], None
    try:
        parsed = parse_lfm_python_tool_calls(stripped)
    except LfmPythonParseError as exc:
        # Prose instead of a call is a real prediction (no_action rows want it),
        # so an unparseable completion scores as "made no call", not as an error.
        return [], str(exc)
    return [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
        }
        for index, call in enumerate(parsed)
    ], None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="model dir holding the chat template")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    import os

    api_key = os.environ.get("LLAMA_API_KEY", "")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    examples = load_eval_jsonl(args.eval_set)

    results: list[dict[str, Any]] = []
    passed = 0
    for example in examples:
        prompt = prompt_for(example, tokenizer)
        try:
            text = complete(args.server, prompt, args.n_predict, args.timeout, api_key)
        except (urllib.error.URLError, TimeoutError) as exc:
            results.append(
                {
                    "candidate_id": (example.get("metadata") or {}).get("candidate_id"),
                    "error": str(exc),
                    "pass": False,
                }
            )
            continue
        calls, parse_error = to_tool_calls(text)
        scored = score_quality_gold(example, [{"role": "assistant", "content": text, "tool_calls": calls}])
        passed += bool(scored["pass"])
        results.append(
            {
                "candidate_id": (example.get("metadata") or {}).get("candidate_id"),
                "category": (example.get("metadata") or {}).get("category"),
                "completion": text,
                "parsed_calls": calls,
                "parse_error": parse_error,
                **scored,
            }
        )

    payload = {
        "scorer": "repoparse",
        "scorer_commit_file": str(Path(__file__).relative_to(ROOT.parent)),
        "eval_set": str(args.eval_set),
        "eval_set_sha256_16": _sha16(args.eval_set),
        "tokenizer": args.tokenizer,
        "passed": passed,
        "total": len(examples),
        "by_failure_category": _tally(results),
        "examples": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{passed}/{len(examples)}  {args.eval_set.name}  ->  {args.out}")
    return 0


def _sha16(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _tally(results: list[dict[str, Any]]) -> dict[str, int]:
    tally: dict[str, int] = {}
    for item in results:
        if item.get("pass"):
            continue
        key = str(item.get("failure_category") or item.get("error") or "unknown")
        tally[key] = tally.get(key, 0) + 1
    return dict(sorted(tally.items(), key=lambda pair: -pair[1]))


if __name__ == "__main__":
    raise SystemExit(main())
