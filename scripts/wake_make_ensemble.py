#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def _run(path: Path, x):
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {"embeddings": x})[0]


def build_ensemble(members: list[Path], out: Path, prob: bool = False) -> None:
    import onnx
    from onnx import compose, helper

    for member in members:
        if not member.is_file():
            raise SystemExit(f"member not found: {member}")
    base = onnx.load(str(members[0]))
    nodes, inits, vi, outs = [], [], [], []
    for j, member in enumerate(members):
        g = compose.add_prefix(
            onnx.load(str(member)), prefix=f"m{j}_", rename_inputs=False
        ).graph
        last = [n for n in g.node if g.output[0].name in n.output]
        if len(last) != 1 or last[0].op_type != "Sigmoid":
            raise SystemExit(f"member output must end in a Sigmoid node: {member}")
        nodes += [n for n in g.node if not (not prob and n is last[0])]
        inits += list(g.initializer)
        vi += list(g.value_info)
        outs.append(last[0].input[0] if not prob else g.output[0].name)
    if prob:
        nodes.append(helper.make_node("Mean", outs, ["score"]))
    else:
        nodes.append(helper.make_node("Mean", outs, ["mean_logit"]))
        nodes.append(helper.make_node("Sigmoid", ["mean_logit"], ["score"]))
    graph = helper.make_graph(
        nodes,
        out.stem,
        list(base.graph.input),
        list(base.graph.output),
        inits,
        value_info=vi,
    )
    model = helper.make_model(
        graph, opset_imports=base.opset_import, ir_version=base.ir_version
    )
    onnx.checker.check_model(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))


def self_check(members: list[Path], out: Path, prob: bool = False) -> None:
    import numpy as np

    x = np.random.default_rng(0).standard_normal((8, 16, 96)).astype(np.float32)
    if prob:
        expected = np.mean([_run(m, x).astype(np.float64) for m in members], axis=0)
    else:
        logits = [
            np.log(p / (1 - p))
            for p in (_run(m, x).astype(np.float64) for m in members)
        ]
        expected = 1.0 / (1.0 + np.exp(-np.mean(logits, axis=0)))
    got = _run(out, x).astype(np.float64)
    if not np.allclose(got, expected, atol=1e-5):
        raise SystemExit(
            "self-check failed: ensemble output diverges from the member mean"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a wake ensemble ONNX with the same interface as its members "
        "(embeddings [batch,16,96] -> score [batch,1])."
    )
    ap.add_argument("out", type=Path)
    ap.add_argument("members", nargs="+", type=Path, metavar="member")
    ap.add_argument(
        "--prob",
        action="store_true",
        help="mean of member sigmoids (default: mean of pre-sigmoid logits, then Sigmoid)",
    )
    args = ap.parse_args()
    if len(args.members) < 2:
        raise SystemExit("need at least two members")
    build_ensemble(args.members, args.out, prob=args.prob)
    self_check(args.members, args.out, prob=args.prob)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"ok {args.out} {args.out.stat().st_size} bytes sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
