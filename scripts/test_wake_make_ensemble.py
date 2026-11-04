import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.wake_make_ensemble import build_ensemble


def _make_member(path: Path, seed: int, final_op: str = "Sigmoid") -> None:
    rng = np.random.default_rng(seed)
    weight = (rng.standard_normal((16 * 96, 1)) / 40.0).astype(np.float32)
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1])
    nodes = [
        helper.make_node("Flatten", ["embeddings"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "weight"], ["logits"]),
    ]
    if final_op == "Sigmoid":
        nodes.append(helper.make_node("Sigmoid", ["logits"], ["score"]))
    else:
        nodes.append(helper.make_node(final_op, ["logits"], ["score"]))
    graph = helper.make_graph(
        nodes,
        f"member-{seed}",
        [
            helper.make_tensor_value_info(
                "embeddings", TensorProto.FLOAT, ["batch", 16, 96]
            )
        ],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch", 1])],
        [numpy_helper.from_array(weight, "weight")],
        value_info=[logits] if final_op != "Sigmoid" else [],
    )
    model = helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.save(model, str(path))


def _run(path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {"embeddings": x})[0]


def test_logit_ensemble_is_sigmoid_of_mean_logits(tmp_path):
    members = [tmp_path / f"m{i}.onnx" for i in range(3)]
    for i, m in enumerate(members):
        _make_member(m, seed=i)
    out = tmp_path / "ensemble.onnx"
    build_ensemble(members, out)

    x = np.random.default_rng(7).standard_normal((5, 16, 96)).astype(np.float32)
    logits = [
        np.log(p / (1 - p)) for p in (_run(m, x).astype(np.float64) for m in members)
    ]
    expected = 1.0 / (1.0 + np.exp(-np.mean(logits, axis=0)))
    assert np.allclose(_run(out, x), expected, atol=1e-5)


def test_prob_ensemble_is_mean_of_sigmoids(tmp_path):
    members = [tmp_path / f"m{i}.onnx" for i in range(2)]
    for i, m in enumerate(members):
        _make_member(m, seed=i)
    out = tmp_path / "ensemble_prob.onnx"
    build_ensemble(members, out, prob=True)

    x = np.random.default_rng(7).standard_normal((5, 16, 96)).astype(np.float32)
    expected = np.mean([_run(m, x).astype(np.float64) for m in members], axis=0)
    assert np.allclose(_run(out, x), expected, atol=1e-5)


def test_member_without_sigmoid_output_is_rejected(tmp_path):
    good = tmp_path / "good.onnx"
    bad = tmp_path / "bad.onnx"
    _make_member(good, seed=0)
    _make_member(bad, seed=1, final_op="ReLU")
    with pytest.raises(SystemExit):
        build_ensemble([good, bad], tmp_path / "ensemble.onnx")
