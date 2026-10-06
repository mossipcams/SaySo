"""Export the DMA-KWS Stage II (QbyT) matcher to a fixed-shape ONNX file for the satellite.

Source: https://github.com/aizhiqi-work/DMA-KWS (commit below), checkpoint
ckpts/stage2/155k-v2-ft.ckpt. Dev-machine tool: needs torch, onnx and a clone of that repo.
The repo has no LICENSE file, so neither the checkpoint nor the exported model is
committed here; copy the .onnx to the satellite (see satellite/models/README.md).

  git clone https://github.com/aizhiqi-work/DMA-KWS && git -C DMA-KWS checkout 207d0561dee3c024ec7a8d4e749b01fe95a670ea
  python scripts/export_dma_kws_onnx.py --repo DMA-KWS --out dma-kws-stage2.onnx

Shapes are baked in: a 78-frame (0.8 s) feature crop and a 4-phoneme keyword. Variable
keyword length does not export (attention reshapes are traced as constants).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import types
from pathlib import Path

CKPT_SHA256 = "1135cd36a479c0db449499ba4c076b2a81c2dcdc99203e91db790696c85d2c48"  # 155k-v2-ft.ckpt
FRAMES = 78
ANCHOR_LEN = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ckpt = args.repo / "ckpts/stage2/155k-v2-ft.ckpt"
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if digest != CKPT_SHA256:
        raise SystemExit(f"checkpoint SHA-256 mismatch: {digest}")

    # wenet utils import openai-whisper only for a language table; stub it.
    whisper = types.ModuleType("whisper")
    tokenizer = types.ModuleType("whisper.tokenizer")
    tokenizer.LANGUAGES = {}
    whisper.tokenizer = tokenizer
    sys.modules.update({"whisper": whisper, "whisper.tokenizer": tokenizer})
    sys.path.insert(0, str(args.repo / "decode"))

    import torch
    from stage2.model import QbyT
    from stage2.models.encoder import ConformerEncoder

    enc = ConformerEncoder(
        input_size=80, output_size=144, attention_heads=4, linear_units=576, num_blocks=6,
        dropout_rate=0.1, positional_dropout_rate=0.1, attention_dropout_rate=0.0,
        use_cnn_module=True, input_layer="conv2d", pos_enc_layer_type="rel_pos",
        selfattention_layer_type="rel_selfattn", cnn_module_kernel=3,
    )
    qbyt = QbyT(encoder_output_size=144, num_embeds=73, embed_dim=128, post_num_layers=2)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)  # plain tensors only
    state = state.get("state_dict", state)
    enc.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("encoder.")})
    qbyt.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith("qbyt.")})

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc, self.qbyt = enc.eval(), qbyt.eval()

        def forward(self, feats, anchor):
            lengths = torch.full((feats.shape[0],), FRAMES, dtype=torch.long)
            out, _ = self.enc(feats, lengths)
            logits, _ = self.qbyt(out, anchor)
            return torch.sigmoid(logits)

    net = Net().eval()
    feats = torch.randn(1, FRAMES, 80)
    anchor = torch.tensor([[44, 50, 23, 9]])  # K OW1 D AH0
    torch.onnx.export(net, (feats, anchor), str(args.out), input_names=["feats", "anchor"],
                      output_names=["score"], opset_version=17, dynamo=False)
    print(f"wrote {args.out} sha256={hashlib.sha256(args.out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
