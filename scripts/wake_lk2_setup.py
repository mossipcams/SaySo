#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from pathlib import Path

import yaml

HARD_YAML = Path("/srv/llm/data/wake/runs/koda-phonics-v6-20261018/koda-phonics-v10t2.yaml")


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("LIVEKIT_URL", "local-piper")
    os.environ.setdefault("LIVEKIT_API_KEY", "local-piper")
    os.environ.setdefault("LIVEKIT_API_SECRET", "local-piper")
    importlib.import_module("wake_lk2_piper")
    from livekit.wakeword_train import pipeline
    from livekit.wakeword_train.config import WakeWordConfig

    logging.basicConfig(level=logging.INFO)
    with open(args.hard_yaml) as fh:
        hard = yaml.safe_load(fh)["custom_negative_phrases"]
    pipeline.phoneme_steered = lambda config: False
    _gen = pipeline._generate
    pipeline._generate = lambda config, d, splits, overwrite=False, **k: _gen(
        config, d, splits, overwrite, hard_negatives=json.dumps(hard), tts__concurrency=args.tts_concurrency, **k
    )
    cfg = WakeWordConfig(
        wakeword=args.wakeword,
        data_dir=args.out,
        n_positive=args.n_positive,
        n_confusable=args.n_confusable,
        n_sentence=args.n_sentence,
    )
    assert pipeline._wanted(cfg).keys() == {"natural"}
    print(pipeline.run_setup(cfg, eval=args.eval))


def main() -> int:
    ap = argparse.ArgumentParser(description="livekit-wakeword v2 setup for Koda on local Piper (no LiveKit Cloud).")
    ap.add_argument("out", help="data_dir for generated sets")
    ap.add_argument("n_positive", type=int)
    ap.add_argument("n_confusable", type=int)
    ap.add_argument("n_sentence", type=int)
    ap.add_argument("--eval", action="store_true", help="run v2's eval step after setup")
    ap.add_argument("--wakeword", default="Koda")
    ap.add_argument("--hard-yaml", type=Path, default=HARD_YAML, help="recipe YAML whose custom_negative_phrases are the hard negatives")
    ap.add_argument("--tts-concurrency", type=int, default=4)
    run(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
