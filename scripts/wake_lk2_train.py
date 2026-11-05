#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging


def run(args: argparse.Namespace) -> None:
    from livekit.wakeword_train import pipeline
    from livekit.wakeword_train.config import WakeWordConfig

    logging.basicConfig(level=logging.INFO)
    pipeline.phoneme_steered = lambda config: False
    cfg = WakeWordConfig(
        wakeword=args.wakeword,
        model_name=args.model_name,
        data_dir=args.data,
        output_dir=args.out,
        n_positive=args.n_positive,
        n_confusable=args.n_confusable,
        n_sentence=args.n_sentence,
        steps=args.steps,
    )
    assert pipeline.train_datasets(cfg), f"no generated training sets under {args.data}: run wake_lk2_setup.py first"
    print(pipeline.run_train(cfg, "standard"))


def main() -> int:
    ap = argparse.ArgumentParser(description="livekit-wakeword v2 training for Koda (run after wake_lk2_setup.py).")
    ap.add_argument("data", help="data_dir written by wake_lk2_setup.py")
    ap.add_argument("out", help="output_dir for the trained model")
    ap.add_argument("n_positive", type=int)
    ap.add_argument("n_confusable", type=int)
    ap.add_argument("n_sentence", type=int)
    ap.add_argument("steps", type=int)
    ap.add_argument("model_name", nargs="?", default="koda_v2")
    ap.add_argument("--wakeword", default="Koda")
    run(ap.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
