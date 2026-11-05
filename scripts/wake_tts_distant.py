#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

KODA = ("K", "OW", "D", "AH")

EXAMPLES = """\
example (VM layout; ASR word lists of the train TV hours, never the holdout hours):
  CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \\
  PYTHONPATH=/srv/llm/data/wake/runs/koda-phonics-v6-20261018/src \\
    /srv/llm/wake/runs/koda-phonics-v3-20261008/.venv/bin/python scripts/wake_tts_distant.py \\
    /srv/llm/data/wake/runs/near-replay-20261101/tvwords \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/data/piper/en-us-libritts-high.pt \\
    /srv/llm/data/wake/runs/koda-phonics-v6-20261018/staging/distant/raw \\
    --exclude tv-20261008T204900Z --exclude tv-20261009T134733Z
"""


def edit(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def near_koda(phones: list[str], max_edit: int = 2) -> bool:
    """True if any 3-5 phone span is within max_edit edits of K OW D AH."""
    return any(
        edit(tuple(phones[i : i + n]), KODA) <= max_edit
        for n in (3, 4, 5)
        for i in range(max(1, len(phones) - n + 1))
    )


def chunk_phones(words: list[str]) -> list[str] | None:
    import pronouncing

    out: list[str] = []
    for w in words:
        p = pronouncing.phones_for_word(w)
        if not p:
            return None
        out += [re.sub(r"\d", "", x) for x in p[0].split()]
    return out


def phrases(words_dir: Path, exclude: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    for p in sorted(glob.glob(str(words_dir / "*.words.json"))):
        if any(e in Path(p).name for e in exclude):
            continue
        ws = [re.sub(r"[^a-z']", "", x["w"].lower()) for x in json.load(open(p))]
        ws = [w for w in ws if w]
        for i in range(len(ws)):
            n = rng.randint(1, 3)
            c = ws[i : i + n]
            ph = chunk_phones(c)
            if ph and len(ph) >= 3 and not near_koda(ph):
                seen.add(" ".join(c))
    return sorted(seen)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Piper TTS of 1-3 word TV-transcript chunks that are phonetically far from Koda.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("words_dir", type=Path, help="dir of <hour>.words.json ASR word lists")
    ap.add_argument("vits", type=Path, help="Piper VITS checkpoint (.pt)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--exclude", action="append", default=[], help="hour name to skip (holdout)")
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=37)
    args = ap.parse_args()
    ph = phrases(args.words_dir, args.exclude, args.seed)
    print("phrases", len(ph), ph[:10])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "phrases.json").write_text(json.dumps(ph))
    from livekit.wakeword.data.generate import synthesize_clips

    have = len(glob.glob(str(args.out / "clip_[0-9][0-9][0-9][0-9][0-9][0-9].wav")))
    synthesize_clips(ph, args.out, args.n - have, vits_model_path=args.vits, start_index=have)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
