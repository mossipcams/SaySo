#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "satellite"))

from sayso.wake.livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES
from sayso.wake.replay import ReplayConfig

SAY_SO = re.compile(r"\bSAY SO\b")
HARD_TERMS = re.compile(r"\b(SAY SOMETHING|SAY SOME|SAY|SO|SAID|SAYS|SAME|SAVE|SAFE)\b")
CONTEXT_MS = 2500

_provider = None


def classify(transcripts: list[str]) -> tuple[str, list[str]]:
    text = " ".join(t.upper() for t in transcripts)
    if SAY_SO.search(text):
        return "excluded_say_so", ["SAY SO"]
    terms = sorted(set(HARD_TERMS.findall(text)))
    return ("hard_negative" if terms else "false_positive"), terms


def load_chapter(chapter_dir: Path) -> tuple[np.ndarray, list[tuple[str, int, int, str]]]:
    import soundfile as sf

    trans = next(chapter_dir.glob("*.trans.txt"))
    audio, spans, offset = [], [], 0
    for line in trans.read_text(encoding="utf-8").splitlines():
        utt_id, _, text = line.partition(" ")
        pcm, rate = sf.read(chapter_dir / f"{utt_id}.flac", dtype="int16")
        if rate != SAMPLE_RATE or pcm.ndim != 1:
            raise ValueError(f"{utt_id}: expected mono {SAMPLE_RATE} Hz, got {rate} Hz ndim={pcm.ndim}")
        audio.append(pcm)
        spans.append((utt_id, offset, offset + pcm.size, text))
        offset += pcm.size
    return np.concatenate(audio), spans


def _write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.astype("<i2").tobytes())


def _init_worker(model: str, threshold: float, refractory: float) -> None:
    global _provider
    logging.basicConfig(level=logging.WARNING)
    from sayso.wake.livekit import LiveKitWakeWordProvider

    _provider = LiveKitWakeWordProvider(Path(model), "SaySo", threshold=threshold, refractory_seconds=refractory)
    if not _provider.available:
        raise RuntimeError(f"wake model failed to load: {model}")


def mine_chapter(chapter_dir: Path, out: Path) -> dict:
    from sayso.wake.replay import collect_session_activations

    samples, spans = load_chapter(chapter_dir)
    confidences: dict[int, float] = {}

    def predict(window, sample_index):
        detection = _provider.predict_window(window, sample_index=sample_index)
        if detection is not None:
            confidences[sample_index] = detection.confidence
        return detection

    activations, _ms, duration, _ = collect_session_activations(
        None, _provider, predict=predict, pcm=samples.tobytes()
    )
    chapter_id = chapter_dir.parent.name + "-" + chapter_dir.name
    records = []
    for index in activations:
        win_start = index - WINDOW_SAMPLES
        under = [s for s in spans if s[1] < index and s[2] > win_start]
        category, terms = classify([s[3] for s in under])
        pad = CONTEXT_MS * SAMPLE_RATE // 1000
        ctx_start, ctx_end = max(0, win_start - pad), min(samples.size, index + pad)
        record_id = f"{chapter_id}-{index:09d}"
        rec_dir = out / "activations" / category / record_id
        rec_dir.mkdir(parents=True, exist_ok=True)
        _write_wav(rec_dir / "window.wav", samples[win_start:index])
        _write_wav(rec_dir / "context.wav", samples[ctx_start:ctx_end])
        record = {
            "id": record_id,
            "chapter": chapter_id,
            "category": category,
            "hard_terms": terms,
            "confidence": confidences.get(index),
            "sample_index": index,
            "window_start": win_start,
            "context_start": ctx_start,
            "context_end": ctx_end,
            "window_offset_in_context": win_start - ctx_start,
            "utterances": [{"id": s[0], "start": s[1], "end": s[2], "text": s[3]} for s in under],
        }
        (rec_dir / "record.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        records.append(record)
    excluded = sum(s[2] - s[1] for s in spans if SAY_SO.search(s[3].upper())) / SAMPLE_RATE
    summary = {
        "chapter": chapter_id,
        "duration_seconds": duration,
        "excluded_say_so_seconds": excluded,
        "activations": {c: sum(r["category"] == c for r in records)
                        for c in ("false_positive", "hard_negative", "excluded_say_so")},
    }
    (out / "chapters" / f"{chapter_id}.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return summary


def summarize(out: Path, *, model: Path, threshold: float, refractory: float) -> dict:
    chapters = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "chapters").glob("*.json"))]
    hours = sum(c["duration_seconds"] for c in chapters) / 3600.0
    scored_hours = hours - sum(c["excluded_say_so_seconds"] for c in chapters) / 3600.0
    counts = {k: sum(c["activations"][k] for c in chapters) for k in ("false_positive", "hard_negative", "excluded_say_so")}
    counted = counts["false_positive"] + counts["hard_negative"]
    return {
        "corpus": "LibriSpeech train-other-500 (openslr/librispeech_asr other/train.500)",
        "model": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "threshold": threshold,
        "refractory_seconds": refractory,
        "window_samples": WINDOW_SAMPLES,
        "hop_samples": HOP_SAMPLES,
        "chunk_samples": ReplayConfig().chunk_samples,
        "context_ms": CONTEXT_MS,
        "chapters": len(chapters),
        "audio_hours": round(hours, 3),
        "scored_hours": round(scored_hours, 3),
        "activations": counts,
        "false_activations_per_hour": round(counted / scored_hours, 4) if scored_hours else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="LibriSpeech/train-other-500 directory")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.25, help="deployed satellite threshold")
    ap.add_argument("--refractory", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only the first N chapters (smoke run)")
    args = ap.parse_args(argv)

    (args.out / "chapters").mkdir(parents=True, exist_ok=True)
    chapters = sorted(p for p in args.root.glob("*/*") if p.is_dir())
    if args.limit:
        chapters = chapters[: args.limit]
    done = {p.stem for p in (args.out / "chapters").glob("*.json")}
    todo = [c for c in chapters if f"{c.parent.name}-{c.name}" not in done]
    print(f"chapters={len(chapters)} done={len(chapters) - len(todo)} todo={len(todo)}", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(str(args.model), args.threshold, args.refractory)) as pool:
        for i, s in enumerate(pool.map(mine_chapter, todo, [args.out] * len(todo)), 1):
            print(f"[{i}/{len(todo)}] {s['chapter']} {s['duration_seconds']:.0f}s {s['activations']}", flush=True)
    summary = summarize(args.out, model=args.model, threshold=args.threshold, refractory=args.refractory)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
