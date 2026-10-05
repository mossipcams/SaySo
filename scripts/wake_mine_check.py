#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import wave
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "satellite"))
from sayso.wake.mining import ingest_record

SAMPLE_RATE = 16000
CLASSES = ("wake", "false_positive", "missed_wake", "near_miss")


def _norm(word: str) -> str:
    return re.sub(r"[^a-z']", "", word.lower())


def phrase_spans(words: list[dict], phrase: str) -> list[tuple[float, float]]:
    target = [_norm(w) for w in phrase.split()]
    tokens = [_norm(w["word"]) for w in words]
    n = len(target)
    return [(words[i]["start"], words[i + n - 1]["end"])
            for i in range(len(tokens) - n + 1) if tokens[i:i + n] == target]


def classify(fired: bool, said: bool) -> str:
    if fired:
        return "wake" if said else "false_positive"
    return "missed_wake" if said else "near_miss"


def _read(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.zeros(0, dtype=np.float32)
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16 kHz mono")
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def _stt_transcript(spool: Path, capture_id: str) -> str | None:
    path = spool / "outcomes" / f"{capture_id}.stt.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("transcript")


def check_spool(spool: Path, phrase: str, model_name: str) -> list[dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def words_of(audio: np.ndarray) -> list[dict]:
        segments, _ = model.transcribe(audio, language="en", beam_size=5, temperature=0.0,
                                       word_timestamps=True, condition_on_previous_text=False)
        return [{"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
                for seg in segments for w in (seg.words or [])]

    rows = []
    for record_dir in sorted(p for p in (spool / "records").iterdir() if p.is_dir()):
        ok, message = ingest_record(record_dir)
        if not ok:
            print(f"skip {record_dir.name}: {message}", file=sys.stderr)
            continue
        meta = json.loads((record_dir / "record.json").read_text(encoding="utf-8"))
        pre, window = _read(record_dir / "pre.wav"), _read(record_dir / "window.wav")
        window_words = words_of(window)
        words = words_of(np.concatenate([pre, window, _read(record_dir / "post.wav")]))
        check = {
            "phrase": phrase,
            "class": classify(bool(meta.get("fired")), bool(phrase_spans(window_words, phrase))),
            "transcript": " ".join(w["word"] for w in window_words),
            "context_transcript": " ".join(w["word"] for w in words),
            "words": window_words,
            "stt_transcript": _stt_transcript(spool, record_dir.name),
            "asr_model": f"faster-whisper {model_name}",
        }
        (record_dir / "check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
        rows.append({**meta, "check": check, "_dir": record_dir})
    return rows


def report(rows: list[dict]) -> None:
    counts = Counter(r["check"]["class"] for r in rows)
    stamps = sorted(datetime.strptime(r["published_utc"], "%Y%m%dT%H%M%SZ") for r in rows)
    span_h = (stamps[-1] - stamps[0]).total_seconds() / 3600 if stamps else 0.0
    print(f"{len(rows)} records over {span_h:.2f} h of spool time "
          "(not uptime; caps/acks/drops apply)")
    for name in CLASSES:
        print(f"  {name:<15} {counts.get(name, 0)}")
    fired = counts["wake"] + counts["false_positive"]
    if fired:
        print(f"  false positives among fires: {counts['false_positive']}/{fired}")
    for name in ("false_positive", "missed_wake"):
        picked = sorted((r for r in rows if r["check"]["class"] == name), key=lambda r: -r["score"])[:15]
        if picked:
            print(f"\n{name} (highest score first):")
            for r in picked:
                print(f"  {r['score']:.3f}  {r['capture_id']}  {r['check']['transcript'] or '<no speech>'}")


def export(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    manifest = []
    for r in rows:
        name = r["check"]["class"]
        (out / name).mkdir(exist_ok=True)
        dest = out / name / f"{r['capture_id']}.wav"
        shutil.copy2(r["_dir"] / "window.wav", dest)
        manifest.append({
            "file": f"{name}/{dest.name}",
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "class": name,
            "needs_verification": name in ("wake", "missed_wake"),
            **{k: r.get(k) for k in ("capture_id", "published_utc", "score", "detect_threshold",
                                     "model_sha256", "sampling_reason")},
            **{k: r["check"][k] for k in ("transcript", "context_transcript", "stt_transcript", "asr_model")},
        })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nexported {len(manifest)} windows to {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spool", type=Path, help="pulled mining spool (records/, outcomes/)")
    ap.add_argument("--phrase", default="Koda")
    ap.add_argument("--asr-model", default="small.en")
    ap.add_argument("--export", type=Path, help="new dir for per-class window wavs + manifest.json")
    args = ap.parse_args()
    rows = check_spool(args.spool, args.phrase, args.asr_model)
    if not rows:
        print("no verified records", file=sys.stderr)
        return 1
    report(rows)
    if args.export:
        export(rows, args.export)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
