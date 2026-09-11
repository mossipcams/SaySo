#!/usr/bin/env python3
"""Fixed 20-command STT benchmark: current pipeline vs native-rate pipeline.

Runs the *same* Faster Whisper model over the *same* recordings through two
front ends and reports the difference, so a change in transcription quality can
only be attributed to the audio path, not to the model.

  current    the previous behaviour: hand the device the 16 kHz request and let
             the audio server resample implicitly, with no anti-alias control.

  corrected  capture at the device's native rate, apply fixed gain, and resample
             once to 16 kHz with a continuous polyphase filter.

The runner also reports speech RMS, peak, and clipping per phrase and emits a
single stable gain recommendation. It deliberately does not sweep noise
suppression: the working hypothesis is gain, not NS, and an NS sweep would
confound the comparison.

Recordings are consumed as WAV files named ``<command id>.wav`` under
``--audio-dir`` at the native rate. Missing recordings are skipped, never
fabricated.

Usage:

    python3 satellite/benchmarks/run.py \
        --model large-v3 --audio-dir recordings --commands commands.json

``faster_whisper`` is imported lazily so the reporting path (metrics, gain
advice) remains usable and testable without the model installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Pure helpers (no model, no IO side effects beyond reading WAVs)
# ---------------------------------------------------------------------------


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV as mono float32 plus its sample rate."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
    data = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        data = data.reshape(-1, channels)[:, 0]
    return data.astype(np.float32) / 32768.0, rate


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def naive_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Approximate the *implicit* audio-server resample.

    A linear interpolation with no anti-alias filter. This is not a faithful
    model of any specific resampler, but it reproduces the property that made
    the old path lossy: broadband energy folds back into the speech band.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / src_rate
    out_count = int(round(duration * dst_rate))
    target_x = np.arange(out_count, dtype=np.float64) / dst_rate
    source_x = np.arange(samples.size, dtype=np.float64) / src_rate
    return np.interp(target_x, source_x, samples.astype(np.float64)).astype(np.float32)


def correct_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """One deliberate resample through the production capture resampler."""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sayso.wake.capture import CaptureResampler

    resampler = CaptureResampler(src_rate, dst_rate)
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    out = np.frombuffer(resampler.process(pcm.tobytes()), dtype="<i2")
    return out.astype(np.float32) / 32768.0


def apply_gain(samples: np.ndarray, gain_db: float) -> tuple[np.ndarray, int]:
    """Apply fixed gain, returning the signal and the clipped-sample count."""
    if gain_db == 0.0:
        return samples, 0
    scaled = samples * (10.0 ** (gain_db / 20.0))
    clipped = int(np.sum(np.abs(scaled) > 1.0))
    return np.clip(scaled, -1.0, 1.0), clipped


def measure_levels(samples: np.ndarray) -> dict[str, float]:
    if samples.size == 0:
        return {"peak": 0.0, "rms": 0.0, "rms_dbfs": -120.0, "peak_dbfs": -120.0, "clip_pct": 0.0}
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    clip_pct = float(np.sum(np.abs(samples) >= 0.999) / samples.size * 100.0)
    return {
        "peak": peak,
        "rms": rms,
        "rms_dbfs": 20.0 * np.log10(rms) if rms > 0 else -120.0,
        "peak_dbfs": 20.0 * np.log10(peak) if peak > 0 else -120.0,
        "clip_pct": clip_pct,
    }


def normalize_text(text: str) -> list[str]:
    """Lowercase, strip punctuation, and split into comparable tokens."""
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return cleaned.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein distance over tokens.
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            current.append(min(prev[j] + 1, current[j - 1] + 1, prev[j - 1] + cost))
        prev = current
    return prev[-1] / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = " ".join(normalize_text(reference))
    hyp = " ".join(normalize_text(hypothesis))
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            current.append(min(prev[j] + 1, current[j - 1] + 1, prev[j - 1] + cost))
        prev = current
    return prev[-1] / len(ref)


def recommend_gain(levels: list[dict[str, float]], target_rms_dbfs: float = -26.0) -> dict[str, Any]:
    """Recommend one stable gain from measured speech RMS, not noise suppression.

    Uses the median speech RMS across phrases, so one shouted or one whispered
    phrase does not set the operating point. The recommendation is bounded so a
    bad recording session cannot propose a deafening gain.
    """
    measured = [lv["rms_dbfs"] for lv in levels if lv["rms_dbfs"] > -100.0]
    if not measured:
        return {"recommended_gain_db": 0.0, "median_rms_dbfs": None, "reason": "no measurable speech"}
    median = _percentile(measured, 0.5)
    gain = float(np.clip(target_rms_dbfs - median, -12.0, 24.0))
    return {
        "recommended_gain_db": round(gain, 1),
        "median_rms_dbfs": round(median, 1),
        "target_rms_dbfs": target_rms_dbfs,
        "reason": "median speech RMS across phrases",
    }


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


@dataclass
class CommandSpec:
    id: str
    text: str
    entity: Optional[str] = None
    difficulty: str = "medium"


@dataclass
class PhraseResult:
    command_id: str
    text: str
    variant: str
    hypothesis: str = ""
    wer: Optional[float] = None
    cer: Optional[float] = None
    levels: dict[str, float] = field(default_factory=dict)
    clipped: int = 0


def load_commands(path: Path) -> list[CommandSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        CommandSpec(
            id=entry["id"],
            text=entry["text"],
            entity=entry.get("entity"),
            difficulty=entry.get("difficulty", "medium"),
        )
        for entry in raw["commands"]
    ]


def build_variants(
    samples: np.ndarray,
    native_rate: int,
    target_rate: int,
    gain_db: float,
) -> dict[str, np.ndarray]:
    """Produce the two front ends for one recording."""
    current = naive_resample(samples, native_rate, target_rate)
    corrected = correct_resample(samples, native_rate, target_rate)
    current, _ = apply_gain(current, gain_db)
    corrected, _ = apply_gain(corrected, gain_db)
    return {"current": current, "corrected": corrected}


def run_benchmark(
    commands: list[CommandSpec],
    audio_dir: Path,
    transcribe: Callable[[np.ndarray, int], str],
    *,
    native_rate: int = 44100,
    target_rate: int = 16000,
    gain_db: float = 0.0,
    dump_dir: Optional[Path] = None,
) -> dict[str, Any]:
    results: list[PhraseResult] = []
    levels_by_phrase: list[dict[str, float]] = []
    missing: list[str] = []

    for command in commands:
        wav_path = audio_dir / f"{command.id}.wav"
        if not wav_path.is_file():
            missing.append(command.id)
            continue
        samples, rate = read_wav(wav_path)
        levels_by_phrase.append(measure_levels(samples))
        variants = build_variants(samples, rate, target_rate, gain_db)
        for variant_name, variant_samples in variants.items():
            if dump_dir is not None:
                write_wav(dump_dir / variant_name / f"{command.id}.wav", variant_samples, target_rate)
            hypothesis = transcribe(variant_samples, target_rate)
            results.append(
                PhraseResult(
                    command_id=command.id,
                    text=command.text,
                    variant=variant_name,
                    hypothesis=hypothesis,
                    wer=word_error_rate(command.text, hypothesis),
                    cer=character_error_rate(command.text, hypothesis),
                    levels=measure_levels(variant_samples),
                )
            )

    return {
        "results": results,
        "missing": missing,
        "levels": levels_by_phrase,
        "summary": summarize(results),
        "gain": recommend_gain(levels_by_phrase),
    }


def summarize(results: list[PhraseResult]) -> dict[str, Any]:
    by_variant: dict[str, list[PhraseResult]] = {}
    for result in results:
        by_variant.setdefault(result.variant, []).append(result)

    summary: dict[str, Any] = {}
    for variant, entries in by_variant.items():
        wers = [e.wer for e in entries if e.wer is not None]
        cers = [e.cer for e in entries if e.cer is not None]
        clip = [e.levels.get("clip_pct", 0.0) for e in entries if e.levels]
        summary[variant] = {
            "phrases": len(entries),
            "wer": round(float(np.mean(wers)), 4) if wers else None,
            "cer": round(float(np.mean(cers)), 4) if cers else None,
            "wer_p50": round(_percentile(wers, 0.5), 4) if wers else None,
            "wer_p95": round(_percentile(wers, 0.95), 4) if wers else None,
            "exact_matches": sum(1 for e in entries if e.wer == 0.0),
            "mean_clip_pct": round(float(np.mean(clip)), 3) if clip else 0.0,
        }

    if "current" in summary and "corrected" in summary:
        current_wer = summary["current"]["wer"]
        corrected_wer = summary["corrected"]["wer"]
        if current_wer is not None and corrected_wer is not None:
            summary["delta"] = {
                "wer": round(corrected_wer - current_wer, 4),
                "wer_relative": (
                    round((corrected_wer - current_wer) / current_wer, 4)
                    if current_wer
                    else None
                ),
                "exact_matches": (
                    summary["corrected"]["exact_matches"] - summary["current"]["exact_matches"]
                ),
            }
    return summary


def _faster_whisper_transcriber(model: str, device: str = "cpu", compute_type: str = "int8"):
    from faster_whisper import WhisperModel  # imported lazily

    whisper = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(samples: np.ndarray, rate: int) -> str:
        segments, _info = whisper.transcribe(samples.astype(np.float32), language="en")
        return " ".join(segment.text.strip() for segment in segments).strip()

    return transcribe


def format_report(report: dict[str, Any], native_rate: int) -> str:
    lines: list[str] = []
    summary = report["summary"]
    gain = report["gain"]
    lines.append(f"Benchmark: {len(report['results']) // 2} phrases, native rate {native_rate} Hz")
    if report["missing"]:
        lines.append(f"Skipped (no recording): {', '.join(report['missing'])}")
    lines.append("")
    header = f"{'variant':<12}{'WER':>8}{'CER':>8}{'exact':>8}{'clip%':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for variant in ("current", "corrected"):
        entry = summary.get(variant)
        if not entry:
            continue
        lines.append(
            f"{variant:<12}"
            f"{entry['wer']:>8.3f}"
            f"{entry['cer']:>8.3f}"
            f"{entry['exact_matches']:>8}"
            f"{entry['mean_clip_pct']:>8.2f}"
        )
    if "delta" in summary:
        delta = summary["delta"]
        lines.append("")
        lines.append(
            f"WER delta (corrected - current): {delta['wer']:+.3f}"
            + (f" ({delta['wer_relative']:+.1%})" if delta["wer_relative"] is not None else "")
        )
        lines.append(f"Exact-match delta: {delta['exact_matches']:+d}")
    lines.append("")
    lines.append(
        f"Gain recommendation: {gain['recommended_gain_db']} dB "
        f"(median speech RMS {gain.get('median_rms_dbfs')} dBFS, target "
        f"{gain.get('target_rms_dbfs')} dBFS)"
    )
    lines.append("Tune gain, not noise suppression: NS stays off unless a separate test proves it helps.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sayso-stt-benchmark")
    parser.add_argument("--model", required=True, help="Faster Whisper model name or path")
    parser.add_argument(
        "--commands",
        type=Path,
        default=Path(__file__).with_name("commands.json"),
    )
    parser.add_argument("--audio-dir", type=Path, default=Path(__file__).with_name("audio"))
    parser.add_argument("--native-rate", type=int, default=44100)
    parser.add_argument("--target-rate", type=int, default=16000)
    parser.add_argument("--gain-db", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="Write the exact per-variant audio sent to the model for listening",
    )
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    commands = load_commands(args.commands)
    transcribe = _faster_whisper_transcriber(args.model, args.device, args.compute_type)
    report = run_benchmark(
        commands,
        args.audio_dir,
        transcribe,
        native_rate=args.native_rate,
        target_rate=args.target_rate,
        gain_db=args.gain_db,
        dump_dir=args.dump_dir,
    )
    print(format_report(report, args.native_rate))
    if args.json:
        payload = {
            "summary": report["summary"],
            "gain": report["gain"],
            "missing": report["missing"],
            "results": [
                {
                    "command_id": r.command_id,
                    "variant": r.variant,
                    "text": r.text,
                    "hypothesis": r.hypothesis,
                    "wer": r.wer,
                    "cer": r.cer,
                }
                for r in report["results"]
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
