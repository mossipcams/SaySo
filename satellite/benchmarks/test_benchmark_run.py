"""Benchmark logic: metrics, resample comparison, and gain advice."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "satellite"))

from benchmarks.run import (  # noqa: E402
    CommandSpec,
    apply_gain,
    build_variants,
    character_error_rate,
    correct_resample,
    format_report,
    load_commands,
    measure_levels,
    naive_resample,
    recommend_gain,
    run_benchmark,
    word_error_rate,
    write_wav,
)


def test_load_commands_has_twenty_fixed_commands() -> None:
    commands = load_commands(Path(__file__).with_name("commands.json"))
    assert len(commands) == 20
    ids = [c.id for c in commands]
    assert len(set(ids)) == 20


def test_commands_include_the_difficult_entity_names() -> None:
    commands = load_commands(Path(__file__).with_name("commands.json"))
    texts = " ".join(c.text for c in commands).lower()
    assert "living room light switch" in texts
    assert "bedroom tv" in texts


def test_word_error_rate_is_zero_for_exact_match() -> None:
    assert word_error_rate("turn on the bedroom TV", "turn on the bedroom tv.") == 0.0


def test_word_error_rate_counts_substitutions() -> None:
    assert word_error_rate("turn on the light", "turn on the bright") == pytest.approx(0.25)


def test_word_error_rate_is_bounded_for_empty_hypothesis() -> None:
    assert word_error_rate("turn on the light", "") == 1.0


def test_character_error_rate_is_finer_grained_than_wer() -> None:
    reference = "bedroom tv"
    near = "bedrooms tv"
    assert character_error_rate(reference, near) < word_error_rate(reference, near)


def test_correct_resample_preserves_length() -> None:
    samples = np.sin(np.linspace(0, 40 * np.pi, 44100)).astype(np.float32)
    out = correct_resample(samples, 44100, 16000)
    assert abs(out.size - 16000) <= 8


def test_correct_resample_keeps_a_tone_clean_where_naive_aliases() -> None:
    """A band-limited tone must survive the deliberate resample intact."""
    rate = 44100
    target = 16000
    tone_hz = 440.0
    t = np.arange(rate, dtype=np.float64) / rate
    samples = (0.5 * np.sin(2 * np.pi * tone_hz * t)).astype(np.float32)

    good = correct_resample(samples, rate, target)
    # Recover the dominant frequency of each version.
    def dominant_hz(signal: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / target)
        return float(freqs[int(np.argmax(spectrum))])

    assert abs(dominant_hz(good) - tone_hz) < 20.0


def test_naive_resample_is_the_uncorrected_comparison() -> None:
    """The naive path must be different, or the benchmark compares nothing."""
    rate = 44100
    t = np.arange(rate, dtype=np.float64) / rate
    samples = (0.5 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    naive = naive_resample(samples, rate, 16000)
    corrected = correct_resample(samples, rate, 16000)
    assert naive.size > 0 and corrected.size > 0
    # Both roughly the same length; their content differs.
    assert abs(naive.size - corrected.size) <= 8
    n = min(naive.size, corrected.size)
    assert not np.allclose(naive[:n], corrected[:n], atol=1e-3)


def test_apply_gain_reports_clipping() -> None:
    samples = np.full(100, 0.8, dtype=np.float32)
    scaled, clipped = apply_gain(samples, 6.0)
    assert clipped > 0
    assert float(np.max(scaled)) <= 1.0


def test_measure_levels_reports_rms_and_peak() -> None:
    samples = np.full(1000, 0.25, dtype=np.float32)
    levels = measure_levels(samples)
    assert levels["peak"] == pytest.approx(0.25)
    assert levels["rms_dbfs"] == pytest.approx(20 * np.log10(0.25), rel=1e-3)


def test_recommend_gain_targets_median_speech_rms() -> None:
    levels = [{"rms_dbfs": -32.0}, {"rms_dbfs": -30.0}, {"rms_dbfs": -34.0}]
    advice = recommend_gain(levels, target_rms_dbfs=-26.0)
    # Median is -32, so +6 dB reaches the -26 dBFS target.
    assert advice["recommended_gain_db"] == pytest.approx(6.0, abs=1.0)


def test_recommend_gain_is_bounded() -> None:
    advice = recommend_gain([{"rms_dbfs": -90.0}])
    assert advice["recommended_gain_db"] <= 24.0
    advice = recommend_gain([{"rms_dbfs": -3.0}])
    assert advice["recommended_gain_db"] >= -12.0


def test_run_benchmark_skips_missing_recordings(tmp_path: Path) -> None:
    commands = [CommandSpec(id="a", text="turn on the light")]
    report = run_benchmark(commands, tmp_path, lambda s, r: "turn on the light")
    assert report["missing"] == ["a"]
    assert report["results"] == []


def test_run_benchmark_compares_both_variants_with_one_model(tmp_path: Path) -> None:
    commands = [
        CommandSpec(id="living-room-light-switch", text="turn on the living room light switch"),
        CommandSpec(id="bedroom-tv", text="turn on the bedroom TV"),
    ]
    for command in commands:
        samples = np.zeros(44100, dtype=np.float32)
        write_wav(tmp_path / f"{command.id}.wav", samples, 44100)

    seen_rates: list[int] = []

    def transcribe(samples: np.ndarray, rate: int) -> str:
        seen_rates.append(rate)
        return "turn on the living room light switch"

    report = run_benchmark(commands, tmp_path, transcribe)

    assert seen_rates == [16000] * 4  # two variants x two phrases
    variants = {r.variant for r in report["results"]}
    assert variants == {"current", "corrected"}
    assert "delta" in report["summary"]


def test_run_benchmark_dumps_listenable_variants(tmp_path: Path) -> None:
    commands = [CommandSpec(id="x", text="hello")]
    samples = np.sin(np.linspace(0, 100, 44100)).astype(np.float32)
    write_wav(tmp_path / "x.wav", samples, 44100)
    dump = tmp_path / "dump"
    run_benchmark(commands, tmp_path, lambda s, r: "hello", dump_dir=dump)
    assert (dump / "current" / "x.wav").is_file()
    assert (dump / "corrected" / "x.wav").is_file()


def test_format_report_mentions_gain_over_noise_suppression(tmp_path: Path) -> None:
    commands = [CommandSpec(id="x", text="hello")]
    samples = np.full(44100, 0.1, dtype=np.float32)
    write_wav(tmp_path / "x.wav", samples, 44100)
    report = run_benchmark(commands, tmp_path, lambda s, r: "hello")
    text = format_report(report, 44100)
    assert "Gain recommendation" in text
    assert "not noise suppression" in text
