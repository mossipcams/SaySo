"""Recorded-audio wake-word evaluation harness."""

from __future__ import annotations

import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .buffer import WakeAudioBuffer
from .livekit import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES, LiveKitWakeWordProvider

WAKE_EVAL_CATEGORIES = frozenset(
    {
        "positive_sayso",
        "continuous_command",
        "negative_natural_say_so",
        "negative_tv_conversation",
        "negative_distance_noise",
    }
)

CHUNK_SAMPLES = 512


def satellite_eval_root() -> Path:
    return Path(__file__).resolve().parents[2] / "eval"


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def compute_latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    ordered = sorted(latencies_ms)
    return {"p50": _percentile(ordered, 0.5), "p95": _percentile(ordered, 0.95)}


@dataclass(frozen=True)
class WakeEvalCase:
    id: str
    category: str
    audio: str
    expect_detection: bool
    command_transcript: Optional[str] = None
    expected_transcript: Optional[str] = None
    transcript_fixture: Optional[str] = None
    speech_end_ms: Optional[float] = None
    expect_missing_first_word: Optional[bool] = None


@dataclass(frozen=True)
class WakeEvalCaseSet:
    version: int
    cases: tuple[WakeEvalCase, ...]


@dataclass(frozen=True)
class WakeCaseResult:
    case_id: str
    category: str
    status: str
    detected: Optional[bool] = None
    detection_ok: Optional[bool] = None
    detection_sample: Optional[int] = None
    activation_samples: Optional[tuple[int, ...]] = None
    missing_first_word: Optional[bool] = None
    missing_first_word_ok: Optional[bool] = None
    stt_transcript_success: Optional[bool] = None
    actual_transcript: Optional[str] = None
    speech_end_to_ack_ms: Optional[float] = None
    pi_inference_ms: Optional[dict[str, float]] = None
    inference_ms: tuple[float, ...] = ()
    audio_duration_seconds: float = 0.0
    skip_reason: Optional[str] = None
    error: Optional[str] = None


def _parse_case(raw: dict[str, Any]) -> WakeEvalCase:
    category = str(raw["category"])
    if category not in WAKE_EVAL_CATEGORIES:
        raise ValueError(f"unknown wake eval category: {category}")
    return WakeEvalCase(
        id=str(raw["id"]),
        category=category,
        audio=str(raw["audio"]),
        expect_detection=bool(raw.get("expect_detection", False)),
        command_transcript=raw.get("command_transcript"),
        expected_transcript=raw.get("expected_transcript"),
        transcript_fixture=raw.get("transcript_fixture"),
        speech_end_ms=float(raw["speech_end_ms"]) if raw.get("speech_end_ms") is not None else None,
        expect_missing_first_word=(
            bool(raw["expect_missing_first_word"])
            if raw.get("expect_missing_first_word") is not None
            else None
        ),
    )


def load_wake_cases(path: Path) -> WakeEvalCaseSet:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = int(payload["version"])
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("cases must be a list")
    cases = tuple(_parse_case(entry) for entry in raw_cases if isinstance(entry, dict))
    return WakeEvalCaseSet(version=version, cases=cases)


def read_wav_pcm(path: Path, target_rate: int = SAMPLE_RATE) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM in {path}")
    data = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        data = data.reshape(-1, channels)[:, 0]
    if rate != target_rate:
        duration = data.size / float(rate)
        target_size = max(int(round(duration * target_rate)), 1)
        source_x = np.linspace(0.0, duration, num=data.size, endpoint=False)
        target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
        resampled = np.interp(target_x, source_x, data.astype(np.float64))
        data = np.clip(resampled, -32768, 32767).astype("<i2")
        rate = target_rate
    return data.tobytes(), rate


def _normalize_transcript(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _command_after_wake(command: str) -> list[str]:
    return _normalize_transcript(command).split()


def detect_missing_first_word(command_transcript: str, actual_transcript: str) -> bool:
    """True when the first command word after wake is absent from the STT stub."""
    expected_words = _command_after_wake(command_transcript)
    if not expected_words:
        return False
    actual_norm = _normalize_transcript(actual_transcript)
    first_word = expected_words[0]
    actual_words = actual_norm.split()
    if not actual_words:
        return True
    if actual_words[0] == first_word:
        return False
    # ponytail: substring fallback for partial transcripts; upgrade to fuzzy match if corpus grows
    return first_word not in actual_words[:2]


def transcript_matches(expected: str, actual: str) -> bool:
    expected_norm = _normalize_transcript(expected)
    actual_norm = _normalize_transcript(actual)
    return expected_norm == actual_norm or expected_norm in actual_norm


def load_transcript_fixture(eval_root: Path, relative_path: str) -> dict[str, Any]:
    path = eval_root / relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"transcript fixture must be a JSON object: {path}")
    return payload


def scan_wake_audio(
    provider: LiveKitWakeWordProvider,
    pcm: bytes,
    sample_rate: int = SAMPLE_RATE,
    *,
    predict: Optional[Callable[..., Any]] = None,
) -> tuple[bool, list[float], Optional[int], tuple[int, ...]]:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected sample rate {SAMPLE_RATE}, got {sample_rate}")
    buffer = WakeAudioBuffer(WINDOW_SAMPLES, HOP_SAMPLES)
    provider.start()
    inference_ms: list[float] = []
    activation_samples: list[int] = []
    predict_fn = predict

    samples = np.frombuffer(pcm, dtype="<i2")
    for start in range(0, samples.size, CHUNK_SAMPLES):
        chunk = samples[start : start + CHUNK_SAMPLES]
        if chunk.size == 0:
            continue
        window_end = start + chunk.size
        if buffer.feed(chunk.tobytes()):
            window = buffer.window()
            start_time = time.perf_counter()
            if predict_fn is not None:
                detection = predict_fn(window)
            else:
                try:
                    detection = provider.predict_window(window, sample_index=window_end)
                except TypeError:
                    detection = provider.predict_window(window)
            inference_ms.append((time.perf_counter() - start_time) * 1000.0)
            if detection is not None:
                sample_idx = getattr(detection, "sample_index", None)
                activation_samples.append(int(sample_idx if sample_idx is not None else window_end))

    first_sample = activation_samples[0] if activation_samples else None
    return first_sample is not None, inference_ms, first_sample, tuple(activation_samples)


def evaluate_case(
    case: WakeEvalCase,
    eval_root: Path,
    provider: LiveKitWakeWordProvider,
) -> WakeCaseResult:
    provider.reset()
    audio_path = eval_root / case.audio
    if not audio_path.is_file():
        return WakeCaseResult(
            case_id=case.id,
            category=case.category,
            status="skipped",
            skip_reason=f"missing audio: {case.audio}",
        )

    try:
        pcm, rate = read_wav_pcm(audio_path)
    except (OSError, ValueError, wave.Error) as exc:
        return WakeCaseResult(
            case_id=case.id,
            category=case.category,
            status="error",
            error=str(exc),
        )

    detected, inference_ms, detection_sample, activation_samples = scan_wake_audio(provider, pcm, rate)
    audio_duration_seconds = len(pcm) / 2 / rate
    detection_ok = detected == case.expect_detection
    pi_inference = compute_latency_percentiles(inference_ms) if inference_ms else {"p50": 0.0, "p95": 0.0}

    actual_transcript: Optional[str] = None
    stt_success: Optional[bool] = None
    missing_first_word: Optional[bool] = None
    missing_first_word_ok: Optional[bool] = None
    speech_end_to_ack_ms: Optional[float] = None

    fixture_path = case.transcript_fixture
    if fixture_path:
        fixture_file = eval_root / fixture_path
        if not fixture_file.is_file():
            return WakeCaseResult(
                case_id=case.id,
                category=case.category,
                status="skipped",
                skip_reason=f"missing transcript fixture: {fixture_path}",
                detected=detected,
                detection_ok=detection_ok,
                detection_sample=detection_sample,
                activation_samples=activation_samples or None,
                pi_inference_ms=pi_inference,
                inference_ms=tuple(inference_ms),
                audio_duration_seconds=audio_duration_seconds,
            )
        fixture = load_transcript_fixture(eval_root, fixture_path)
        actual_transcript = str(fixture.get("text", "")).strip()
        stt_delay_ms = float(fixture.get("stt_delay_ms", 0.0))
        speech_end_to_ack_ms = stt_delay_ms

        expected = case.expected_transcript or case.command_transcript
        if expected is not None:
            stt_success = transcript_matches(expected, actual_transcript)

        if detected and case.command_transcript and actual_transcript:
            missing_first_word = detect_missing_first_word(case.command_transcript, actual_transcript)
            if case.expect_missing_first_word is not None:
                missing_first_word_ok = missing_first_word == case.expect_missing_first_word

    passed = detection_ok
    if stt_success is not None:
        passed = passed and stt_success
    if missing_first_word_ok is not None:
        passed = passed and missing_first_word_ok

    return WakeCaseResult(
        case_id=case.id,
        category=case.category,
        status="passed" if passed else "failed",
        detected=detected,
        detection_ok=detection_ok,
        detection_sample=detection_sample,
        activation_samples=activation_samples or None,
        missing_first_word=missing_first_word,
        missing_first_word_ok=missing_first_word_ok,
        stt_transcript_success=stt_success,
        actual_transcript=actual_transcript,
        speech_end_to_ack_ms=speech_end_to_ack_ms,
        pi_inference_ms=pi_inference,
        inference_ms=tuple(inference_ms),
        audio_duration_seconds=audio_duration_seconds,
    )


def _finalize_case_status(result: WakeCaseResult, *, strict: bool) -> WakeCaseResult:
    if not strict or result.status not in {"skipped", "error"}:
        return result
    reason = result.skip_reason or result.error or result.status
    return WakeCaseResult(
        case_id=result.case_id,
        category=result.category,
        status="failed",
        skip_reason=result.skip_reason,
        error=result.error,
        detected=result.detected,
        detection_ok=False,
        detection_sample=result.detection_sample,
        activation_samples=result.activation_samples,
        missing_first_word=result.missing_first_word,
        missing_first_word_ok=result.missing_first_word_ok,
        stt_transcript_success=result.stt_transcript_success,
        actual_transcript=result.actual_transcript,
        speech_end_to_ack_ms=result.speech_end_to_ack_ms,
        pi_inference_ms=result.pi_inference_ms,
        inference_ms=result.inference_ms,
        audio_duration_seconds=result.audio_duration_seconds,
    ) if reason else result


def run_wake_eval(
    model_path: Path,
    eval_root: Path,
    *,
    phrase: str = "SaySo",
    threshold: float = 0.65,
    refractory_seconds: float = 0.0,
    strict: bool = False,
    hardware: str = "unknown",
) -> dict[str, Any]:
    cases_path = eval_root / "cases.json"
    if not cases_path.is_file():
        errors = 1 if strict else 0
        return {
            "version": 0,
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": errors,
            },
            "aggregate": {
                "pi_inference_ms": {"p50": 0.0, "p95": 0.0},
                "background_duration_seconds": 0.0,
            },
            "strict": strict,
            "hardware": hardware,
            "results": [],
            "note": f"missing case manifest: {cases_path}",
        }

    case_set = load_wake_cases(cases_path)
    provider = LiveKitWakeWordProvider(
        model_path=model_path,
        phrase=phrase,
        threshold=threshold,
        refractory_seconds=refractory_seconds,
    )
    if not provider.available:
        raise RuntimeError(f"wake model unavailable: {model_path}")

    results: list[WakeCaseResult] = []
    all_inference_ms: list[float] = []
    background_duration_seconds = 0.0
    for case in case_set.cases:
        result = _finalize_case_status(evaluate_case(case, eval_root, provider), strict=strict)
        results.append(result)
        all_inference_ms.extend(result.inference_ms)
        if case.category in {"negative_tv_conversation", "negative_distance_noise"}:
            background_duration_seconds += result.audio_duration_seconds

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.status == "passed"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "errors": sum(1 for r in results if r.status == "error") + int(strict and not results),
    }

    return {
        "version": case_set.version,
        "strict": strict,
        "hardware": hardware,
        "refractory_seconds": refractory_seconds,
        "summary": summary,
        "aggregate": {
            "pi_inference_ms": compute_latency_percentiles(all_inference_ms),
            "background_duration_seconds": background_duration_seconds,
        },
        "results": [case_result_to_dict(r) for r in results],
    }


def case_result_to_dict(result: WakeCaseResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "case_id": result.case_id,
        "category": result.category,
        "status": result.status,
    }
    if result.skip_reason is not None:
        payload["skip_reason"] = result.skip_reason
    if result.error is not None:
        payload["error"] = result.error
    if result.detected is not None:
        payload["detected"] = result.detected
    if result.detection_ok is not None:
        payload["detection_ok"] = result.detection_ok
    if result.detection_sample is not None:
        payload["detection_sample"] = result.detection_sample
    if result.activation_samples is not None:
        payload["activation_samples"] = list(result.activation_samples)
    if result.missing_first_word is not None:
        payload["missing_first_word"] = result.missing_first_word
    if result.missing_first_word_ok is not None:
        payload["missing_first_word_ok"] = result.missing_first_word_ok
    if result.stt_transcript_success is not None:
        payload["stt_transcript_success"] = result.stt_transcript_success
    if result.actual_transcript is not None:
        payload["actual_transcript"] = result.actual_transcript
    if result.speech_end_to_ack_ms is not None:
        payload["speech_end_to_ack_ms"] = result.speech_end_to_ack_ms
    if result.pi_inference_ms is not None:
        payload["pi_inference_ms"] = result.pi_inference_ms
    return payload


def false_activations_per_hour(
    *,
    activation_count: int,
    duration_seconds: float,
) -> float:
    if duration_seconds <= 0:
        return float("inf")
    hours = duration_seconds / 3600.0
    return activation_count / hours if hours > 0 else float("inf")


def activation_matches_labeled_positive(
    sample_index: int,
    labeled_positives: Sequence[Any],
) -> bool:
    for event in labeled_positives:
        if getattr(event, "label", None) != "positive":
            continue
        start = int(getattr(event, "sample_start", 0) or 0)
        end = int(getattr(event, "sample_end", 0) or 0)
        if start <= sample_index <= end:
            return True
    return False


def filter_false_activations(
    activation_samples: Sequence[int],
    labeled_positives: Sequence[Any] | None = None,
) -> list[int]:
    labeled = labeled_positives or []
    return [
        sample
        for sample in activation_samples
        if not activation_matches_labeled_positive(sample, labeled)
    ]


def evaluate_holdout_session(
    session: "RecordingSession",
    provider: LiveKitWakeWordProvider,
    *,
    predict: Optional[Callable[..., Any]] = None,
    labeled_positives: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Continuous replay of one holdout session; report detections and FA/hour."""
    from .replay import collect_session_activations
    from .sessions import RecordingSession

    _ = RecordingSession  # re-export for type checkers
    activation_samples, inference_ms, duration_seconds, detected = collect_session_activations(
        session,
        provider,
        predict=predict,
    )
    false_activations = filter_false_activations(activation_samples, labeled_positives)
    detection_sample = activation_samples[0] if activation_samples else None
    return {
        "session_id": session.session_id,
        "duration_seconds": duration_seconds,
        "detected": detected,
        "detection_sample": detection_sample,
        "activation_samples": list(activation_samples),
        "activation_count": len(activation_samples),
        "false_activation_count": len(false_activations),
        "false_activations_per_hour": false_activations_per_hour(
            activation_count=len(false_activations),
            duration_seconds=duration_seconds,
        ),
        "pi_inference_ms": compute_latency_percentiles(inference_ms) if inference_ms else {"p50": 0.0, "p95": 0.0},
    }


def evaluate_holdout_sessions(
    sessions: Sequence["RecordingSession"],
    provider: LiveKitWakeWordProvider,
    *,
    predict: Optional[Callable[..., Any]] = None,
    labeled_positives_by_session: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    from .sessions import RecordingSession

    _ = RecordingSession
    labeled_by_session = labeled_positives_by_session or {}
    results = [
        evaluate_holdout_session(
            session,
            provider,
            predict=predict,
            labeled_positives=labeled_by_session.get(session.session_id),
        )
        for session in sessions
    ]
    total_seconds = sum(float(entry["duration_seconds"]) for entry in results)
    total_false_activations = sum(int(entry["false_activation_count"]) for entry in results)
    return {
        "sessions": results,
        "aggregate": {
            "session_count": len(results),
            "duration_seconds": total_seconds,
            "activation_count": sum(int(entry["activation_count"]) for entry in results),
            "false_activation_count": total_false_activations,
            "false_activations_per_hour": false_activations_per_hour(
                activation_count=total_false_activations,
                duration_seconds=total_seconds,
            ),
        },
    }


def write_synthetic_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype("<i2").tobytes())
