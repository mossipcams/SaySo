# SaySo satellite wake-word eval

Recorded-audio evaluation replaces the silence-only ONNX load check as the primary wake-word validation path.

## Bootstrap blocker (part 3 gate)

`satellite/eval/audio/` has **no trusted fixtures**. All five cases in `cases.json` skip in default mode and **fail** in `--strict` mode. Part 3 (unattended training) cannot start until trusted seed, calibration, and independent eval recordings are collected on the target hardware. See `manifest.json`, `baseline.json`, and `splits.json`.

Do not fabricate evaluation WAVs.

## Corpus layout

- `cases.json` — case definitions and expectations
- `manifest.json` — corpus metadata and bootstrap status
- `splits.json` — source-group split/lineage contract (freeze before augmentation)
- `baseline.json` — recorded deployed metrics and calibration thresholds (freezing blocked until trusted audio exists)
- `audio/` — WAV fixtures (16-bit PCM; mono preferred; resampled to 16 kHz if needed)
- `fixtures/stt/` — optional STT transcript stubs for transcript and acknowledgement timing checks

## Case categories

| Category | Purpose |
| --- | --- |
| `positive_sayso` | Isolated SaySo wake |
| `continuous_command` | SaySo plus command in one utterance |
| `negative_natural_say_so` | Natural "say so" without wake intent |
| `negative_tv_conversation` | TV or conversation negatives |
| `negative_distance_noise` | Distance and background-noise negatives |

Label audit: acoustically identical SaySo / say-so inputs in the 2 s classifier window are **unresolved ambiguity**, not separable classes. See `splits.json` `label_audit`.

## Metrics per case

- `detected` / `detection_ok` — wake fired vs expectation
- `detection_sample` / `activation_samples` — audio sample time of activations (production cooldown applied)
- `missing_first_word` — wake fired but first command word clipped in STT stub
- `stt_transcript_success` — transcript stub matches expected command when fixtures exist
- `pi_inference_ms` — p50/p95 ONNX predict latency (report on target Pi hardware)
- `background_duration_seconds` — total available TV/conversation and distance/noise negative audio
- `speech_end_to_ack_ms` — stub STT delay from speech end to acknowledgement chime

## Run

Default developer mode (skips missing audio, no refractory):

```bash
python3 satellite/eval/run.py --model /path/to/sayso.onnx
sayso-satellite test-wake-word
```

Strict baseline mode (missing/skipped cases fail; production refractory from config):

```bash
python3 satellite/eval/run.py --strict --model /path/to/sayso.onnx
```

## Tests

Synthetic WAVs in colocated tests verify scoring without the full recorded corpus:

```bash
python3 -m pytest satellite/sayso/wake/test_eval.py satellite/eval/test_run.py -q
```
