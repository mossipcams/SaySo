# SaySo satellite wake-word eval

Recorded-audio evaluation replaces the silence-only ONNX load check as the primary wake-word validation path.

## Bootstrap blocker (part 3 gate)

`satellite/eval/audio/` holds **silent30**, the current 30-positive candidate evaluation set recorded 2026-09-25 on the Pi's eMeet. Their masters are read-only at `llm:/srv/llm/data/wake/eval/silent30/`. Cases read `audio/silent30_leadin/`, which adds 1.5 s of real room tone so the 2 s window has lead-in (see its `derivation.json`), and the audio is not tracked in Git. This set shares speaker, room, mic, and session with the 60-take training set; preserve that lineage caveat when interpreting results. The five unrecorded diagnostic cases remain in full eval reports but are marked `promotion_required: false`. `--strict --promotion-only` omits those five absent cases and enforces all 30 silent30 eval cases. Each candidate is evaluated at its own LiveKit validation `optimal_threshold`; this does not change the deployed threshold. The baseline remains blocked until calibration data and independent negative/background recordings are collected on target hardware. See `manifest.json`, `baseline.json`, and `splits.json`.

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

Full strict diagnostic mode (all missing/skipped cases fail):

```bash
python3 satellite/eval/run.py --strict --model /path/to/sayso.onnx
```

Promotion-candidate smoke gate (uses only cases marked `promotion_required: true`):

```bash
python3 -m satellite.eval.run --strict --promotion-only --model /path/to/sayso.onnx
```

Pass `--threshold` with the candidate's LiveKit validation
`optimal_threshold`; the experiment runner does this automatically and records
the applied threshold in both the report and result metadata.

## Tests

Synthetic WAVs in colocated tests verify scoring without the full recorded corpus:

```bash
python3 -m pytest satellite/sayso/wake/test_eval.py satellite/eval/test_run.py -q
```
