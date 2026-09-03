# SaySo satellite wake-word eval

Recorded-audio evaluation replaces the silence-only ONNX load check as the primary wake-word validation path.

## Corpus layout

- `cases.json` — case definitions and expectations
- `manifest.json` — corpus metadata
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

## Metrics per case

- `detected` / `detection_ok` — wake fired vs expectation
- `missing_first_word` — wake fired but first command word clipped in STT stub
- `stt_transcript_success` — transcript stub matches expected command when fixtures exist
- `pi_inference_ms` — p50/p95 ONNX predict latency (Pi-class hardware target)
- `speech_end_to_ack_ms` — stub STT delay from speech end to acknowledgement chime

## Run

```bash
python3 satellite/eval/run.py --model /path/to/sayso.onnx
sayso-satellite test-wake-word
```

Missing audio fixtures are **skipped** (not failed) in default pytest and when running the harness without a full corpus.

## Tests

Synthetic WAVs in colocated tests verify scoring without the full recorded corpus:

```bash
python3 -m pytest satellite/sayso/wake/test_eval.py satellite/eval/test_run.py -q
```
