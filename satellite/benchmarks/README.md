# SaySo STT audio benchmark

A fixed 20-command benchmark that compares two audio front ends using the **same
Faster Whisper model**, so a transcription difference can only be attributed to
the audio path.

## The two front ends

| Variant | What it does |
| --- | --- |
| `current` | Requests 16 kHz from the device and lets the audio server resample implicitly, with no anti-alias filter under our control. |
| `corrected` | Captures at the device's native rate (44.1 kHz for the Snowball), applies fixed gain, and resamples once to 16 kHz through the production `CaptureResampler`. |

Both variants get the same fixed gain. The benchmark deliberately does **not**
sweep noise suppression: the working hypothesis is that gain, not NS, is the
lever, and an NS sweep would confound the comparison.

## Commands

`commands.json` holds 20 phrases. Difficult multi-word entity names are included
on purpose (`living room light switch`, `bedroom TV`, `washer dryer`,
`nursery white noise machine`): a smeared or aliased signal shows up there as an
articulation error rather than an obvious decode failure.

## Recordings

Record one WAV per command id under `--audio-dir`, named `<id>.wav`, at the
native rate:

```text
satellite/benchmarks/audio/
  living-room-light-switch.wav
  bedroom-tv.wav
  ...
```

Missing recordings are reported as skipped, never fabricated.

## Run

```bash
python3 satellite/benchmarks/run.py \
  --model large-v3 \
  --audio-dir satellite/benchmarks/audio \
  --native-rate 44100 \
  --json /tmp/bench.json \
  --dump-dir /tmp/bench-audio
```

`--dump-dir` writes the exact per-variant audio sent to the model, so the WAV a
phrase failed on can be listened to directly.

## What it reports

- Per-variant WER, CER, exact matches, and mean clipping.
- `WER delta (corrected - current)`, plus the exact-match delta.
- Per-phrase speech RMS, peak, and clip%.
- One bounded gain recommendation derived from the **median** speech RMS across
  phrases, so one shouted phrase does not set the operating point.

The output tells you to tune gain, not noise suppression.
