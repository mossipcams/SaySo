# SaySo satellite wake-word eval

Recorded-audio evaluation replaces the silence-only ONNX load check as the primary wake-word validation path.

## Wake rule

The phrase `/seɪ soʊ/` wakes **only in isolation**. Any word in front of it
disqualifies the wake; a word after it does not. Both realizations — the fused
"SaySo" and the two-word "say so" — are wakes. The rule and its consequences
live in `../../docs/WAKE_WORD_DATA.md`; `cases.json` is its executable form.

## Corpus layout

- `cases.json` — case definitions, expectations, and a `notes` line per case
  saying what its audio holds
- `manifest.json` — corpus metadata
- `audio/` — WAV fixtures (16-bit PCM; mono preferred; resampled to 16 kHz if needed)
- `fixtures/stt/` — optional STT transcript stubs for transcript and acknowledgement timing checks

## Where the audio lives

**The recordings are not in the repo.** This repository is public and the corpus
is real household speech, so `audio/` is gitignored except for `.gitkeep`.
`cases.json` is the tracked definition; the WAVs are built outside it:

```bash
# build a corpus outside the repo from the labelled mining spool
python scripts/wake_prepare_real_data.py eval-corpus \
    --spool /var/lib/sayso-satellite/wake-mining \
    --ledger satellite/models/wake-spool-labels-20260914.json \
    --eval-root ~/sayso-eval-corpus \
    --held-out satellite/models/wake-held-out-20260914.json

# run the harness against it
python satellite/eval/run.py --model satellite/models/sayso.onnx \
    --eval-root ~/sayso-eval-corpus
```

The builder copies `cases.json` into the eval root if it is there already, picks
one utterance per case from the ledger (highest-scoring, distinct utterances
where the spool allows), writes the STT stubs, and records every window it used
in `--held-out`. Those clusters are then excluded from training entirely — not
just from `positive_train` but from the `*_test` splits too, because those are
what select the operating threshold, and the eval must not be judged on the
clips that chose its threshold.

Cases the spool cannot fill are reported as `NOT populated` and stay skipped:
the fused "Sayso" realization is not attested in the current recordings, and no
distance capture exists yet. Read the `skipped` count before believing a run: an
all-skipped run is green and says nothing.

## Case categories

| Category | Purpose | Expects detection |
| --- | --- | --- |
| `positive_sayso` | Isolated fused SaySo, near and at distance | yes |
| `positive_say_so_two_word` | Isolated two-word "say so" | yes |
| `continuous_command` | SaySo plus command in one utterance | yes |
| `negative_say_so_carrier` | Phrase said **after another word** — not a wake | no |
| `negative_tv_conversation` | TV or conversation, no wake phrase | no |
| `negative_distance_noise` | Distance chatter or room noise, no wake phrase | no |

The carrier category replaces the old `negative_natural_say_so` ("natural
speech containing say so without an intentional wake"). That description was
false under the wake rule: the phrase alone *is* a wake, and only a preceding
word makes it not one. A carrier that fires is a false positive, and the pair
`negative_say_so_carrier_*` / `positive_say_so_*` is what keeps "the model
rejects carriers" apart from "the model rejects the wake word".

## Metrics per case

- `detected` / `detection_ok` — wake fired vs expectation
- `missing_first_word` — wake fired but first command word clipped in STT stub
- `stt_transcript_success` — transcript stub matches expected command when fixtures exist
- `pi_inference_ms` — p50/p95 ONNX predict latency (Pi-class hardware target)
- `speech_end_to_ack_ms` — stub STT delay from speech end to acknowledgement chime

The report also carries `by_category`, with `detection_rate` per category over
scored cases only. On a positive category that is recall; on a negative category
it is the false-positive rate. Skips are counted separately and do not dilute
it.

## Run

```bash
python3 satellite/eval/run.py --model /path/to/sayso.onnx --eval-root <corpus>
sayso-satellite test-wake-word
```

`sayso-satellite test-wake-word` uses `satellite_eval_root()`, i.e. this
directory, where `audio/` is empty — so it skips every case on a clean checkout.
It is a load check, not a wake-word regression test; point `run.py` at a built
corpus for that.

## Tests

Synthetic WAVs in colocated tests verify scoring without the full recorded
corpus:

```bash
python3 -m pytest satellite/sayso/wake/test_eval.py satellite/eval/test_run.py -q
```

`test_shipped_corpus_matches_the_isolation_rule` reads the committed
`cases.json` and fails if a carrier case ever expects a detection, so the wake
rule cannot be edited out of the corpus by accident.
