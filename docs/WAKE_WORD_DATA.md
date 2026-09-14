# SaySo wake-word data strategy

Decision record for how `sayso.onnx` gets its training data. Every future model
should be traceable to the rules here.

## Status: spool exists, labelling in progress

**Update 2026-09-14.** The blocked state recorded below is the pre-collection
history; the blocker is cleared. `wake_word.mine_dir` has 152 scored windows /
66 utterance clusters from the living-room satellite
(`/var/lib/sayso-satellite/wake-mining`). The analysis is in
`docs/PLAN_WAKE_WORD_REAL_DATA.md`. Two findings from that drain change the
workflow stated above:

- The spool is **mostly genuine wakes** — under the isolation rule 40 of 66
clusters are isolated positives against 25 negatives and 1 the transcripts
cannot call — so "real production audio -> identify false positives" is the
wrong framing. Mined does not mean negative. Ingesting the spool as the
hard-negative set its docstring promises would train the model to reject the
wake word.
- Every sidecar is still `label: null`. Labelling is the outstanding task, and
the labels follow the wake rule below rather than the score.
`scripts/wake_label_spool.py` applies the rule and emits the per-cluster verdict;
`satellite/models/wake-spool-labels-20260914.json` is that ledger for this spool
(16 of 66 clusters need a listen before their label means anything).

## Wake rule (decided 2026-09-14): the phrase must be isolated

**If another word is in front of the wake word, it is not a wake.**

- The phrase is `/seɪ soʊ/`, in two realizations: the fused "SaySo" and the
two-word "say so". Both wake.
- A word *after* the phrase does not disqualify it. "SaySo turn on the TV" is a
wake; that is the continuous-command shape the product depends on.
- Anything spoken in front of the phrase disqualifies it. "if you say so",
  "just say so", "they say so", "assistant tools say so" are negatives.
- Containment — any occurrence wakes — was considered and rejected: it wakes on
  TV dialogue and on quoted speech, and nothing in the pipeline can claw that
  back afterwards.

This is a labelling rule, not a new runtime gate. The classifier's window is
2 s and the mined window is the exact array it scored, so a word said ~0.4 s
before the phrase is inside the window the model decides on. The rule is
learnable from audio that already reaches the classifier; no second signal, no
phrase-boundary detector, no runtime change.

Where the rule is enforced:

- `satellite/models/sayso-training.yaml` — `target_phrases` lists both
  realizations; the carriers stay in `custom_negative_phrases` and are the only
  place the model is taught what a preceded phrase looks like.
- `satellite/eval/cases.json` — the carrier category
  (`negative_say_so_carrier`) pins the negative side; the isolated case pins the
  positive side. Both halves are required, because a model that rejects carriers
  by rejecting the phrase has not learned the rule.
- Spool labelling — a window that contains the phrase preceded by speech is a
  negative no matter how high it scored. `scripts/wake_label_spool.py` applies
  the rule to a spool plus transcripts and emits a review ledger; it does not
  write sidecar `label`s, because labelling stays a listening decision.

The intended workflow is:

```
real production audio -> identify false positives -> cluster failure modes
-> compare against training coverage -> select real hard negatives
-> targeted synthetic expansion -> retrain -> evaluate on untouched real audio
```

As of 2026-09-09 this could not start. **No real wake-word audio existed
anywhere.** A full search of the satellite filesystem, this repository, and its
git history found only generated UI chimes, upstream LVA button sounds, and one
88-second ambient clip in `/tmp`. `satellite/eval/audio/` contains `.gitkeep` and
nothing else, so all five cases in `satellite/eval/cases.json` — including
`negative_natural_say_so` and `negative_tv_conversation` — skip silently. (That
category is now `negative_say_so_carrier`; the audio directory is still empty.)

The satellite never retained detection audio. Its own log says so:
`Wake phrase detected ... (no audio retained)`. Worse for diagnosis,
`flush_preroll()` hands STT only `[detection_index - wake_skip_ms, end)`, and
`wake_skip_ms` is a 120 ms margin sized to the detection lag, so the
transcript attached to a false positive describes what was said *after*
the trigger. The audio that actually fired the model was never observed, by
anyone, at any point.

Since `2861fa6` this is fixed: setting `wake_word.mine_dir` makes
`HardNegativeMiner` write the exact 2 s window the classifier scored
(`satellite/sayso/wake/mining.py`). The paragraphs below record the state of
the world before that landed.

This is why every previous attempt to fix false positives had to guess at hard
negatives. The hard negatives in `satellite/models/sayso-training.yaml` are
phonetic inference and are explicitly marked as such.
## What the telemetry does establish

126.8 hours of per-second peak scores from the journal (real production audio,
threshold-independent):

| Percentile | Score |
|---|---|
| p50 | 0.0039 |
| p99 | 0.0131 |
| p99.9 | 0.0311 |
| p99.99 | 0.1926 |
| p99.999 | 0.5618 |

Normal conversation is a tight cluster near 0.004; 99.99% of production audio
sits below 0.19. The model is not broadly confused. The entire false-positive
problem lives in an extreme tail of ~116 windows out of 456,382. **Scaling
dataset volume cannot fix a tail this thin** — the failure is coverage of a rare,
specific acoustic event, which is exactly the quality-over-volume argument.
Identifying it requires the audio.

## Ingestion rules (implemented)

`satellite/sayso/wake/mining.py`, wired through `LiveKitWakeWordProvider`.

- Capture is opt-in: `wake_word.mine_dir` unset means no mining.
- Mining is evaluated **before** the detect-threshold and refractory early
  returns, so near-misses and refractory-suppressed windows are captured. Windows
  just under threshold sit closest to the decision boundary and are the most
  valuable negatives; mining only what already fired re-collects what is already
  known.
- `mine_threshold` must be strictly below `threshold`; `validate_config` rejects
  otherwise.
- The clip written is the exact `np.ndarray` handed to the classifier, so a mined
  clip is bit-exact with what inference saw. Verified by round-trip assertion in
  `mining.demo()`.
- Clips are **unlabelled at capture**. A window over threshold may be a genuine
  wake or a false positive and only a human listening can say which. The sidecar
  records score, `fired`, detect/mine thresholds, model path and UTC timestamp;
  `label`, `transcript` and `notes` stay null until review. A high score is not a
  negative and a `say so` transcript is not a positive: the isolation rule above
  decides the label.
- Spool capped at `DEFAULT_MAX_CLIPS` (2000, ~128 MB). Full spool logs once and
  stops writing rather than filling the disk.
- Mining runs on the wake worker thread after predict, and never raises: a
  mining failure must not take down wake detection.

Live config (`/etc/sayso-satellite/config.yaml`): `mine_dir:
/var/lib/sayso-satellite/wake-mining`, `mine_threshold: 0.1`. At the measured
tail that is ~22 clips/day.

Review with `scripts/wake_mine_report.py`.

## Train/eval split rule

**Split at the session level, never the window level.** Neighbouring 2-second
windows overlap by ~92% at a 160 ms hop, so a random window-level split puts
near-identical audio on both sides and reports a recall that does not exist.
Partition by recording/session/conversation, then take windows.

Every failure cluster keeps representatives permanently held out. Evaluation runs
continuous audio through the production sliding-window path, not isolated clips.

## Deliberately not built yet

Clustering, dedup, representative sampling, and the coverage report are
specified but unimplemented. All four need a real score and similarity
distribution to calibrate against. Build them once the spool has a few hundred
labelled clips. (The 152-window spool is one speaker in one room and is not that
yet; the isolation-aware labelling pass, `scripts/wake_label_spool.py`, is not
one of these four and is implemented.)

## Resolved: the satellite was dropping and misaligning inference windows

Fixed 2026-09-09. Three compounding defects, all in the inference path:

1. **Redundant embedding passes.** `WakeWordModel.predict()` is stateless: it
   recomputes mel over the full 2 s and runs the speech-embedding model 16 times
   per call, over audio 92% identical to the previous call. Now cached —
   `wake/streaming.py` reuses embeddings across the overlap and computes 2 per
   hop instead of 16.
2. **Window misalignment, which made the cache useless and predates it.**
   `WakeAudioBuffer.feed()` reset its counter to zero on emit, so windows
   advanced by the caller's chunk size rounded up to the hop — 4000 samples in
   production, or 25 mel frames. Not a whole number of stride-8 embedding steps,
   so measured reuse in production was exactly **zero**, and, far worse, the
   wake phrase landed at an arbitrary offset in every window. The buffer now
   emits on a fixed sample grid, so the advance is exactly 160 ms regardless of
   chunk size.
3. **ONNX Runtime threading.** Default settings ran the wake models at 53.6 ms
   p50 while burning 399% CPU; one non-spinning thread runs them at 39.2 ms p50
   for 100%. The graphs are far too small to parallelise. Scoped to the wake
   models via `single_threaded_ort()`.

Measured result on the satellite: p50 **232.8 ms → 48.0 ms** (4.85x), realtime
factor **1.45x → 0.30x**, process CPU **398% → 30%**, load average 8.5 → 2.5.
Scores are bit-identical to stateless `predict()` (max diff 0.000e+00 over 50
windows of real audio) — verified by `scripts/wake_bench.py`, which fails if they
ever diverge.

**Consequence for training: re-measure before retraining.** Every score ever
recorded from this satellite, including the 0.57-0.72 ceiling on genuine wakes
and the 126.8 hours of telemetry above, was produced under defect 2. The phrase
was never reliably centred in the window it was scored on. That ceiling may lift
on its own now. Retraining against numbers gathered from a misaligned pipeline
would bake the wrong operating point into the next model.
