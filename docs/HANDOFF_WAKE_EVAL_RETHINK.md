# Handoff: rebuild wake-word evaluation around retained audio

**Updated:** 2026-09-25  
**Status:** candidate retained for analysis; promotion blocked

## Current VM run

The run at `/srv/llm/wake/runs/livekit-restart-20260923/` completed setup,
generation, augmentation, training, export, and LiveKit validation. Its ONNX
candidate SHA-256 is
`33b77a3c4bb3a50df304653b9859b24ca6de745a7d1bca1545d1b09860e7c9b3`.
The latest LiveKit validation used generated validation audio: 4,000 positives
and 35,084 negatives, with 75.0% recall and 0.00 reported FPPH at threshold
0.50. This is trainer feedback, not room qualification.

An exploratory score over the run's 50-positive/38-negative `real-holdout`
files found 0/50 candidate positives and 0/38 negatives at threshold 0.28.
Living2 scored 14/50 positives and 0/38 negatives at the same threshold. The
split was randomized by WAV file in `split_real.py`; source/session lineage was
not grouped or verified. Treat this as a warning signal, not independent
qualification evidence. The report is
`/srv/llm/wake/runs/livekit-restart-20260923/real-holdout-eval.json`.

The VM run status is recorded in
`/srv/llm/wake/runs/livekit-restart-20260923/status.json` as complete with
`promotion_state: blocked`. The candidate was not promoted; the shipped
living2 model remains in place. GPU serving was restored after export and
evaluation.

## Audio inventory found on `llm`

The repo's `satellite/eval/audio/` fixtures are absent and
`satellite/eval/baseline.json` remains blocked. The VM still has candidate
source material under `/srv/llm/data/wake/live-20260924/`:

| Path | WAV count | Notes |
| --- | ---: | --- |
| `recordings/` | 402 | Includes real SaySo recordings and derived files; map originals to variants before splitting. |
| `tv-fp-corpus-hn-v1-20260922/` | 99 | TV false-positive hard negatives. |
| `tv-fp-emeet-20260922/` | 45 | eMeet false-positive hard negatives. |
| `wake-mining-archive-labelled/` | 104 | Labeled mined windows; verify labels and capture/session lineage. |
| `wake-mining/` | 2,470 | Mined WAVs; audit labels, outcomes, duplicates, and lineage. |
| `stt_capture/` | 518 | Command audio; inspect correlation and wake coverage before any eval use. |

These counts establish that audio remains on the VM; they do not establish that
it is independent, correctly labeled, or suitable for a locked eval. Verify
backups and preserve immutable copies before cleanup or retraining.

## Split audit (2026-09-25)

Replaying `split_real.py` selection with its seed reproduces the split
exactly: all 50/38 holdout files and all 203/156 injected `clip_9*.wav` train
files match by SHA-256. The holdout is not independent:

- 79/88 holdout items share a capture group with training data, and 88/88
  share a session. There are no exact duplicate files, but related audio
  appears in both train and holdout.
- TV-FP captures are `pre.wav`/`window.wav`/`post.wav` triples, and the glob
  split them file by file (holdout 13 window, 6 pre, 10 post). The 143
  negatives come from 3 sessions, 95 of them from one.
- All 200 recorded positives come from a single session (`positive_sayso_20260921`).
  35 are in holdout and 165 in train. All 53 mined positives come from
  2026-09-13, with several windows per captured second.
- The candidate still scored 0/50 on holdout positives that are near
  neighbours of its training data. That makes the failure more serious.

Labelled audio the candidate never trained on:

- `wake-mining/*.json` (09-19): 3 positive, 32 negative, 1 unsure.
- `wake-mining/records`: 20 negatives. There are 747 records across 21
  sessions, but 727 are unlabelled and 48 overlap the TV-FP captures.
- `stt_capture`: 438 command clips plus 80 failures. These are command audio,
  not wake labels.

Conclusion: no valid source-group holdout for positives can be formed from
surviving audio. New recording sessions are required, meaning different days
and ideally different speakers/positions, collected before any training
touches them. Negatives could be built from untouched sessions after operator
labelling. Label candidates are the 20 labelled records plus unlabelled
`wake-mining/records` sessions not present in TV-FP.

## LibriSpeech false-activation mining (started 2026-09-25)

`scripts/wake_librispeech_mine.py` replays LibriSpeech `train-other-500`
(OpenSLR tarball, md5-checked; same audio as HF
`openslr/librispeech_asr` `other`/`train.500`) through the production path
with the deployed model: `sayso.onnx` `7f9a84e9…`, threshold 0.25, 2 s
refractory, 160 ms hop. Chapters are concatenated in transcript order, so
replay is continuous within each chapter. Each activation is saved as
`window.wav` + `context.wav` (±2.5 s) + `record.json`. Activations are
bucketed by the transcripts of the utterances under the window:

- `excluded_say_so`: the transcript contains "say so". These are kept, but
  their utterance time is removed from the FPPH denominator.
- `hard_negative`: the transcript contains say/so/said/says/same/save/safe/
  say some/say something. These are tagged at utterance level, since
  LibriSpeech has no word alignments.
- `false_positive`: everything else.

Hard negatives still count as false activations. VM layout:
`/srv/llm/data/wake/librispeech-mine/`, containing the pinned model, the code
copy, `run/` (resumable; `chapters/` holds per-chapter summaries and
`summary.json` the totals), and `run-full.log`. Smoke run (12 chapters,
2.2 h): 1 activation, which re-scored identically with plain
`WakeWordModel.predict`. The FPPH is a fair out-of-domain figure for the
deployed model, which never saw LibriSpeech. Once mined clips feed training,
a later model's FPPH on this corpus is no longer unbiased. Hold out a
speaker-disjoint slice, or use a different corpus, for later models.

## Prompted eMeet takes (2026-09-25)

- 60 training takes: `llm:/srv/llm/data/wake/data/sayso-manual-emeet-20260925/`
  (sha256-verified).
- `silent30` eval takes: `llm:/srv/llm/data/wake/eval/silent30/` (read-only,
  sha256-verified). They are cases `silent30_001..030` in
  `satellite/eval/cases.json` and holdout group `silent30` in `splits.json`
  (smoke only, because they share speaker/room/mic/session with the 60).
- The as-recorded clips start after the prompt beep, so the phrase sits in the
  first second and no 2 s window has lead-in. Scored raw, the deployed model
  got 0/30, an artifact of the format. Cases therefore read
  `silent30_leadin/`, which prepends 1.5 s of real session room tone (see
  its `derivation.json`; masters untouched).
- Strict eval of the deployed model (`7f9a84…`, threshold 0.25, production
  hop/refractory) detects 12/30 silent30 takes. Under current rules every
  failed case rejects a candidate, so promotion now needs 30/30 on silent30.
  Decide whether that bar should become a recall floor.

## Evaluation problems to fix

1. The canonical strict evaluator cannot qualify a candidate while its recorded
   cases are missing. Keep it fail-closed.
2. `scripts/wake_train.py` silently substitutes tiny synthetic fixtures when
   `cases.json` is absent. Its feasibility gate prevents qualification, but the
   fallback can make an eval report look meaningful. Missing real eval data
   should be an explicit blocked result.
3. The VM experiment split by file rather than source/session group. Related
   utterances and derived variants must stay together. Unknown lineage cannot
   prove independence.
4. Per-file maximum-score comparisons are useful diagnostics, but do not
   replace continuous replay through production hop, lag, cooldown, and wake to
   command handoff behavior.
5. Thresholds must be chosen on calibration data and frozen before locked eval.
   Never choose a threshold on the final holdout.

## Replacement workflow

1. Inventory and hash surviving audio, sidecars, labels, and parent/variant
   relationships. Restore an off-VM backup before changing the only copy.
2. Build a manifest over source recordings and sessions. Record source group,
   label evidence, channel/device, timestamps, audio hash, and derived-window
   ranges. Keep audio outside Git; track the manifest and evaluation definitions
   in Git.
3. Split source groups into train, calibration, and locked evaluation before
   making windows or augmentations. If provenance cannot support disjoint
   groups, collect more labeled sessions instead of calling a file split
   independent.
4. Evaluate continuous sessions through the production replay path. Report
   positive wake recall by condition, false activations per representative
   background hour, activation timing, and command first-word retention.
   Preserve natural “say so” as an ambiguity challenge where the classifier
   window is acoustically identical; do not score intent as acoustically
   separable.
5. Run timing and handoff checks on the target satellite. Keep promotion blocked
   until the locked source-group holdout passes and background exposure supports
   the false-activation bound. The plan's 0.02/hour target requires about 150
   representative independent zero-event hours for a one-sided 95% Poisson
   upper bound. Historical telemetry from before the audio-window alignment fix
   does not count toward that exposure.
6. Save immutable model, data, recipe, threshold, evaluator, and report hashes.
   Back up the corpus before training and retain the previous deployed model for
   rollback.

## Next actions

- Done: audit and split reconstruction (see Split audit). Result: new positive
  recording sessions are required.
- Record new positive sessions for calibration and locked eval, and keep them
  off the training host paths until the locked set is frozen.
- Done: non-stub `run_pipeline` now returns non-final `blocked` when
  `cases.json` is missing; synthetic fixtures are used only in stub mode
  (`test_real_mode_missing_eval_manifest_blocks`).
- Do not promote the current candidate from its current evidence.
