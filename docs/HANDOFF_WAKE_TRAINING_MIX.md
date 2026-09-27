# Handoff: wake training snapshot

**Updated:** 2026-09-26 15:29 UTC

## Current state

- LibriSpeech mining is complete: 2,784 chapters, 496.858 audio hours (496.411 scored), 77 false positives, 85 hard negatives, and 8 excluded “say so” activations. The reported rate is 0.3263 false activations/hour. Hard negatives count as false activations. Totals: `llm:/srv/llm/data/wake/librispeech-mine/run/summary.json`; log: `run-full.log` in the same directory.
- The 60 eMeet positives are in a separate training snapshot: `llm:/srv/llm/wake/runs/livekit-restart-emeet-20260925/`. Source WAVs remain at `llm:/srv/llm/data/wake/data/sayso-manual-emeet-20260925/`; source and derived hashes are in `emeet-manifest.json`.
- The source takes are 3-second, 16 kHz mono PCM. Each training copy is a 2-second window made from its quiet final 0.5 seconds followed by its first 1.5 seconds. All 60 passed the speech/tail level checks. Two augmentations per take were added.
- Snapshot `positive_train` has 30,180 WAVs. `positive_features_train.npy` has shape `(20120, 16, 96)`: the original 20,000 rows plus 120 augmented eMeet rows. The original rows are unchanged. Validation/evaluation feature arrays and `positive_test` remain unchanged.
- All 77 LibriSpeech `false_positive` and 85 `hard_negative` windows were added to `negative_train`, with two augmented WAVs per window: 486 WAVs added. `negative_train` now has 30,486 WAVs and `negative_features_train.npy` has shape `(20324, 16, 96)`, including 324 new rows. Provenance, source records, and hashes are in `output/sayso/librispeech_negative_manifest.json`.
- The separate two-pass LiveKit experiment below trained and evaluated two independent candidates. The mining snapshot above remains unchanged.
- Pass 2 is the user-designated champion **SaySo Voicev1**, active on the Pi at `/opt/sayso-satellite/models/sayso-voicev1.onnx`, threshold `0.22`. Its SHA-256 is `f1313f031ae5c05867f4d716c9e47a0c459e0c6b586400335cce5ae8e72d2199`; Pi eval detected 24/30 silent30 positives. `sayso-satellite.service` was verified active after switching config and restarting. The hardware report is archived as `pass2/strict_eval_pi_voicev1.json`; rollback copies remain in `/home/pi/sayso-voicev1-rollback-20260926/`.

## Two-pass LiveKit experiment

- Both finished model pairs (`.onnx` and `.pt`), recipes, metrics, internal evals, hashes, and strict eval reports are archived at `llm:/srv/llm/wake/runs/livekit-prodscale-emeet-20260926-models/{pass1,pass2}/`. `experiment.json` records the recipe changes and both promotion decisions.
- Pass one used 25k/5k generated train/validation splits. Its fresh LiveKit eval reached optimal recall `0.85080` at `0.0773` false positives/hour (threshold `0.24`). Since recall missed the `0.90` target, pass two increased the recipe to 30k/6k and added slower/faster TTS rates and more extreme speaker blends.
- Pass two includes all 60 eMeet positives and 162 mined LibriSpeech negatives. Its feature arrays have shapes `(90000, 16, 96)` positive train, `(18000, 16, 96)` positive validation, `(90000, 16, 96)` negative train, `(18000, 16, 96)` negative validation, plus `(6000, 16, 96)` and `(1500, 16, 96)` background train/validation. LiveKit eval reached optimal recall `0.85089` at `0.0726` false positives/hour (threshold `0.22`). The near-miss phrase expansion was not triggered because FPPH stayed below the configured `0.10` target.
- The 30 `silent30` clips are the current positive evaluation set. At threshold `0.5`, pass one detected 0/30 and pass two 3/30; at their LiveKit validation thresholds (`0.24` / `0.22`), they detected 8/30 and 24/30. The five unavailable diagnostic fixtures are marked `promotion_required: false` and omitted by `--promotion-only`; full diagnostics still report them as missing. Candidate reports use each LiveKit `optimal_threshold`. Pass two still fails the 30-case strict gate, and the baseline remains blocked pending calibration plus independent negative/background recordings. The set shares speaker/session lineage with training. Reports are archived as `strict_eval.json`, with the previous all-case reports at threshold `0.5` preserved as `strict_eval_full_at_deployed_threshold.json`. Voicev1 is deployed by the user's champion selection; automated promotion remains blocked.

## Before training

The snapshot was made with hardlinks to the completed run at
`/srv/llm/wake/runs/livekit-restart-20260923/`. Unchanged WAVs and inherited
model artifacts still share inodes with that run, including `sayso.onnx`,
`sayso.pt`, and `sayso_metrics.json`. Do not overwrite or edit these linked
files in place. Break the links or make a fully independent work copy before
running stages that write model artifacts or WAVs.

The inherited candidate is not qualified for deployment. Its prior recorded
holdout was split by WAV rather than source/session lineage; see
`docs/HANDOFF_WAKE_EVAL_RETHINK.md`. The `silent30` takes are the current
positive evaluation set and share a recording session with the 60 eMeet
training takes; preserve that lineage caveat when interpreting results. Because
this snapshot includes all 162 LibriSpeech false-
positive and hard-negative windows, its false-activation rate is not an
independent evaluation measure for a model trained from this snapshot.

## Verification performed

- Confirmed all 60 source hashes against the snapshot manifest.
- Confirmed 60 derived 2-second windows, 120 augmented clips, and 20,120 positive training feature rows; all added feature values are finite and nonzero.
- Confirmed all 162 LibriSpeech windows, 486 added training WAVs, and 20,324 negative feature rows; added features are finite and nonzero, and source/derived hashes are recorded in the snapshot manifest.
- Confirmed existing positive feature rows are byte-equivalent in value; validation/evaluation feature arrays match the completed run.
- Confirmed the final LibriSpeech summary and that no mining or model-training process remains active.
