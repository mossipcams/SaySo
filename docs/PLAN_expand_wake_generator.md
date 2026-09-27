# Plan: run two-pass LiveKit wake experiment

## Scope

- Add a babysitter script that runs generation, augmentation/features, training, and internal evaluation for two passes.
- Pass one resumes the isolated 25k/5k, three-round dataset expansion already underway. Use its LiveKit validation metrics to broaden pass-two TTS settings; add 5k samples if recall is below 0.90 or FPPH exceeds target, and add mined near-miss phrases if FPPH exceeds target. Never tune from the locked promotion evaluation.
- Seed pass two with the 60 eMeet positives and 162 mined LibriSpeech negatives, then generate and train it independently.
- Save both `.pt` and `.onnx` models with their recipes and reports. Run strict promotion evaluation for both; fail closed while the recorded baseline is blocked.

## Files to touch

- `docs/PLAN_expand_wake_generator.md`
- `docs/HANDOFF_WAKE_TRAINING_MIX.md`
- `scripts/wake_livekit_experiment.py`
- On `llm`: the active isolated pass-one snapshot, an independently generated pass-two snapshot, and an archive directory containing both finished models and reports.

## Verification

- Confirm the original snapshot and locked evaluation fixtures are unchanged.
- Confirm both generated datasets retain eMeet and LibriSpeech sources, complete 25k/5k positive and negative splits, and have finite feature arrays from three augmentation rounds.
- Confirm both model archives include hashes, recipe, LiveKit metrics, and promotion-only strict evaluation report at each candidate's LiveKit `optimal_threshold`; promotion remains blocked until the independent baseline/calibration gate is satisfied.
- Run the script's CLI help/syntax check and record paths, counts, metrics, and promotion decisions in the handoff.
