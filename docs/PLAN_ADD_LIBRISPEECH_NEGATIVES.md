# Plan: add mined LibriSpeech negatives

## Scope

- Add all 77 `false_positive` and 85 `hard_negative` mined 2-second windows, with two training augmentations each, to the eMeet snapshot's negative training data.
- Preserve source audio and sidecars; do not change validation/evaluation data, model artifacts, or run training.

## Files to touch

- `docs/PLAN_ADD_LIBRISPEECH_NEGATIVES.md`
- `docs/HANDOFF_WAKE_TRAINING_MIX.md`
- On `llm`: `/srv/llm/wake/runs/livekit-restart-emeet-20260925/output/sayso/negative_train/`, `negative_features_train.npy`, and a provenance manifest alongside the output.

## Verification

- Confirm all 162 source records have the expected categories, valid 2-second WAVs, and unique hashes.
- Check for collisions with existing training WAVs and verify target artifacts are safe to update despite inherited hardlinks.
- Create each source clip plus `_r0` and `_r1` variants using the pinned LiveKit augmentor; extract and append 324 feature rows with its pinned frontend, confirming all added values are finite and nonzero.
- Confirm validation/evaluation arrays and model artifact hashes are unchanged; inspect final counts and manifest hashes.
- Update the handoff with the new training counts and the LibriSpeech evaluation caveat.
