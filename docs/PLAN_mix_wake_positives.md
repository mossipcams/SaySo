# Plan: mix new wake positives into training data

## Scope

- Use the 60 SHA-256-verified takes in `/srv/llm/data/wake/data/sayso-manual-emeet-20260925/`.
- Preserve completed run `/srv/llm/wake/runs/livekit-restart-20260923/` unchanged.
- Create a separate dataset snapshot from its LiveKit `output/sayso` training data. Convert each 3-second take into a 2-second training window (its quiet final 0.5 seconds, then its first 1.5 seconds), add the 60 windows to `positive_train`, augment them, and append their extracted features while reusing unchanged split features.
- Do not alter validation/evaluation sets or run a new model training job.

## Files/data to touch

- This plan file.
- New remote snapshot under `/srv/llm/wake/runs/livekit-restart-emeet-20260925/`; source set remains read-only.

## Verification

- Confirm exactly 60 source WAVs and verify copied SHA-256 hashes against the source files.
- Confirm each derived window is 2 seconds and retains speech; confirm the new snapshot contains all original training positives plus 60 eMeet clips, with no additions to `positive_test`.
- Verify the positive training feature array contains the 20,000 existing examples plus 120 new augmentations; verify all other split features match the source run.
