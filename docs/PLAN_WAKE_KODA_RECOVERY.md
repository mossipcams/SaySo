# Koda wake word: round 4 recovery

Koda runs live on the llm VM under `/srv/llm/wake/runs/koda-*` with output in
`/home/LLM/livekit-wakeword-koda/output/koda-*`. Nothing Koda-specific is in
this repo besides this file.

## What broke (2026-10-05)

`koda-round4-20261005` set `noise_scale_ws: [3.0, 2.0, 1.0]`, `noise_scales:
[0.5, 0.75, 1.0]` and dropped `length_scales`. Stock LiveKit and the runs that
generated cleanly (`koda-fp-corrected`, `koda-massive-livekit`) use
`noise_scales: [0.98]`, `noise_scale_ws: [0.98]`, `length_scales: [0.75, 1.0, 1.25]`.
VITS duration noise of 3.0 makes some clips run away (a 691 s clip, a 181 GB CPU
allocation, 53 HIP OOMs), so batches were skipped and pools fell below quota
(gate 1: negative_train 1,650/5,000).

Two recovery runs chased the symptom (CPU masking, batch cap 1, AMD override) and
were dead ends and have been deleted from the VM
(`koda-round4-recovery-20261005`, `koda-round4-amd-20261005`). The original
failed `koda-round4-20261005` stays as the evidence for the cause above.

## Fix

`koda-round4b-20261005`: same script, quotas, phrases and gates as round 4;
only the synthesis noise/length settings are back to the working values and
the paths are fresh. No overrides, no batch cap.

## Rule

Do not change `noise_scale_ws` above ~1.0 with the Piper VITS backend.
Result of round 4b: see `pipeline.log` in its run dir.
