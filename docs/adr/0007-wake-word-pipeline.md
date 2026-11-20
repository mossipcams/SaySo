# 0007. Wake word pipeline

## Status

Accepted. The live Pi test of the 10u ensemble and the quiet-room misses are still open; see Consequences.

## Context

The Koda wake model runs on the Pi satellite (`satellite/sayso/wake/`) over 16 speech embeddings per window, at threshold 0.65 and an 80 ms hop. Training runs on the `llm` VM. The repo holds the scripts and `training/wake/README.md`, not the model weights.

Three problems showed up in 2026-11:

- **Wakes failed with the TV on.** The deployed v9b model scored seven user-confirmed Kodas over the TV at 0.14 to 0.97, and two fired. The real TV is loud, at about -22 dBFS RMS, only 11 to 14 dB under the voice. Every other real Koda in the corpus was said in near silence.
- **The satellite dropped wake windows.** With the TV on, 14 to 19% of windows were never scored. A dropped window that would have passed 0.65 became a missed or late wake, and a late fire clipped the start of the command.
- **Synthetic data taught the wrong thing.** TTS Koda mixed over TV made "TV means Koda" and kept 1 to 41 false wakes per hour on holdout TV.

Quiet-room results were at a ceiling of about 10 of 16. Whisper heard no "Koda" in five of the six records every model missed.

## Decision

- **Scoring path.** Windows go through a bounded FIFO of 25 window copies (2 s of hops). It drops the oldest window and counts it, with a rate-limited warning. It is cleared on suspend, resume and re-arm, using a sample-index boundary, so a window from before a boundary cannot fire. The speech-embedding session uses 2 intra-op threads. The mel and classifier sessions stay single-threaded. The model, threshold, hop, and refractory period are unchanged.
- **Ensemble.** The shipped model averages the members' pre-sigmoid logits, then applies sigmoid. It is exported as one ONNX with the same interface, `embeddings [b,16,96] -> score [b,1]`. Probability averaging was tried and did not work.
- **Training domain.** Real room Koda recordings are mixed over training TV at -30 to -16 dBFS (v10k). Hard negatives are mined from the training TV hours (v10u, 1,264 rows). Distant TTS speech over TV, chosen to be more than two phone edits from K OW D AH (v10t), supplies the synthetic TV rows.
- **Hop.** The hop stays at 80 ms. Hop-phase misses are fixed in training. A 40 ms hop was measured and rejected as not best practice. On the holdout TV, its top score was 0.62, and at 0.55 it fired about once an hour, so a finer hop does not buy a lower threshold.

Rejected:

- Synthetic Koda over TV (v10a, b, e, g, h): 1 to 41 false wakes per hour.
- Prefix negatives (v10r): stopped prefix fires but cost about 10% synthetic recall. A wash in the ensemble. Not shipped.
- Near-miss negatives over TV (v10s): a frozen-embedding probe separated TTS Koda from near-misses over TV at AUC 0.78 (0.96 clean), so the model rejected every TV positive.
- Shift-jittered real Kodas (v10v, v10w): better loud-TV recall, but the quiet-room "Hello." fires at 0.67. It stops firing only at 0.75, where recall falls back to the deployed level.
- Four- and five-member ensembles, and adding v9b. Adding v9b cost TV-on wakes.
- livekit-wakeword v2 on local Piper (v2a): strong recall, but 2 false wakes per hour on our TV. Its top holdout score was 0.83. It saw only 1.6 hours of generated data and none of our TV. Adding VoxCPM2 voices (v2b) was worse. The user-voice clones also speak hard negatives ("Kawda", "Kowda"), so user clones should stay out of negatives if v2 is retried.
- Speaker verification as a wake stage. The user ruled it out.
- Lowering the threshold below 0.65.

Measured on the 132-clip real voice set over holdout TV (bands -48..-38 / -38..-28 / -28..-18 dBFS), at threshold 0.65:

| | deployed 10t2+10k+10l | **10u+10k+10l** |
| --- | --- | --- |
| Real voice x TV | 108 / 100 / 65 | **117 / 104 / 70** |
| TV-on, expected of 7 | 4.25 | **4.56** |
| Quiet room, expected of 16 | 9.19 | **9.38** |
| Holdout TV FA/h at 0.35 / 0.45 / 0.55 / 0.65 | 2 / 1 / 0 / 0 | **0 / 0 / 0 / 0** |

For comparison, v9b (deployed before the ensemble) fired on 41 of the 132 real voice x TV clips and on 2 of 7 TV-on Kodas.

## Consequences

- **Deployed models.** On 2026-11-04 at 6:17 PM CT, the Pi ran the ensemble `koda-phonics-ens-10t2-10k-10l` (threshold 0.65). On 2026-11-05 at 5:21 AM CT it was replaced by `koda-phonics-ens-10u-10k-10l-20261105.onnx`, which is the current model and takes about 7 ms per window on the Pi. The deployment, its config backups on the Pi, and the ONNX files are not in git. Rollback is to point `wake_word.model` back at the 10t2 ensemble file and restart `sayso-satellite`.
- **Hop drops.** Real-time replay of 8.7 minutes of 35 real wake recordings, on the Pi, went from 16.5% of windows dropped to 0. Ideal wakes caught went from 33/37 to 37/37, and late fires from 16 to 0. The speech-embedding recompute went from about 230 ms to about 145 ms with 2 threads. Scores matched exact offline scoring (0.0000 difference in the replay). The cause was the single-slot queue and the full recompute. `WakeAudioBuffer.feed` was not the cause, since the 1024-sample mic block is smaller than the 1280-sample hop.
- **Handoff ring.** `wake_ring_capacity` in `satellite/sayso/wake/hook.py` sizes the ring with 5 s of headroom over the pre-roll and window. That covers 2 s of queue lag and the 220 ms `wake_skip_ms`.
- **Open.** The live Pi test of 10u (TV-on Kodas, quiet-room Kodas, and an hour of TV with nobody speaking) is still pending in the repo record. Quiet-room misses are tracked in issue #186. Collect about ten labelled misses across days before running another quiet-room experiment. The optional v10q ensemble eval never ran.
- **v2.** Livekit-wakeword v2 needs our TV hours as negatives before it can compete on false wakes.
- **Cost.** The 10t2 ensemble measured 6.0 ms per 80 ms hop on the Pi, against 2.1 ms for v9b. The 10u model takes about 7 ms per window.
- The shift-jittered and prefix-negative recipes stay in `scripts/`, so the next round can reuse them.

## Shipped in

- `8ddd013` fix(wake): score every wake window on the Pi satellite (merged as `903946b`, PR #187)
- `008b859` docs(wake): plan the TV-on logit ensemble (v9b + v10k + v10l)
- `213b7fe` feat(wake): add the TV-on logit ensemble builder, staging and real-TV eval
- `b1ebc9a` feat(wake): synthetic voices over TV for Koda TV-on wakes (v10t)
- `d7e4d06` feat(wake): livekit-wakeword v2 on local Piper + v10u/v10v/eval tools
- `3701c7c` docs(wake): round 3 results, deploy v10u ensemble
