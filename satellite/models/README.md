Place this LiveKit-exported Sayso classifier on the satellite:

  /opt/sayso-satellite/models/sayso.onnx

It detects the spoken phrase "Sayso" only. Operating point is in `sayso_eval.json`.
Use threshold **0.5** (`fpph` 0.0, recall 0.60). Do not use the file's
`optimal_threshold` of 0.19 in a room with background speech: it carries a
documented `optimal_fpph` of 0.25 false wakes/hour (~6/day), and measured
living-room scores put real wakes at 0.57-0.72 with every false positive
below 0.42. Do not substitute hey_livekit, hey_jarvis, or another model.

## Retraining

`sayso-training.yaml` in this directory is the aggressive retrain config. It
exists because the shipped model was trained on `livekit-wakeword` defaults,
including `target_fp_per_hour: 0.2`. The model could not meet even that target,
so `find_best_threshold()` fell through to its max-balanced-accuracy fallback
and emitted `optimal_threshold` 0.19 — which is why the deployed satellite was
firing on ambient conversation. The config drops that target to 0.02, adds the
"say so" confusable set as hard negatives, and moves small → medium.

Do not run this on the Pi. `setup` downloads ~16 GB of ACAV100M features plus
MUSAN (~1.1 GB) and RIRs, and 120k steps on the satellite's 4-core ARM is days.
Use a CUDA host.

```
pip install "livekit-wakeword[training]"
python -m livekit.wakeword setup    --config sayso-training.yaml
python -m livekit.wakeword generate sayso-training.yaml
python -m livekit.wakeword augment  sayso-training.yaml
python -m livekit.wakeword train    sayso-training.yaml
python -m livekit.wakeword export   sayso-training.yaml
python -m livekit.wakeword eval     sayso-training.yaml
```

Ship `output/sayso/sayso.onnx` and its metrics JSON back into this directory,
then set `wake_word.threshold` in `/etc/sayso-satellite/config.yaml` from the
new operating point. Read the threshold off the metrics rather than reusing 0.5;
0.5 is calibrated to the *current* model's score distribution and means nothing
for a retrained one.

Two gaps worth closing before trusting a retrain:

- `../eval/audio/` is empty (`.gitkeep` only), so `cases.json` — including its
  `negative_natural_say_so` and `negative_tv_conversation` cases — skips every
  case. There is no recorded-audio regression test for this model. Populate it
  with real Blue Snowball recordings from the living room.
- The satellite never retains the audio that fired a detection, and
  `flush_preroll` skips `wake_skip_ms` before STT, so false-positive transcripts
  show what was said *after* the trigger, not the trigger itself. The hard
  negatives in the training config are therefore phonetic inference, not
  measured. A debug mode that saves the 2 s window on detection would turn the
  next round of negatives into real data.
