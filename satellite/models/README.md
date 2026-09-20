Place this LiveKit-exported Sayso classifier on the satellite:

  /opt/sayso-satellite/models/sayso.onnx

It detects the spoken phrase "Sayso" only. Operating point is in
`sayso_eval.json` and `living2.yaml`. Use threshold **0.5** and the mel
verifier at **0.445**. Do not use the trainer `optimal_threshold` (~0.05).
Do not substitute hey_livekit, hey_jarvis, or another model.

Shipped primary: living2 (`b840f51f312abcd5b205e1fc1e32b2ed`) plus
`sayso-verifier.npz` (`0c632e778ca263e51c92d9ca95f451af`).

## Retraining (living2)

`living2.yaml` is the production recipe. Skip `livekit.wakeword generate`.
The mix is 50 this-room Snowball positives, 90 this-room train negatives,
and 89 overlapping-talk clips as **val only**. Holdout, miner party, and
`nano_live_fp` stay out of the classifier.

living1 used the same wavs with `positive: 16` / `ACAV100M_sample: 256` /
`max_negative_weight: 3000` and never fired at 0.50. living2 only changes
the class prior (`positive: 96`, `ACAV100M_sample: 64`,
`max_negative_weight: 200`) so 0.50 can fire. Mixing extra TTS positives
(blend) or more ACAV (living3) failed the this-room gates — do not repeat
those.

Do not run this on the Pi. Use a CUDA host. Do not set
`CUDA_VISIBLE_DEVICES` to empty.

```
pip install "livekit-wakeword[training]"
python -m livekit.wakeword augment living2.yaml
python -m livekit.wakeword train    living2.yaml
python -m livekit.wakeword export   living2.yaml
```

Ship `output-living2/sayso/sayso.onnx` back into this directory. Score at
**0.50**, then AND with the verifier below. `sayso-training.yaml` is the
older voxcpm/hard-negative recipe; it is not this operating point.

## Mel verifier (second stage)

Frozen Google speech embeddings separate this-mic SaySo from overlapping
talk (AUROC 0.97) but not from the 19 live false wakes (a probe fit on the
89 still calls 13/19 SaySo; **mel AUROC 1.0**). Do not dump those FPs into
the classifier.

`sayso-verifier.npz` is a logistic on frozen-mel mean+std of the last-16
embedding mel union, fit on the 50 recorded SaySo vs the 19 `nano_live_fp`
windows only. Miner 74 and the 89 stay out of that fit.

When `wake_word.verifier` is set, LiveKit remains the primary scorer and
the verifier must also pass before the satellite fires. Mine on the LiveKit
score as today — the veto runs after mining, not before.

Fire iff `living2 ≥ 0.50` **and** `verifier ≥ 0.445`. Omit `verifier` for
single-stage LiveKit. If `verifier` is set but the file is missing, wake
detection fails closed.

Host AND-gate (hop-scan at 0.50 / 0.445):

| Set | Result |
| --- | ---: |
| living-room SaySo | **6/8** |
| isolated talk | **0/8** |
| overlapping talk (89) | **0/89** |
| miner party (74) | **0/74** |
| nano_live_fp (19) | **0/19** |

The 19 are verifier-train, not an unbiased FP set. Best unbiased-FP backup
on the Pi is `sayso.onnx.bak-03e612d8`.
