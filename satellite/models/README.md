What ships on the satellite. Training (LiveKit, host, data, recipes):
[training/wake/README.md](../../training/wake/README.md). Design:
[docs/SAYSO_WAKE_WORD_TRAINING_PLAN.md](../../docs/SAYSO_WAKE_WORD_TRAINING_PLAN.md).

Place this LiveKit-exported Sayso classifier on the satellite:

  /opt/sayso-satellite/models/sayso.onnx

It detects the spoken phrase "Sayso" only. The generate-first model runs
**single-stage LiveKit** by default — omit `wake_word.verifier`. Do not use the
trainer `optimal_threshold` (~0.05). Do not substitute hey_livekit, hey_jarvis,
or another model.

Shipped on the Pi today: living2 (`b840f51f312abcd5b205e1fc1e32b2ed`) at
threshold **0.28** with legacy `sayso-verifier.npz`
(`0c632e778ca263e51c92d9ca95f451af`). That mel artifact is **not** a phrase
check for the generate-first export; the satellite ignores it and stays
single-stage when it is configured.

## Optional embedding verifier (second stage)

The generate-first model already classifies from (16, 96) speech embeddings.
A compatible second stage must score those same embeddings — not a parallel
collapsed-mel pass.

Legacy `sayso-verifier.npz` (`feature_kind` absent, treated as `mel_union`) is
a this-room recording-envelope logistic on mel mean+std, fit on 50 Snowball
clips vs 19 `verifier_live_fp` windows. It is **not** a phrase check for the
generate-first export and does **not** AND-gate LiveKit when configured; the
satellite logs once and runs single-stage.

A future compatible artifact sets `feature_kind=speech_embedding` in the npz and
carries logistic weights for concat(mean, std) over the last 16 speech
embeddings (192-d). None is shipped yet — do not invent one in this tree.

When `wake_word.verifier` points at a compatible npz, LiveKit remains the
primary scorer and the embedding verifier must also pass before the satellite
fires. Mine on the LiveKit score as today — the veto runs after mining, not
before. Omit `verifier` for single-stage LiveKit (generate-first default). If
`verifier` is set but the file is missing, wake detection fails closed.

### living2 + mel verifier (historical Pi operating point)

Frozen Google speech embeddings separate this-mic SaySo from overlapping talk
(AUROC 0.97) but not from the 19 live false wakes. The mel logistic fit on
50 vs 19 achieved mel AUROC 1.0 on that narrow set. Do not dump those FPs
into the classifier.

Host AND-gate (hop-scan at living2 **0.50** / mel verifier **0.445**):

| Set | Result |
| --- | ---: |
| living-room SaySo | **6/8** |
| isolated talk | **0/8** |
| overlapping talk (89) | **0/89** |
| miner party (74) | **0/74** |
| verifier_live_fp (19) | **0/19** |

Pi operating point (living2 **0.28** / mel verifier **0.445**):

| Set | Result |
| --- | ---: |
| living-room SaySo | **7/8** |

Miss on Pi: `live_sayso_05` = 0.179 collides with `live_talk_02` = 0.178
(verifier blesses both). 8/8 is blocked by that collision.

The 19 are verifier-train, not an unbiased FP set. Best unbiased-FP backup
on the Pi is `sayso.onnx.bak-03e612d8`.
