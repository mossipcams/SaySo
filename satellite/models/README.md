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

## NanoWakeWord (opt-in)

The satellite can load a NanoWakeWord ONNX model instead of the LiveKit
classifier when `wake_word.provider` is set to `nanowakeword`. LiveKit remains
the default production path; do not flip a live satellite to Nano without an
explicit operator decision.

Shipped model: `sayso-nanowakeword.onnx` (md5 `83d9a507e35530adffaf55465a3b1478`).
Operating point: threshold **0.50**. Living-room hop-feed promotion scores
(1 s silence pad each side, fire if max ≥ 0.50):

| Set | ACAV `83d9a507` | official `32eaa92e` | 80-pos `2fe297e0` |
| --- | ---: | ---: | ---: |
| SaySo (`live_sayso_*`) | **8/8** | 8/8 | 6/8 |
| Isolated talk | **0/8** | 0/8 | 0/8 |
| Overlapping talk (89) | **0/89** | 2/89 | 50/89 |
| Miner party (74) | **0/74** | 1/74 | 42/74 |

`sayso-nanowakeword.yaml` is the promoted ACAV recipe: AE29H + RACON + OpenWakeWord
ACAV100M bulk negatives (`oww` batch **1000**), no `-G`, no `from_list` “say so”
clone negatives, `generate_clips: false`. Do not run training on the Pi.

```
pip install "nanowakeword[train]"
nanowakeword -c satellite/models/sayso-nanowakeword.yaml -t -T --overwrite
```

Copy the exported ONNX to `/opt/sayso-satellite/models/sayso-nanowakeword.onnx`
and set `wake_word.provider: nanowakeword` only when switching off LiveKit.
Generated `data/`, `output/`, Piper artifacts, and `.wav`/`.npy` features are
gitignored — do not commit them. The LiveKit `sayso.onnx` is not a NanoWakeWord
model.

Compare LiveKit vs Nano on recorded clips:

```
python3 -m satellite.eval.compare_providers \
  --audio-dir DIR --livekit models/sayso.onnx --nano models/sayso-nanowakeword.onnx
```
