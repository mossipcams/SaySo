Place this LiveKit-exported Sayso classifier on the satellite:

  /opt/sayso-satellite/models/sayso.onnx

It detects the spoken phrase "Sayso" only. Operating point is in `sayso_eval.json`.
Use threshold **0.5** (`fpph` 0.0, recall 0.60). Do not use the file's
`optimal_threshold` of 0.19 in a room with background speech: it carries a
documented `optimal_fpph` of 0.25 false wakes/hour (~6/day), and measured
living-room scores put real wakes at 0.57-0.72 with every false positive
below 0.42. Do not substitute hey_livekit, hey_jarvis, or another model.

## Active training recipe (LiveKit)

`sayso-training.yaml` is the **only active** wake training recipe. It targets
`livekit-wakeword` 0.2.1 with `target_fp_per_hour: 0.02`, the `/seI soU/`
confusable set as hard negatives, and a deliberately small corpus. The shipped
model was trained on upstream defaults (`target_fp_per_hour: 0.2`); it could not
meet even that, so `find_best_threshold()` fell through to max-balanced-accuracy
and emitted `optimal_threshold` 0.19.

Do not run training on the Pi. `setup` downloads ~16 GB of ACAV100M features plus
MUSAN (~1.1 GB) and RIRs. Use a CUDA host.

Install host-only deps from `requirements-wake-train.txt` (not the satellite
runtime requirements):

```bash
pip install -r satellite/models/requirements-wake-train.txt
```

Prefer the batch wrapper (snapshots, run-key idempotency, candidate bundles):

```bash
python3 scripts/wake_train.py \
  --spool /var/lib/sayso-satellite/wake-mining \
  --seed-dir satellite/models/data/seed \
  --work-dir satellite/models/output/wake_runs
```

Manual LiveKit stages remain available for debugging:

```bash
pip install "livekit-wakeword[training]==0.2.1"
python -m livekit.wakeword setup    --config satellite/models/sayso-training.yaml
python -m livekit.wakeword generate satellite/models/sayso-training.yaml
python -m livekit.wakeword augment  satellite/models/sayso-training.yaml
python -m livekit.wakeword train    satellite/models/sayso-training.yaml
python -m livekit.wakeword export   satellite/models/sayso-training.yaml
python -m livekit.wakeword eval     satellite/models/sayso-training.yaml
```

Ship `output/sayso/sayso.onnx` and its metrics JSON back into this directory,
then set `wake_word.threshold` from calibration data (not by reusing 0.5).
Freeze both the deployed threshold and the calibration-selected threshold in
`satellite/eval/baseline.json` before retraining.

Source splits and holdout rules live in `satellite/eval/splits.json`. Holdouts
must not leak into training, calibration, background mixing, or derived features.

## NanoWakeWord prototype (archived)

`sayso-nanowakeword.yaml` is an **optional prototype** only. The satellite
defaults to LiveKit (`wake_word.provider: livekit`). NanoWakeWord uses a
different feature frontend and score distribution — do not treat its output as
production-equivalent.

```bash
pip install "nanowakeword[train]"
nanowakeword -c satellite/models/sayso-nanowakeword.yaml -G -t -T
```

Generated `data/`, `output/`, Piper artifacts, and `.wav`/`.npy` features are
gitignored. The LiveKit `sayso.onnx` is not a NanoWakeWord model.

## Bootstrap blockers

- `../eval/audio/` has no trusted fixtures. `cases.json` skips every case in
  default mode and fails in `--strict`. Populate with real Blue Snowball
  recordings from the living room before baseline freeze, qualification, or
  enabling `--schedule` on `scripts/wake_train.py`.
- Hard negatives in `sayso-training.yaml` remain phonetic inference until the
  mining spool supplies labelled real confusions. Run
  `python scripts/wake_mine_report.py <mine_dir> --inventory` to audit wake
  assets before cleanup.
