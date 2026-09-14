Place this LiveKit-exported Sayso classifier on the satellite:

  /opt/sayso-satellite/models/sayso.onnx

It detects the spoken phrase `/seɪ soʊ/` — the fused "Sayso" or the two-word
"say so" — and only when nothing is spoken in front of it: a wake word preceded
by another word is not a wake. It does not detect "if you say so". Operating
point is in `sayso_eval.json`. Use threshold **0.27** (`optimal_fpph` 0.0,
`optimal_recall` 0.81). Do not substitute hey_livekit, hey_jarvis, or another
model.

The 0.27 is load-bearing and is **not** interchangeable with the 0.5 the
previous model shipped at. This model's score distribution is materially more
conservative — at 0.5 its recall is 0.54, *worse* than the 0.60 the old model
gave at the same threshold. Deploying the file without moving
`wake_word.threshold` to 0.27 is a regression, not a no-op.

Two caveats on that operating point, both worth knowing before trusting it:

- `optimal_fpph: 0.0` is measured over `validation_hours: 19.02`. One false
  positive in 19 hours reads as 0.053/hour, so this eval cannot resolve the
  config's `target_fp_per_hour: 0.02` — "0.0" here means "below 0.053", not
  zero. Confirming 0.02 needs 50+ hours of eval negatives.
- Recall 0.81 is against synthetic TTS positives. The only live evidence is a
  3-minute ambient soak on the satellite (156 windows, p50 0.0048, max 0.0464,
  zero fires), which shows it does not false-fire on an empty room — it does not
  show it wakes reliably on a real voice across a room.

## Retraining

`sayso-training.yaml` in this directory is the aggressive retrain config. It
exists because the shipped model was trained on `livekit-wakeword` defaults,
including `target_fp_per_hour: 0.2`. The model could not meet even that target,
so `find_best_threshold()` fell through to its max-balanced-accuracy fallback
and emitted `optimal_threshold` 0.19 — which is why the deployed satellite was
firing on ambient conversation. The config drops that target to 0.02 and teaches
the `/seɪ soʊ/` neighbourhood explicitly: both realizations as targets
(`target_phrases`), the carriers ("if you say so", "just say so") and the
measured near-misses ("stay so", "see you soon") as negatives. It keeps
`model_size: small` — the YAML comment explains why, and an earlier version of
this file wrongly claimed the config moves small → medium.

Do not run this on the Pi. `setup` downloads ~16 GB of ACAV100M features plus
MUSAN (~1.1 GB) and RIRs, and 120k steps on the satellite's 4-core ARM is days.
Use a CUDA host.

```
pip install "livekit-wakeword[train,voxcpm,export]"
python -m livekit.wakeword setup    --config sayso-training.yaml --skip-acav
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

Strip the exporter metadata before shipping the ONNX. Under torch >= 2.9
`torch.onnx.export` defaults to `dynamo=True`, which stamps stack traces, FX
node reprs and module class hierarchies onto every node — 73.7 KB of inert
debug metadata on a model whose weights are only 75 KB, near doubling the file.
onnxruntime ignores it. There is no export flag to suppress it, so post-process:

```python
import onnx
m = onnx.load("output/sayso/sayso.onnx")
for n in m.graph.node:
    del n.metadata_props[:]
    n.doc_string = ""
del m.graph.value_info[:]
onnx.save(m, "sayso.onnx")   # 162,319 B -> 81,978 B, outputs bitwise identical
```

The checked-in `sayso.onnx` has had this applied. Verify with
`sha256sum` that what you ship matches what you tested — strip first, then
checksum, or the two will disagree.

Known gaps before trusting a retrain:

- `../eval/audio/` is empty by design (gitignored: this repo is public and the
  corpus is real household speech), so `cases.json` — including its
  `negative_say_so_carrier` and `negative_tv_conversation` cases — skips every
  case on a clean checkout. Build a corpus with
  `scripts/wake_prepare_real_data.py eval-corpus` (see `../eval/README.md`);
  `wake-held-out-20260914.json` in this directory records the clusters it
  reserves so training never sees them.
- `flush_preroll` hands STT `[detection_index - wake_skip_ms, end)` with
  `wake_skip_ms` only a 120 ms margin, so a false-positive transcript shows
  what was said *after* the trigger, not the trigger itself. `wake_word.mine_dir`
  is now set, so the exact 2 s window is retained; the hard negatives in the
  training config predate it and are still synthetic.

## Retrain runbook

### What the pipeline ignores

Three constraints, all read off the installed `livekit-wakeword`. Real audio
that ignores any of them is trained on zero times, silently:

1. Feature extraction reads **only** `clip_\d{6}_r\d+.wav`
   (`data/features.py::extract_features_from_directory`).
2. Split directories live under **`output/<model_name>/`**
   (`WakeWordConfig.model_output_dir`), not under `data_dir`. `data_dir` holds
   `setup` artifacts (ACAV100M features, MUSAN, RIRs, VoxCPM weights) only.
3. `align_clip_to_end` **tail-truncates**: augmentation round 0 crops every
   positive to `target_length - jitter`, jitter up to 200 ms. A mined 2 s window
   ends at the exact sample the classifier scored, so it would lose up to 200 ms
   of its tail, which can include the end of the phrase.

`scripts/wake_prepare_real_data.py training-data` seeds around all three: real
windows go in as `clip_9XXXXX_r0.wav`, which round 0 skips, round 1 augments like
any other round-0 output, and feature extraction matches. They keep the
production alignment they were mined with instead of being re-aligned with a
random tail cut, and `--repeat` sets how much of the positive batch is real.

### Host gotchas, all measured on the 2026-09-14 Mac run

These are the difference between a ~5 h run and a run that never finishes.

| Symptom | Cause | Fix |
|---|---|---|
| generate at 10.4 s/clip | `tts_batch_size` 50: the VITS decoder pads every sequence in a batch to the longest, so batching a 2-word phrase costs more than it saves (8 → 0.35 s, 16 → 3.9 s, 50 → 10.4 s) | `tts_batch_size: 8` |
| VoxCPM "only" 5 GB but 184 s/clip | 136 diffusion steps, CPU-bound even on an M1 | `tts_backend: piper_vits` |
| generate wedges in uninterruptible state, `kIOGPUCommandBufferCallbackErrorOutOfMemory` | host memory-starved (swap 14.5/15 GB used); Metal allocation fails and the process never recovers | force CPU — `utils.get_device()` consults `torch.backends.mps.is_available()`, so patch it in the venv (not the repo) and keep `SAYSO_ALLOW_MPS=1` as the escape hatch |
| `setup` refuses to be useful | ACAV100M is one 17.5 GB file | check `df` first; `--skip-acav` only if the disk is genuinely short |

Both generate and augment resume: they count existing clips and continue from
there, so a killed stage costs only the batch in flight.

### Order

```bash
pip install "livekit-wakeword[train,voxcpm,export]"           # CUDA host, ~20 GB free

# 1. real clips into output/sayso/{positive,negative}_train
python scripts/wake_prepare_real_data.py training-data \
    --spool /var/lib/sayso-satellite/wake-mining \
    --ledger satellite/models/wake-spool-labels-20260914.json \
    --model-dir output/sayso \
    --held-out satellite/models/wake-held-out-20260914.json

# 2. synthetic fill, augmentation, training (each stage resumes from what exists).
#    --skip-acav because ACAV100M is one 17.5 GB file: see the config note.
python -m livekit.wakeword setup    --config sayso-training.yaml --skip-acav
python -m livekit.wakeword generate sayso-training.yaml
python -m livekit.wakeword augment  sayso-training.yaml
python -m livekit.wakeword train    sayso-training.yaml
python -m livekit.wakeword export   sayso-training.yaml
python -m livekit.wakeword eval     sayso-training.yaml

# 3. judge it on recorded audio it never saw, including the threshold
python satellite/eval/run.py --model output/sayso/sayso.onnx --eval-root ~/sayso-eval-corpus
```

Do not run `setup` on the Pi: ACAV100M plus MUSAN and RIRs is ~18 GB, and 30k
steps on a 4-core ARM is days. `get_device()` picks CUDA → MPS → CPU, so an
Apple-silicon laptop works for the classifier — but **not** for VoxCPM TTS,
which is CPU-bound and measured at 184 s/clip there (42 days of generation),
hence `tts_backend: piper_vits`.

The `training` extra in the upstream README does not exist; the extras are
`train`, `voxcpm`, `export`, `eval`, `listener`.

### Gates before shipping

- `by_category.detection_rate` on the recorded corpus: positives up, negatives
  still 0. Both halves are required — a model that rejects carriers by rejecting
  the wake word has not learned the rule.
- Threshold comes off the new metrics JSON. 0.5 is calibrated to the *current*
  model's score distribution and means nothing for a retrained one.
- `scripts/wake_bench.py` p50 well under the 160 ms hop, measured on the Pi.
- `docs/PLAN_WAKE_WORD_REAL_DATA.md` has the full verification list.
