# SaySo wake-word training (LiveKit)

How to train the "SaySo" wake classifier the satellite runs. The SaySo
language model has its own guide: [training/README.md](../README.md).

- Design and plan: [docs/SAYSO_WAKE_WORD_TRAINING_PLAN.md](../../docs/SAYSO_WAKE_WORD_TRAINING_PLAN.md)
- Data ownership and corpus layout: [docs/WAKE_TRAINING_DATA_ARCHITECTURE.md](../../docs/WAKE_TRAINING_DATA_ARCHITECTURE.md)
- How the Pi collects audio: [docs/SATELLITE_DATA_COLLECTION.md](../../docs/SATELLITE_DATA_COLLECTION.md)
- What ships on the Pi, and the verifier: [satellite/models/README.md](../../satellite/models/README.md)

## Host

Wake training runs on the same VM as the language model (`ssh llm`,
`LLM@192.168.1.76`, Radeon RX 7900 XTX, ROCm 7.2.4). Never on the Pi. The old
CUDA host `192.168.1.140` is retired, and its wake data was not migrated.

| Path on `llm` | Disk | Role |
| --- | --- | --- |
| `/srv/llm/data/wake/.venv` | HDD | Python 3.12 venv: `livekit-wakeword` 0.2.1, torch `2.7.1+rocm6.3`, onnx/onnxruntime, audiomentations |
| `/srv/llm/data/wake/corpus/` | HDD | **Canonical** long-form session corpus. `wake_corpus.py ship` lands here |
| `/srv/llm/data/wake/data/` | HDD | Named wav sets (Snowball), e.g. `200-positive/` |
| `/srv/llm/wake/runs/` | SSD | LiveKit work dirs: `setup` assets (backgrounds, RIRs, ACAV features), generated clips, checkpoints, exports |

Datasets and code live on the HDD (`/srv/llm/data`). Only LiveKit work dirs sit
on the SSD, because training reads the ACAV feature memmaps at random offsets
every epoch, and that's slow on a spinning disk.

As of 2026-09-23 all of these exist but are **empty**: the living2 inputs,
`200-positive`, and the ACAV/RIR assets have to be recovered or regenerated
before a real train.

Not yet ready on the host:

- `espeak-ng`, `ffmpeg`, `sox` are not installed (needs sudo).
- `faster-whisper` from `satellite/models/requirements-wake-train.txt` is
  missing from the venv. Install with
  `/srv/llm/data/wake/.venv/bin/pip install -r satellite/models/requirements-wake-train.txt`,
  **without** letting pip replace the ROCm torch.

The XTX is shared with vLLM (Qwen3.8, `:8000`, ~97% VRAM) and Unsloth. Start
every wake training run through the GPU lock. It stops vLLM, marks the GPU as
`train:wake`, puts `/srv/llm/data/wake/.venv/bin` first on `PATH`, and releases
the GPU when the command exits, even if it crashes:

```bash
ssh llm 'cd /srv/llm/sayso && /srv/llm/bin/gpu train wake --serve-after \
  python scripts/wake_livekit_run.py run --config satellite/models/sayso.yaml \
  --work-dir /srv/llm/wake/runs/livekit-restart'
ssh llm /srv/llm/bin/gpu status
```

`--serve-after` brings vLLM back when the run ends; leave it off to keep the GPU free.

ROCm torch uses the `torch.cuda` API names. Do not empty `HIP_VISIBLE_DEVICES`.

## Next train (LiveKit generate-first)

`satellite/models/sayso.yaml` is the primary training contract for the next
model. It follows LiveKit `configs/prod.yaml` with SaySo-only deltas (phrase
spelling variants, homophone custom negatives, `conv_attention`/`small` for Pi
inference). **Do not skip generate.** Do not train the next model from the 50
living2 clips alone.

Pipeline (isolated work dir; never write into `output-living2/`):

Run it through `gpu train wake` as shown under Host.

Stages: `setup` → `generate` (Piper TTS) → `augment` → `train` → `export` →
`eval`. Smoke wiring: `sayso-smoke.yaml` + a temp `--work-dir`.

Real Snowball / miner audio stays eval and optional later overlay, not the
primary positive class for this restart. Holdouts (`holdout_living`,
`holdout_eval`, miner party 74, `verifier_live_fp` 19) stay out of training.
`200-positive` (`/srv/llm/data/wake/data/200-positive/` once recovered) is
clean recorded data. Keep it. `runs/pos200-*` were disposable workspaces.

## Shipped model (living2, historical recipe)

`satellite/models/living2.yaml` documents the **skip-generate** recipe that
produced the ONNX on the Pi today. It is the current operating point, not the
next train. The mix is 50 this-room Snowball positives, 90 this-room train
negatives, and 89 overlapping-talk clips as **val only**.

living1 used the same wavs with `positive: 16` / `ACAV100M_sample: 256` /
`max_negative_weight: 3000` and never fired at 0.50. living2 only changes the
class prior (`positive: 96`, `ACAV100M_sample: 64`, `max_negative_weight: 200`)
so 0.50 can fire. Mixing extra TTS positives (blend) or more ACAV (living3)
failed the this-room gates. Do not repeat those.

To reproduce the shipped artifact (not recommended for the next model):

```bash
python -m livekit.wakeword augment living2.yaml
python -m livekit.wakeword train    living2.yaml
python -m livekit.wakeword export   living2.yaml
```

Ship `output-living2/sayso/sayso.onnx` back into `satellite/models/`.

`sayso-training.yaml` is a deprecated VoxCPM experiment; use `sayso.yaml`.

## Corpus sessions (Pi → host)

The Pi stages long-form sessions and ships them here over rsync/SSH. The Pi's
`~/.ssh/config` has `Host llm` / `192.168.1.76` with `id_ed25519_sayso_train`,
and that key is authorized on the VM (restricted to `from="192.168.1.54"`).

```bash
# on the Pi
/opt/sayso-satellite/.venv/bin/python scripts/wake_corpus.py \
  --corpus /var/lib/sayso-satellite/wake-sessions ship SESSION
```

Defaults: remote `LLM@192.168.1.76`, corpus `/srv/llm/data/wake/corpus`
(`satellite/sayso/wake/sessions.py`). Host-side replay, labeling, splits, and
snapshots: `wake_corpus.py replay|label|split|snapshot|holdout-eval`.

## Mining / snapshot path (secondary)

Prefer the batch wrapper when retraining from the mining spool or corpus
snapshots (labeling and idempotency, not the primary LiveKit generate path):

```bash
python3 scripts/wake_train.py \
  --spool /var/lib/sayso-satellite/wake-mining \
  --seed-dir satellite/models/data/seed \
  --work-dir satellite/models/output/wake_runs
```

Source splits and holdout rules live in `satellite/eval/splits.json`. Holdouts
must not leak into training, calibration, background mixing, or derived
features. Freeze both the deployed threshold and the calibration-selected
threshold in `satellite/eval/baseline.json` before a later retrain.

## Bootstrap blockers

- `satellite/eval/audio/` has no trusted fixtures. `cases.json` skips every case
  in default mode and fails in `--strict`. Populate with real Blue Snowball
  recordings from the living room before baseline freeze, qualification, or
  enabling `--schedule` on `scripts/wake_train.py`.
- Homophone negatives in `sayso.yaml` remain phonetic inference until the
  mining spool supplies labelled real confusions. Run
  `python scripts/wake_mine_report.py <mine_dir> --inventory` to audit wake
  assets before cleanup.

### GPU reservation and cancellation

Use `/srv/llm/bin/gpu train lfm|wake [--serve-after] COMMAND ...` for every
training run. Commands must stay in the foreground. `gpu status` reports the
reservation, owner/worker liveness, and VRAM use. A dead wrapper or worker leader does not free
a reservation while non-zombie members of its process group survive.

SIGTERM, SIGINT, and SSH hangup terminate the worker group before releasing the
GPU. A failed or interrupted LFM command restarts Unsloth to stop container-side
workers; Studio briefly disconnects. If that cleanup fails, serving remains
blocked pending inspection of `/srv/llm/run/gpu.state` and the container.
`--serve-after` restarts vLLM after successful cleanup, including command failure;
without it, the GPU remains idle. Studio is for exploration while idle; launching
training through its UI bypasses the reservation and is unsupported.

After an LFM wrapper and Docker client both crash, `gpu status` reports
`orphan:lfm` without changing services. The next `serve`, `stop`, or `train`
command restarts Unsloth under the lock before reusing the GPU; restart failure
leaves it blocked. An interrupted launch with no recorded worker PID stays
blocked for process inspection instead of assuming the GPU is idle.
