# Wake-word training: "Atlas" (LiveKit)

This guide covers training the "Atlas" wake classifier that the satellite runs.
The SaySo language model has its own guide: [training/README.md](../README.md).

- Plan: [docs/PLAN_WAKE_ATLAS.md](../../docs/PLAN_WAKE_ATLAS.md). The
  measurement-first program is in [docs/PLAN_WAKE_RETHINK.md](../../docs/PLAN_WAKE_RETHINK.md).
- Recipe: [satellite/models/atlas.yaml](../../satellite/models/atlas.yaml). It's
  LiveKit's `configs/prod.yaml` with only the phrase, the near-miss negatives,
  and `model_size: small` changed.

The "SaySo" wake word was retired on 2026-09-27. Its recordings, mined and TV
data, models, recipes, and evals were deleted from the VM, the Pi, and this
repo.

## Host

Wake training runs on the LLM VM (`ssh llm`, `LLM@192.168.1.76`, Radeon RX
7900 XTX, ROCm). It never runs on the Pi.

| Path on `llm` | Role |
| --- | --- |
| `/srv/llm/data/wake/.venv` | Python 3.12 venv: `livekit-wakeword` 0.2.1, ROCm torch, onnx/onnxruntime, audiomentations |
| `/srv/llm/wake/livekit-data/` | Shared LiveKit `setup` assets (Piper checkpoint, backgrounds, RIRs, ACAV features), phrase-agnostic, on the SSD |
| `/srv/llm/wake/runs/` | One work dir per run (generated clips, features, checkpoints, exports), on the SSD |

ROCm torch uses the `torch.cuda` API names. Keep GPU visibility enabled for
training; mask it only on CPU-only commands.

## Train

Keep inference available for the CPU stages. Reserve the GPU only for model
training; augmentation, export, and evaluation run outside the lock. The train
command's `--serve-after` restarts inference when training ends.

```bash
# on llm
W=/srv/llm/wake/runs/atlas-prod-$(date +%Y%m%d)
mkdir "$W" && cd "$W"
ln -s /srv/llm/wake/livekit-data data
cp /path/to/repo/satellite/models/atlas.yaml .
/srv/llm/bin/gpu serve
python -m livekit.wakeword setup --config atlas.yaml
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \
  python -m livekit.wakeword generate atlas.yaml
python -m livekit.wakeword augment atlas.yaml
/srv/llm/bin/gpu train wake --serve-after \
  python -m livekit.wakeword train atlas.yaml
python -m livekit.wakeword export atlas.yaml
python -m livekit.wakeword eval atlas.yaml
```

Output goes to `$W/output/atlas/`: `atlas.onnx`, plus `atlas_eval.json` from
LiveKit eval on synthetic validation data.

LiveKit eval uses synthetic audio only. It doesn't show how the model
performs in the room. Before an Atlas model ships, measure recall on real
Atlas takes and TV false fires per hour on long-form TV recordings; see
`docs/PLAN_WAKE_RETHINK.md`. The satellite also has to allow the new phrase:
`satellite/sayso/config.py` still requires `SaySo`.

## Corpus sessions and mining

The Pi-side tools are phrase-agnostic and still work for recording long-form
sessions and mining near-miss windows:

- `scripts/wake_corpus.py` ships sessions to `/srv/llm/data/wake/corpus`.
- `scripts/wake_train.py` builds a retrain from the mining spool.
- `scripts/wake_mine_report.py` summarizes a mining dir.

### Checking mined records for the wake phrase and false positives

The Pi mines windows that fire, near-threshold windows, and a small sample of
low-scoring ones, with 0.5 s of audio on each side and the HA STT transcript.
Mining only covers Atlas once the Pi runs an Atlas model.

```bash
# pull, verify + ack, push acks (the satellite then deletes acked records)
rsync -a pi@192.168.1.54:/var/lib/sayso-satellite/wake-mining/ SPOOL/
python3 scripts/wake_mine_report.py SPOOL --ingest
rsync -a SPOOL/acks/ pi@192.168.1.54:/var/lib/sayso-satellite/wake-mining/acks/

# classify: wake / false_positive / missed_wake / near_miss, and export sets
uv run --no-project --with faster-whisper --with numpy \
  python scripts/wake_mine_check.py SPOOL --phrase Atlas --export OUT
```

The class is an ASR suggestion written to `check.json`. `wake` and
`missed_wake` clips must be listen-verified before they're used as
positives.

Their SaySo defaults (`sayso-training.yaml`, `satellite/eval/*.json`) were
deleted. Pass explicit paths.

### Real Koda seeds and distance variants

`docs/PLAN_KODA_REAL_TAKES.md`. Listen-verified real takes seed extra
positives; a third are held out for eval and never emitted.

```bash
python3 scripts/wake_koda_seeds.py select SPOOL --out seeds.json
# listen, then set "verified": true on real Koda takes
python3 scripts/wake_koda_seeds.py variants SPOOL --seeds seeds.json \
  --rirs data/rirs/16khz --backgrounds data/backgrounds --out OUT -n 50
```

Copy `OUT/*.wav` into the new run's `output/koda/positive_train/` before
`augment`. Each seed yields near-field clips plus 1, 2 and 3 m clips.

### GPU reservation and cancellation

Use `/srv/llm/bin/gpu train lfm|wake [--serve-after] COMMAND ...` for every
training run. Commands must stay in the foreground. `gpu status` reports the
reservation, owner/worker liveness, and VRAM use. A dead wrapper or worker leader does not free
a reservation while non-zombie members of its process group survive.

SIGTERM, SIGINT, and SSH hangup terminate the worker group before releasing the
GPU. A failed or interrupted LFM command restarts Unsloth to stop container-side
workers; Studio briefly disconnects. If that cleanup fails, serving remains
blocked pending inspection of `/srv/llm/run/gpu.state` and the container.
`--serve-after` restarts serving after successful cleanup, including command failure;
without it, the GPU remains idle. Studio is for exploration while idle; launching
training through its UI bypasses the reservation and is unsupported.

After an LFM wrapper and Docker client both crash, `gpu status` reports
`orphan:lfm` without changing services. The next `serve`, `stop`, or `train`
command restarts Unsloth under the lock before reusing the GPU; restart failure
leaves it blocked. An interrupted launch with no recorded worker PID stays
blocked for process inspection instead of assuming the GPU is idle.
