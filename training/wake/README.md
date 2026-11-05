# Wake-word training: "Koda" (LiveKit)

This guide covers training the "Koda" wake classifier that the satellite runs.
The SaySo language model has its own guide: [training/README.md](../README.md).

- Plan: [docs/PLAN_WAKE_KODA_RECOVERY.md](../../docs/PLAN_WAKE_KODA_RECOVERY.md). The
  measurement-first program is in [docs/PLAN_WAKE_RETHINK.md](../../docs/PLAN_WAKE_RETHINK.md).
- Recipe: [satellite/models/koda.yaml](../../satellite/models/koda.yaml). It's
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
| `/srv/llm/data/wake/librispeech/` | Raw public LibriSpeech download (phrase-agnostic), on the HDD |

ROCm torch uses the `torch.cuda` API names. Keep GPU visibility enabled for
training; mask it only on CPU-only commands.

## Train

Keep inference available for the CPU stages. Reserve the GPU only for model
training; augmentation, export, and evaluation run outside the lock. The train
command's `--serve-after` restarts inference when training ends.

```bash
# on llm
W=/srv/llm/wake/runs/koda-prod-$(date +%Y%m%d)
mkdir "$W" && cd "$W"
ln -s /srv/llm/wake/livekit-data data
cp /path/to/repo/satellite/models/koda.yaml .
/srv/llm/bin/gpu serve
python -m livekit.wakeword setup --config koda.yaml
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \
  python -m livekit.wakeword generate koda.yaml
python -m livekit.wakeword augment koda.yaml
/srv/llm/bin/gpu train wake --serve-after \
  python -m livekit.wakeword train koda.yaml
python -m livekit.wakeword export koda.yaml
python -m livekit.wakeword eval koda.yaml
```

Output goes to `$W/output/koda/`: `koda.onnx`, plus `koda_eval.json` from
LiveKit eval on synthetic validation data.

LiveKit eval uses synthetic audio only. It doesn't show how the model
performs in the room. Before an Koda model ships, measure recall on real
Koda takes and TV false fires per hour on long-form TV recordings; see
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
Mining only covers Koda once the Pi runs an Koda model.

```bash
# pull, verify + ack, push acks (the satellite then deletes acked records)
rsync -a pi@192.168.1.54:/var/lib/sayso-satellite/wake-mining/ SPOOL/
python3 scripts/wake_mine_report.py SPOOL --ingest
rsync -a SPOOL/acks/ pi@192.168.1.54:/var/lib/sayso-satellite/wake-mining/acks/

# classify: wake / false_positive / missed_wake / near_miss, and export sets
uv run --no-project --with faster-whisper --with numpy \
  python scripts/wake_mine_check.py SPOOL --phrase Koda --export OUT
```

The class is an ASR suggestion written to `check.json`. `wake` and
`missed_wake` clips must be listen-verified before they're used as
positives.

Their SaySo defaults (`sayso-training.yaml`, `satellite/eval/*.json`) were
deleted. Pass explicit paths.

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

## TV-on: logit ensemble

TV-only models (real room Kodas mixed over training TV) fix real-voice-over-TV
recall, but every single one false-wakes 1-2.5/h on holdout TV at 0.65. Logit
averaging with v9b fixes that; prob-averaging does not.

Recipe: the user's own room Kodas mixed over training TV at -30..-16 dBFS, in
two runs (v10k, v10l), ensembled with deployed v9b. Build the ensemble ONNX
(same interface as members, `embeddings [batch,16,96]` -> `score [batch,1]`,
with a built-in self-check and sha256):

```bash
python3 scripts/wake_make_ensemble.py \
  koda-phonics-lg_9b_10k_10l.onnx \
  v9b.onnx v10k.onnx v10l.onnx
```

`--prob` gives the mean of member sigmoids instead; use logit averaging.

The staging and real-voice recipes are `scripts/wake_stage_tv.py`
(`python3 scripts/wake_stage_tv.py --variant v10k REALNEG CLIPS OUT SCORER.onnx`) and
`scripts/wake_eval_realtv.py` (`python3 scripts/wake_eval_realtv.py real --pos POS_DIR --tv TV_DIR MODEL.onnx`,
with an `fa` subcommand for holdout false wakes/h). They import
`livekit.wakeword.eval.stream`, which exists only in the patched livekit checkout in the VM
run dir, so run them as:

```bash
CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \
PYTHONPATH=/srv/llm/data/wake/runs/koda-phonics-v6-20261018/src \
  /srv/llm/wake/runs/koda-phonics-v3-20261008/.venv/bin/python scripts/wake_eval_realtv.py ...
```

(`wake_make_ensemble.py` needs only onnx + onnxruntime and no PYTHONPATH.)

At threshold 0.65:

| | v9b (previous) | ensemble 9b+10k+10l (deployed 2026-11-04) |
| --- | --- | --- |
| Holdout TV false wakes | 0 | 0 (also 0 at 0.55) |
| LiveKit FPPH | 0.31 | 0.22 |
| Real TV-on, user (n=7) | 2/7 (expected 1.8) | 4/7 (expected 3.6) |
| Real room voice x holdout TV (n=132) | 41 | 57 |
| Synthetic clean (n=1000) | 869 | 908 |
| Quiet room, user (n=16) | 11/16 (expected 10.1) | 10/16 (expected 9.4) |
| CPU per window (VM) | 0.08 ms | 0.25 ms |
| CPU per 80 ms hop (Pi) | 2.1 ms | 6.0 ms |

See [docs/PLAN_WAKE_TV_ENSEMBLE.md](../../docs/PLAN_WAKE_TV_ENSEMBLE.md) for the
run-up, the open quiet-room gap, and the Pi deploy steps.

## Prefix negatives

`scripts/wake_stage_prefix.py` cuts raw TTS Kodas at 35-70% of the spoken
span ("Ko".."Kod", final vowel dropped), runs them through LiveKit's augmentor,
end-aligns them like positives, and writes `prefix_negative_features_train.npy`
for appending to a run's `negative_features_train.npy`. End alignment matters:
v9b scores a raw Koda ~0.9 at the window end and ~0.02 mid-window, despite
`common_window: true` in its recipe.

2026-11-04, 5,000 v6 clips, threshold 0.65 (full-word control: v9b 85.5%,
ensemble 87.2%):

| prefix kept | v9b fires | ensemble fires |
| --- | --- | --- |
| 35-50% | 14.5% | 14.6% |
| 50-60% | 34.7% | 36.2% |
| 60-70% | 50.2% | 51.8% |

v10r (v9b recipe + these 5,000 rows in the adversarial negatives, one seed;
`run_v10r.sh` on the VM) at 0.65, with held-out prefixes cut from `positive_test`:

| | v9b | v10r | ens 9b+10k+10l | ens 10r+10k+10l |
| --- | --- | --- | --- | --- |
| Held-out prefixes fired (35-50 / 50-60 / 60-70%) | 15 / 31 / 51% | 0.2 / 0 / 2.7% | 12 / 28 / 52% | 0.5 / 1.9 / 12% |
| Synthetic clean (n=1000) | 869 | 781 | 908 | 888 |
| LiveKit FPPH | 0.31 | 0.04 | | |
| Real room voice x TV -48..-38 / -38..-28 / -28..-18 (n=132 each) | 89 / 66 / 41 | 76 / 59 / 37 | 102 / 86 / 57 | 105 / 82 / 54 |
| Real TV-on, user (n=7, expected) | 1.81 | 1.56 | 3.62 | 3.81 |
| Quiet room, user (n=16, expected) | 10.12 | 9.75 | 9.44 | 9.56 |
| Holdout TV false wakes (2 h) | 0 | 0 | 0 | 0 |
| 2026-11-03 TV windows fired (n=1323) | 0 | 1 | 0 | 0 |

Prefix negatives remove prefix firing but cost the single model ~10% synthetic
recall and real-voice-over-TV recall in every band: it now waits for the final
vowel, the phoneme TV masks most. In the ensemble the swap is a wash within
noise. Not deployed; no real false wake has been traced to a prefix.

## TV-on: synthetic voices over TV (v10t)

v10t = v9b + 25,000 TTS Kodas over the train TV hours (positives) + 25,000 TTS
"distant" phrases over the same TV (adversarial negatives). Staging:

- `scripts/wake_stage_tts_tv.py`: end-aligned TTS, voice at -21..-11 dBFS
  active, TV as recorded +(-3..+9) dB (levels independent), and TTS EQ'd onto
  the room Kodas' average spectrum (`--match`).
- `scripts/wake_tts_distant.py`: Piper TTS of 1-3 word chunks from the train
  TV hours' ASR, keeping only chunks more than 2 phone edits from K OW D AH.

Don't pair the TV positives with near-miss words over TV (v10s). The frozen
embedding can't tell Koda from its near-misses under TV (probe AUC 0.78 vs 0.96
clean), so the model rejects all of it.

Two seeds (v10t default 42, v10t2 seed 7) at 0.65; each ensemble swaps the
new model in for v9b alongside v10k + v10l:

| | deployed 9b+10k+10l | 10t+10k+10l | 10t2+10k+10l |
| --- | --- | --- | --- |
| Real room voice x holdout TV -48..-38 / -38..-28 / -28..-18 (n=132 each) | 102 / 86 / 57 | 112 / 100 / 65 | 108 / 100 / 65 |
| Real TV-on, user (n=7, expected) | 3.62 | 3.94 | 4.25 |
| Quiet room, user (n=16, expected) | 9.44 | 9.38 | 9.19 |
| Synthetic clean (n=1000) | 908 | 901 | 901 |
| Holdout TV false wakes/h at 0.65 / 0.55 (2 h) | 0 / 0 | 0 / 0.5 | 0 / 0 |
| 2026-11-03 TV windows fired (n=1323) | 0 | 0 | 0 |
| Room speech over holdout TV fired (n=327) | 0 | 0 | 0 |

Single v10t/v10t2 models reach the deployed ensemble's TV-on numbers, but v10t2
alone has 0.5 false wakes/h at 0.65. Ship them only inside an ensemble.
