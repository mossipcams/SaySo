# Plan: reorganize the `llm` VM around the two training pipelines

Host: `ssh llm` (`LLM@192.168.1.76`), Radeon RX 7900 XTX 24 GiB, ROCm 7.2.4,
19 GiB RAM + 12 GiB swapfile, no passwordless sudo. Audited 2026-09-23; plan
reviewed the same day.

| Disk | Mount | Size | Free (2026-09-23) | Holds |
| --- | --- | --- | --- | --- |
| `sda` SSD | `/` (incl. `/srv/llm`, Docker's `/var/lib/docker`) | 300 G | 56 G | Docker images (133 G), models, training runs |
| `sdb` HDD | `/srv/llm/data` | 400 G | 232 G | **Datasets and generation code**, archives, history |

Rule: datasets and generation code go on the HDD. The SSD is only for what
needs fast random IO or fast loading: Docker images, model weights vLLM and
Unsloth load, training outputs, and LiveKit work dirs.

SaySo trains two models on this one GPU, with separate pipelines:

| Model | Trainer | Guide |
| --- | --- | --- |
| SaySo LFM (`LFM2.5-230M-Base`, full SFT) | Unsloth (`unsloth` container) | [training/README.md](../training/README.md) |
| Wake word | LiveKit (`livekit-wakeword`, host venv) | [training/wake/README.md](../training/wake/README.md) |

vLLM (Qwen3.8-27B-INT4, `:8000`, ~97% VRAM) shares the same GPU for serving.

## Problems found

The VM is three setups stacked on each other: the Sep 18 LLaMA-Factory
layout, the retired `192.168.1.140` host's data copied into `/srv/llm/data`,
and Unsloth added on 2026-09-23.

1. **No GPU arbitration.** `start-vllm.sh` checks `run/locks/rocm.state`, but
   only the removed `llm-rocm` container wrote `train` there. Unsloth
   (`restart: unless-stopped`) and LiveKit never take the lock, so a training
   run and vLLM can run out of GPU memory at the same time.
2. **Unsloth can reach everything.** It mounts all of `/srv/llm` read-write,
   including `config/secrets.env` (vLLM API key). Its UI listens on `0.0.0.0:8002`.
3. **Dead LLaMA-Factory tooling.** `scripts/llm`, `scripts/gpu-lock`,
   `compose/` (`llm-cuda`, `llm-rocm`), `run/locks/cuda.*`, `cache/`,
   `config/versions.env`, LLaMA-Factory recipes in `config/`, and 5.2 GB of
   LLaMA-Factory smoke runs in `runs/`.
4. **Stale paths.** Copied history (`data/models/CHAMPION.txt`,
   `data/training-runs/*.yml|*.sh`) points at `/srv/models`, `/srv/datasets`,
   `/srv/training-runs`, which don't exist here.
5. **Duplicates and junk.** Two data roots (`models|datasets|runs` vs.
   `data/*`). Base weights stored twice. Four HF caches. 91 GB of stale
   dataset caches from the old host. A 6.7 GB CUDA venv. Backup files next to
   the live compose and secrets.
6. **RAM headroom.** vLLM's checkpoint is 18.1 GiB against ~15.5 GiB of
   available RAM; the 12 GiB swapfile (on the SSD) had 3.3 GiB in use. Don't run
   corpus generation while vLLM is loading.
7. **SSD is the tight disk.** 56 G free, and it fell 12 G during the audit for
   reasons that couldn't be traced without sudo (Docker's store). Meanwhile the
   code and small datasets sit on the SSD while the HDD has 232 G free.

## Rules for every task

- **Anything not listed is kept.** Only explicitly listed paths are deleted
  or moved, by explicit path, never by glob over a parent folder. Check each
  target before removing it.
- **The old host is gone; this is the only copy.** Anything that can't be
  regenerated gets archived off the VM before deletion
  (`LLM@192.168.1.76:… → Mac: ~/SaySo-archive/vm-20260923/`). A copy on the
  VM's own HDD doesn't count: same VM, same failure. Hash manifests
  record what existed, but they can't restore it.
- **Venvs can't be moved.** They record absolute paths. Rebuild them at the new
  location; never `mv` them.
- **Path changes land in the repo in the same step.** After any move,
  `grep -rn "<old path>"` over the repo (docs, `satellite/`, `training/`,
  yamls) and over `/srv/llm` (excluding `lfm/legacy-140/`) must come back empty.

## Target layout

```
HDD  /srv/llm/data/                  datasets + generation code
       sayso/                        git checkout (read-only deploy key) + training/.venv (generation)
       lfm/datasets/                 v5 canonical + rendered view
       lfm/legacy-140/               retired host's LFM history (paths as recorded)
       wake/.venv/                   LiveKit env (stays here; venvs can't move)
       wake/corpus/  wake/data/      wake datasets                     ← done 2026-09-23
       archive/                      tarballs (pre-git sayso copy, etc.)

SSD  /srv/llm/
       sayso -> data/sayso           symlink keeps every documented path and venv valid
       lfm/models/                   LFM2.5-230M-Base (one copy)
       lfm/runs/                     Unsloth outputs; llamafactory-smoke/ (logs only)
       wake/runs/                    LiveKit work dirs (ACAV memmaps)   ← done 2026-09-23
       serve/vllm/                   compose, secrets.env, start/stop, cache/, Qwen3.8 model
       services/unsloth/             compose, studio/ state
       hf-cache/                     one shared HF cache
       bin/                          gpu (Task 4) + kept host scripts
       run/                          gpu.state, gpu.lock
```

## Task 1: Cleanup (no running service affected)

Before deleting: copy the two recipes that exist only on the VM,
`config/sayso-lfm-smoke.yml` and `config/sayso-lfm-v5-smoke-bf16.yml` (the bf16
decision in `PLAN_LFM_HOST_V5_SETUP.md`), into
`training/configs/history/` in the repo.

| Remove | Size | Why dead |
| --- | --- | --- |
| `data/hf-cache/datasets/` | 91 GB | Arrow caches keyed to `/home/ubuntu` and `/srv/hf-cache` on the old host |
| `data/training-runs/.venv`, `data/training-runs/gen-venv`, `data/training-runs/__pycache__` | 6.7 GB | CUDA/Pascal envs; this GPU is AMD |
| `compose/`, `scripts/llm`, `scripts/gpu-lock`, `scripts/gpu-lock.bak-*`, `scripts/run-infer`, `scripts/attach-gpus.txt` | small | LLaMA-Factory containers removed (`compose/.env` holds no secrets) |
| `cache/` | 204 KB | HF home and DeepSpeed configs of the removed containers |
| `run/locks/cuda.*`, `run/llamafactory-build.log`, `run/rocm-webui.*`, `run/rocm-supervisord.log`, `run/preflight_lfm2*.py` | small | LLaMA-Factory era |
| `config/versions.env`, `config/lfm25-230m-smoke.yml`, `config/sayso-lfm-*.yml` | small | LLaMA-Factory recipes; in git after the copy above |
| `runs/sayso-lfm-v5-smoke*/checkpoint-*`, `runs/*/model*.safetensors`, `runs/*/training_args.bin` | ~5 GB | LLaMA-Factory smoke weights; trainer gone |
| `docker-compose.yml.bak-*`, `.prev`, `.vllm-qwen3.6`, `config/secrets.env.bak-*` | small | Backups beside live files |
| `hf-cache/hub/models--QuantTrio--Qwen3.6-27B-AWQ` | 12 KB | Empty stub of the pre-3.8 Qwen |

Keep:
- `run/rocm-train.log`: move to `runs/llamafactory-smoke/` (evidence for the HIP smoke).
- The smoke runs' `trainer_log.jsonl`, `*_results.json`, `trainer_state.json`, `training_loss.png`.
- The other `scripts/` files (`bench.sh`, `bootstrap-host.sh`, `containerd-to-ssd.sh`,
  `move-containerd.sh`, `download-lfm.py`, `sayso-lfm-setup.py`, `selfcheck.sh`,
  `try-config.sh`). Task 3 moves them to `bin/` or retires them one by one.

Done when: the HDD (`df -h /srv/llm/data`) gains ~98 GB (the HF cache and CUDA
venv live there), the SSD (`df -h /`) gains ~5 GB (smoke weights), and
`docker ps` shows `vllm` and `unsloth` with unchanged uptime.

## Task 2: Clean up the old host's data

`/srv/llm/data/{models,datasets,training-runs}` (~37 GB after Task 1) is the
retired `192.168.1.140` host copied over unchanged. All of it is rsLoRA-era
LFM work on a GTX 1070. v5 is full SFT from Base on a regenerated corpus, so
none of it is an input anymore. Its value is **evidence**: what shipped, what
won on evals, and the eval sets that old scores in `training/TRAINING_LOG.md`
refer to.

Steps:

1. `mkdir -p /srv/llm/data/lfm/legacy-140` (HDD; the history never leaves it).
2. Write the manifest. Record `sha256  size  path` for every file to be deleted
   in `data/lfm/legacy-140/DELETED.sha256`, and commit a copy to
   `training/legacy-140-deleted.sha256`. In the log's host section, add one line
   per deleted dataset or artifact (its 16-hex hash and path), not the whole
   manifest.
3. Archive to the Mac anything under **Delete** or **Decide** that can't be
   regenerated. That's every dataset and GGUF, not checkpoints whose selected
   step already exists as a GGUF.
4. Delete the approved paths.
5. Write `data/lfm/legacy-140/CHAMPION.md` and delete both `CHAMPION.txt` files.

**Merged-weights policy:** keep one `-merged` HF folder only for the shipped
model (`run013` step 2500), because it's the source for re-quantizing the
shipped artifact. Everywhere else the f16 GGUF is the kept copy, and the
`-merged` folders go.

### Keep (evidence)

| Path under `data/` | Why |
| --- | --- |
| `training-runs/run013-eval/artifacts/step-2500-{Q8_0.gguf,f16.gguf,merged}` | Provenance of the shipped `model-v1` (md5 `06a7def0…`, verified 2026-09-23) |
| `training-runs/sayso-semantic-early-20260908/artifacts/step-{2500,5000}-{Q8_0.gguf,f16.gguf}` | Champion by evaluation (Run 011/012) |
| Top-level adapter files of `SaySo-LFM2.5-230M-{v3-semantic-early-20260908,40k-grounded-gauntlet}/` | Final (step 5000) adapters |
| `models/LFM2.5-230M-Base-{Q8_0,f16}.gguf` | Untuned Base baseline for scoring |
| `datasets/sayso_test_balanced.jsonl` and the eval files `sayso_v2/sayso_quality_eval_recipe_lock.jsonl`, `sayso_v2/sayso_shadow_eval.jsonl`, `sayso_v3/sayso_quality_eval_v3_{gold,shadow}.jsonl` | Suites the log's scores cite |
| `training-runs/{evals-pinned-namespaced,gauntlet_eval,*-source}/`, `*.yml`, `*.sh`, `*.py`, `*.log`, `*.json`, each run's `README.md` / `trainer_state.json` / `runs/` | Small; the only record of how old runs were launched and scored |

### Delete (bad or redundant)

| Path under `data/` | Size | Why |
| --- | --- | --- |
| `models/SaySo-LFM2.5-230M-40k-gauntlet-step2500-Q8_0.gguf` | 236 MB | Byte-identical to the run013 artifact (same md5) |
| `models/LFM2.5-230M-Base` | 443 MB | Duplicate of `/srv/llm/models/LFM2.5-230M-Base` (confirm hash) |
| `training-runs/run013-eval/artifacts/step-{3250,5000}-*` | 2.2 GB | Scored, not promoted, not shipped (Run 013) |
| `training-runs/sayso-semantic-early-20260908/artifacts/step-250-*` | 1.1 GB | Early probe checkpoint, superseded by 2500/5000 |
| `training-runs/sayso-semantic-early-20260908/artifacts/step-{2500,5000}-merged` | 0.9 GB | Merged-weights policy above |
| `training-runs/SaySo-LFM2.5-230M-{v3-semantic-early-20260908,40k-grounded-gauntlet}/checkpoint-*` | 3.8 GB | Rotating LoRA checkpoints (both runs reached step 5000); the selected steps live in `artifacts/` |
| `datasets/sayso_realistic_20260908/` | 762 MB | Run 010 was stopped for semantic **label defects** |
| `datasets/sayso_v3_grounded_20260911/`, `datasets/sayso_v3_grounded_20260911_r2/` | 795 MB | Superseded by the gauntlet corpus; `_r2` holds only a `generation.log` (keep that log, it's cited in `HANDOFF_gauntlet_v2.md`) |
| `datasets/sayso_40k_ha_contract_20260914/` | 1.6 GB | v1, superseded by `_v2` |
| `datasets/{sample.jsonl,home_assistant_train.jsonl,sayso_train_2551_raw.jsonl,sayso_train_2552_raw.jsonl}` | 0.3 GB | Referenced by no log entry, run config, or launch script (checked 2026-09-23) |
| `models/Qwen3.5-9B-Q4_K_M.gguf` | 5.3 GB | Old serving model; serving is Qwen3.8-27B-INT4 in vLLM. Downloadable again, no archive needed |

### Replace

`models/CHAMPION.txt` claims `served_by: llama-server (PID 139303) on port
8080` and lists `/srv/training-runs/...` paths. `training-runs/CHAMPION.txt`
names a model no longer on disk. Replace both with one
`data/lfm/legacy-140/CHAMPION.md`:

- shipped: `model-v1`, with its release sha256 and its source artifact
- champion by eval: semantic-early step 5000
- a note that serving and eval champion diverge on purpose

### Decide (needs a yes from the user)

| Path under `data/` | Size | State |
| --- | --- | --- |
| `training-runs/Newhaven/` + `newhaven-eval/` + `datasets/haven/` | 1.9 + 1.2 + 1.5 GB | rsLoRA run stopped at step 2,650 of 5,000; never promoted; eval not audited |
| `training-runs/SaySo-LFM2.5-230M-ha-contract-v2-20260914/` + `ha-contract-v2-eval/` + `datasets/sayso_40k_ha_contract_v2_20260914/` | 1.9 + 1.1 + 1.6 GB | Stopped at step 3,950 of 5,000; same |
| `training-runs/sayso-gauntlet-v2*-source/`, `datasets/sayso_40k_grounded_gauntlet_v2_20260914/` | 1.1 GB | Gauntlet v2: not in the log at all |
| `training-runs/bakeoff-gemma-vs-gauntlet/`, `models/Home-FunctionGemma-270m.q8_0.gguf` | 0.3 GB | FunctionGemma bakeoff: not in the log |
| Training corpora (non-eval files) in `sayso_v2/`, `sayso_v3/`, `sayso_ohf_20260910/`, `sayso_quality_20260907/`, `sayso_semantic_20260908/`, `sayso_40k_grounded_gauntlet_20260911/`, plus `sayso_train*.jsonl` | ~5 GB | Logged by hash; can't be regenerated byte-for-byte |

Recommended:
- Delete the checkpoints of the two unfinished runs; keep their logs,
  `trainer_state.json`, and eval summaries.
- Delete FunctionGemma.
- Keep the logged corpora, compressed (`zstd`), in `data/lfm/legacy-140/datasets/`
  (archive copy on the Mac per the rules).

Done when: the manifest is committed, and the kept artifacts still hash to
their recorded values (shipped GGUF md5 `06a7def0…`). One `CHAMPION.md`
replaces both markers. `data/` shrinks from ~37 GB to ~20 GB, or to ~13 GB
with the Decide recommendations.

## Task 3: Per-pipeline, per-disk layout — complete

HDD moves are within one filesystem (`mv` is instant). SSD↔HDD moves copy data.

1. **Code to the HDD.** `mv /srv/llm/sayso /srv/llm/data/sayso`, then
   `ln -s data/sayso /srv/llm/sayso`. The generation venv was built as
   `/srv/llm/sayso/training/.venv`, and that path still resolves through the
   symlink, so it keeps working. Check with `training/.venv/bin/python -c "import generators"`.
2. **Datasets to the HDD.** `datasets/*` (SSD, 2 MB) → `data/lfm/datasets/`.
   Generation writes its corpus into the checkout (HDD); the canonical v5 files go
   to `data/lfm/datasets/`.
3. **History stays on the HDD.** What Task 2 kept of
   `data/{training-runs,datasets,models}` → `data/lfm/legacy-140/` (same disk). Add
   `data/lfm/legacy-140/README.md` recording the old `/srv/...` prefix. Don't
   rewrite historical files. `exports/` → `data/archive/`.
4. **Models and runs on the SSD.** `models/LFM2.5-230M-Base` → `lfm/models/`.
   `runs/*` (what Task 1 kept) → `lfm/runs/llamafactory-smoke/`.
5. **vLLM** → `serve/vllm/`: `models/Qwen3.8-27B-INT4`, root `docker-compose.yml`,
   `start-vllm.sh`, `stop-vllm.sh`, `config/secrets.env`, and
   `data/vllm-cache` → `serve/vllm/cache` (544 MB of compiled kernels, HDD → SSD;
   without it the next start recompiles for ~3 min). Update the compose volume paths.
6. **Unsloth** → `services/unsloth/`: its compose, plus `data/unsloth/studio` →
   `services/unsloth/studio`. Mounts: `/srv/llm/lfm` (rw, SSD),
   `/srv/llm/data/lfm/datasets` (ro, HDD), `/srv/llm/data/sayso` (ro, HDD, for LFM
   training scripts), `/srv/llm/hf-cache`, and `studio/`. No `secrets.env`, no
   rest of `/srv/llm`. Before real runs, check that dataset loading from the HDD
   isn't the bottleneck; if it is, stage the rendered view into `lfm/runs/<run>/`
   for that run only.
7. **One HF cache** on the SSD: merge `hf-cache/` and `data/unsloth/hf-cache/`
   into `/srv/llm/hf-cache`, and point both composes at it.
8. **Wake:** nothing to move. The venv stays at `data/wake/.venv` (HDD, only
   slower to start), data is on the HDD, and work dirs are in `wake/runs/` (SSD).
9. Kept `scripts/*` → `bin/` if still used, otherwise delete one by one. Then
   remove the empty `config/`, `models/`, `datasets/`, `runs/`, `scripts/`,
   `unsloth/`, `exports/`, `data/models/`, `data/datasets/`, `data/training-runs/`,
   `data/unsloth/`.
10. Restart `unsloth` (brief, new mounts), then vLLM at a quiet time (~3 min reload).
11. Repo: update every doc that names a moved path (`training/README.md`,
    `training/wake/README.md`, `docs/PLAN_LFM_HOST_V5_SETUP.md`,
    `docs/SAYSO_LFM_TRAINING_PLAN.md`, `training/TRAINING_LOG.md` host note), plus
    the Claude memory note for the host.

Done when:
- `vllm` serves on `:8000`.
- Unsloth Studio loads on `:8002` and sees the Base model and the datasets.
- The generation venv imports `generators` through `/srv/llm/sayso`.
- `wake_corpus.py ship --dry-run` from the Pi reaches `/srv/llm/data/wake/corpus`.
- `du` shows no datasets or code on the SSD outside `lfm/runs/` and `wake/runs/`.
- The grep rule passes for `/srv/llm/models`, `/srv/llm/datasets`,
  `/srv/llm/runs`, `/srv/llm/config`, `/srv/llm/data/training-runs`,
  `/srv/llm/data/models`.

## Task 4: One GPU lock (`bin/gpu`) — complete

vLLM runs as a detached container, so no process can hold a lock for as long
as it serves. The lock is a **state file**, `run/gpu.state`, set to one of
`idle | serve | train:lfm | train:wake` plus a PID. `flock run/gpu.lock` is held
only while switching states.

| Command | Does |
| --- | --- |
| `gpu serve` | If the state is `train:*` and its PID is alive, refuse. Otherwise start vLLM and set `serve`. Replaces `start-vllm.sh`. |
| `gpu train lfm <cmd>` | Stop vLLM, set `train:lfm` with its PID, run `<cmd>` via `docker exec unsloth`, set `idle` on exit (via a trap, so crashes release it too). Restarting vLLM is a flag (`--serve-after`), not the default. |
| `gpu train wake <cmd>` | Same, with `<cmd>` run by `/srv/llm/data/wake/.venv` on the host. |
| `gpu stop` | Stop vLLM and set `idle`. Replaces `stop-vllm.sh`. |
| `gpu status` | State, PID, whether that PID is alive, VRAM use (`rocm-smi`). |

A dead owner is not enough to release a reservation: wake worker groups must
be gone, and stale LFM reservations require an Unsloth restart under the lock
before reuse. Status reports stale LFM as `orphan:lfm` without restarting it.

**Decision (2026-09-23):** use the recommended script-only training path below.
Starting a run from the Unsloth Studio UI can't be made to
go through the lock. Options:

- **(Recommended)** Real LFM runs start from a script via `gpu train lfm`.
  Studio is only for exploring, and only while `gpu status` shows `idle`.
- Studio is used for real runs, and `gpu stop` beforehand stays a manual step.

Done when:
- `gpu train wake sleep 30` stops vLLM, and a concurrent `gpu serve` refuses.
- Killing the train makes `gpu status` report `idle`, and `gpu serve` then
  brings vLLM back.
- `start-vllm.sh`, `stop-vllm.sh`, and `run/locks/` are gone.
- Both training READMEs document `bin/gpu` as the only way to start training.

## Task 3 continuation — 2026-09-23

Reviewed scope: finish vLLM service relocation, Unsloth state/cache relocation,
and host script paths; retain current vLLM v0.30.0 flags and project name `llm`.
Completed earlier moves are left in place. Retired setup scripts and alternate
compose recipes are preserved in the HDD archive, with an off-VM copy before
retirement. No training runs or corpus generation are started.

Files: `scripts/llm-host/gpu`, Pi checkout/installed shipping-default constants, the training READMEs, host setup plan, training
log, and Claude host memory; remote service composes/start-stop scripts and
kept host helpers. Historical recipes, manifests, tests, and this migration
plan retain recorded old paths; verify active references separately instead
of rewriting provenance or modifying tests to make a literal grep empty.

Verification: compose config (quiet), existing GPU self-check, vLLM health
and authenticated completion, Studio HTTP and model/dataset visibility,
generation import through the symlink, Pi corpus shipping dry-run, filesystem
placement and active old-path scan. Preserve cache collisions rather than
overwrite different bytes. Stop services before moving writable state.

### Applied layout

- vLLM moved to `serve/vllm/` with Compose project `llm`, its current v0.30.0
  image and flags unchanged. Active cache is `cache/`; previous-version cache
  is retained as `cache-pre-v030/` to avoid mixing compiled kernels.
- Unsloth Studio state moved to `services/unsloth/studio/`; both services now
  mount the shared SSD `hf-cache/`. The remaining old HDD HF cache was merged
  too. Differing cache metadata is preserved in `data/archive/task3-cache-collisions/`
  and in the Mac archive.
- Active host helpers: `bin/bench.sh`, `bin/try-config.sh`, `bin/download-lfm.py`.
  Retired bootstrap, LLaMA-Factory setup/self-check, containerd move scripts,
  old config and alternate compose recipes are in `data/archive/task3-retired/`.
  Private off-VM originals: `~/SaySo-archive/vm-20260923/task3/` on the Mac.
- Empty old directories removed. Generation venv still imports `generators`
  through the checkout symlink. Legacy history README records old prefixes.
- Pi shipping defaults updated in its checkout and installed module. A temporary
  synthetic session dry-run selects `LLM@192.168.1.76:/srv/llm/data/wake/corpus`;
  a separate SSH check confirms destination reachability and write permission.
  Dry-run itself does not connect or transfer audio. The temporary session was removed.
- Task 4's legacy start/stop scripts and state mirror remain for that task;
  the scripts moved into `serve/vllm/`, and `bin/gpu` uses its new compose path.

### Task 3 verification — complete, 2026-09-23

Both compose configurations validate; GPU self-check and shell syntax checks
pass. vLLM `/health` and Studio return HTTP 200; authenticated vLLM completion
returned `OK`. Studio's container sees Base `config.json` and the datasets,
with dataset and checkout mounts read-only and no host secrets mount.
Filesystem checks confirm HDD checkout/datasets and SSD service state/weights.
Approximately 60 G SSD and 361 G HDD are free after migration.

Active host helpers and service configuration contain none of the six retired
path prefixes. Literal whole-repo grep still finds deliberately historical
LLaMA-Factory recipes/validator/tests, this plan, and archived records; those
are not live launch paths and were not rewritten. No tests were modified.
Dataset throughput for a real Unsloth run remains a pre-training check, since
the full v5 corpus and Unsloth recipe are not yet available.

## Task 4 implementation plan — reviewed 2026-09-23

Scope: complete the existing Bash GPU command, retaining its CLI. Remove legacy
state mirroring and launch scripts. Serialize cleanup with state transitions;
record wrapper and worker PIDs so a killed wrapper cannot free a live worker's
reservation. Run wake workers in a process group; on interruption terminate
that group. On failed/interrupted LFM commands restart Unsloth to reap remote
workers before releasing state. Reject malformed state rather than serve.
Studio remains exploration-only; all training uses `gpu train`.

Files: `scripts/llm-host/gpu`, its colocated `test_gpu.sh` (outside `tests/`),
training READMEs, `satellite/models/sayso.yaml` launch comment, this plan and
training log. Remote: deploy GPU script/check, update `bin/try-config.sh` and
compose comment, archive/remove `serve/vllm/{start,stop}-vllm.sh` and
`run/locks/{rocm.lock,rocm.state}`. Back up changed host files to the Mac first.
Existing self-check safety assertions stay; obsolete mirror expectations become
assertions that the retired state files are absent. No files in tests/ change.

Verification: Bash syntax and expanded runnable self-check with stubbed vLLM;
normal/failing commands, cancellation, dead wrapper/live worker, fail-closed
state parsing, concurrent serve/train refusal, restart-after behavior. Then a
real `gpu train wake sleep 30` cancellation and a harmless LFM container command,
followed by vLLM health/authenticated inference and Studio HTTP checks. No real
training or corpus generation. Review all remaining active legacy callers.

### Task 4 applied and checked — 2026-09-23

The GPU command now serializes state reads/writes and cleanup, records both
wrapper and worker PIDs, and fails closed on malformed state or unsuccessful
container cleanup. Normal cancellation stops the wake process group; LFM
failure/cancellation restarts Unsloth before release. A hard-killed wrapper's
live worker continues to block serving until it exits. Training commands must
stay in the foreground. All real runs use scripts; Studio is exploration-only.

- Expanded self-check passes: normal and failing commands, concurrent serve/train
  refusal, stop preserving a reservation, cancellation, hard-killed owner/live
  worker, stale PID recovery, malformed state, failed LFM cleanup, serve-after.
- Live `gpu train wake sleep 30` stopped vLLM, rejected `gpu serve`, and returned
  idle after SIGTERM with its worker gone.
- Live harmless LFM command saw the model/datasets and released the reservation.
  Cancelling an LFM sleep with `--serve-after` restarted Unsloth and started vLLM.
- Legacy launch scripts and `run/locks/` removed after a Mac backup in
  `~/SaySo-archive/vm-20260923/task4/`. `bin/try-config.sh` now calls `bin/gpu`
  and stops on rejected serving. No active legacy callers remain.
- Both training READMEs and the wake config comment use the GPU command; docs
  explain cancellation and the Studio constraint. No files in tests/ changed.

Final service verification: vLLM `/health` and Studio returned HTTP 200;
authenticated vLLM completion returned `OK`. Task 4 is complete. Both services
are running, GPU state is `serve`, and no test training workers remain.

## Crash-recovery correction — reviewed before implementation

The review reproduced unsafe release after both recorded PIDs die while a
wake descendant or container-side LFM process survives. Correct Task 4 rather
than treating the earlier self-check as sufficient.

Scope/files: `scripts/llm-host/gpu`, its colocated self-check, this plan, and
both training READMEs. Preserve existing assertions and add regression cases.
Wake liveness must include non-zombie process-group members. A stale LFM
reservation must remain visibly pending; mutating commands restart Unsloth
under the transition lock before allowing reuse. Failed cleanup stays blocked.
Status stays read-only. Verify group termination before normal release too.

Verification: Bash syntax, existing self-check plus dead leader/live descendant
and stale LFM cleanup success/failure cases; repeat the isolated real-Docker
reproduction with no GPU devices or production mounts. Then deploy matching
files, check checksums and production health. No production restart is needed.

Launch-window follow-up: persist an explicit starting reservation before spawning
the worker. If the wrapper dies before publishing its worker PID, the reservation
stays blocked for inspection instead of being mistaken for a dead training run.
Add a refusal assertion for this incomplete-launch state.

### Crash-recovery correction verified — 2026-09-23

- Existing self-check assertions and added regressions pass. A live wake
  descendant blocks both serving and new training after owner/leader death;
  the reservation clears only after the group stops.
- Stale LFM status is read-only. Serve, stop, and train perform container cleanup
  first; failed restart leaves a persistent blocked state and never starts vLLM.
- An isolated real Docker container (no GPU devices, network, or production
  mounts) reproduced the surviving worker. With the fix, status retained the
  orphan reservation and serving proceeded only after restarting that container
  removed the worker. The test container was removed.
- A starting reservation prevents unsafe release in the spawn/PID-publication
  window. Incomplete launches require inspection; normal cleanup verifies the
  worker group has stopped before marking idle.
- Correction deployed without restarting vLLM or Unsloth; production files
  match the tested source. Earlier PID-only completion evidence was insufficient
  for crash recovery; this correction and its regressions supersede it.
