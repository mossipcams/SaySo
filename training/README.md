# SaySo language model training

Current commands, the active config, and what each eval set is for.

- Stable design and constraints: [docs/SAYSO_LFM_TRAINING_PLAN.md](../docs/SAYSO_LFM_TRAINING_PLAN.md)
- What actually ran and what it scored: [TRAINING_LOG.md](TRAINING_LOG.md)

This directory holds dataset preparation and historical training assets for
`LFM2.5-230M-Base`. v5 is a **full-parameter** fine-tune on the 24 GB AMD
host (`ssh llm`, see [Training host](#training-host-observed-2026-09-23)).
The trainer is **Unsloth** (it replaced LLaMA-Factory on 2026-09-23). The
checked-in `*-llamafactory*.yml` recipes and LLaMA-Factory smoke results below
are history; the Unsloth launcher is `scripts/train_unsloth_full.py`. Canonical JSONL
stays OpenAI-shaped on disk; training consumes a derived prompt/completion
view rendered with the Base `chat_template.jinja`.
Axolotl, FunctionGemma, and rsLoRA YAML are historical, not the v5 path.

SaySo has two models with separate pipelines: this one (LFM, Unsloth) and the
wake word (LiveKit), which has its own guide:
[training/wake/README.md](wake/README.md).

## Training host (observed 2026-09-23)

One VM does generation, training, and LLM serving. SSH alias `llm`
(`LLM@192.168.1.76`, key `~/.ssh/id_ed25519_sayso`). No passwordless sudo;
docker group yes. The VM has no GitHub credentials.

| Item | Observed |
|---|---|
| GPU | Radeon RX 7900 XTX, gfx1100, 24 GiB; host ROCm 7.2.4 |
| CPU / RAM | 8 vCPU; 19 GiB RAM + 12 GiB swapfile |
| Disks | SSD `/` 300 G (Docker images, models, runs; ~60 G free). HDD `/srv/llm/data` 400 G (datasets and generation code; ~361 G free). Layout plan: [docs/PLAN_VM_LAYOUT.md](../docs/PLAN_VM_LAYOUT.md) |
| Repo checkout | `/srv/llm/sayso` → `/srv/llm/data/sayso` (HDD), git tracking `origin/main`, read-only deploy key so the VM can `git pull`. Generation venv: `training/.venv`. |
| Base weights | `/srv/llm/lfm/models/LFM2.5-230M-Base` (SSD; `tokenizer_class` patched to `PreTrainedTokenizerFast`, original kept as `tokenizer_config.json.tf5`) |
| v5 datasets / runs | `/srv/llm/data/lfm/datasets/` (HDD), `/srv/llm/lfm/runs/` (SSD) |
| Training UI | Unsloth Studio container `unsloth` (`unsloth/unsloth-rocm:studio`), UI on host `:8002` (login required). Compose `/srv/llm/services/unsloth/`. Mounts only `/srv/llm/lfm` (rw), `data/lfm/datasets` (ro), `data/sayso` (ro) under `/workspace/host/`; shared SSD `hf-cache/` and SSD `services/unsloth/studio/`; `restart: unless-stopped` |
| LLM serving | vLLM container `vllm`, Qwen3.8-27B-INT4 on `:8000` at 0.98 GPU memory (vLLM v0.30.0, 64k context). Compose `/srv/llm/serve/vllm/docker-compose.yml`, never autostarts. `/srv/llm/bin/gpu serve` / `stop` |
| GPU sharing | vLLM, Unsloth, and LiveKit share one XTX. Every training run starts through `/srv/llm/bin/gpu train lfm|wake` (stops vLLM, holds the GPU, releases on exit); `gpu serve` / `gpu stop` / `gpu status` for vLLM. Source: `scripts/llm-host/gpu`. Runs started by hand in the Studio UI bypass the lock, so only explore there while `gpu status` shows `idle`. |
| Removed 2026-09-23 | LLaMA-Factory and its `llm-rocm` / `llm-cuda` containers and images. The old launchers are removed; retired setup helpers are archived under `data/archive/task3-retired/`. |
| Migrated history | The retired NVIDIA host's `/srv/models`, `/srv/datasets`, `/srv/training-runs` are now `/srv/llm/data/lfm/legacy-140/{models,datasets,training-runs}` (HDD), cleaned 2026-09-23 (see `TRAINING_LOG.md` "Host cleanup"). Champion markers: `legacy-140/CHAMPION.md`. Paths in `TRAINING_LOG.md` and handoffs use the old prefix. |

### Retired host (`192.168.1.140`)

`ssh sayso-train` (`ubuntu@192.168.1.140`, hostname `ai-inference`, GTX 1070
8 GiB) ran every run up to Run 013. It was unreachable on 2026-09-23. It used
`/srv/training-runs/.venv` (Torch `2.13.0+cu126`, Transformers `5.16.1`, TRL
`1.12.0`, PEFT `0.20.0`, xFormers `0.0.35`, Liger Kernel `0.8.2`) with the TRL
launcher `/srv/training-runs/train_lfm2_liger.py`. That environment is not a
dependency lock for v5.

The later host recipes are still rsLoRA rank/alpha 32, batch 1/accumulation 16,
learning rate `2e-4`, and 8,192-token context. They use **FP32, xFormers, and a
custom Liger loss patch**, with checkpoints every 50 steps. Their headers record
Pascal memory/speed limitations behind those choices. The launcher applies both
patches in-process before TRL; a plain CLI invocation is not equivalent.

The proposed ROCm recipe used LLaMA-Factory `finetuning_type: full`, native SDPA,
standard AdamW, and BF16 autocast only after validation on the chosen AMD card.
The old Pascal patches are historical. The model plan defines the native LFM
rendered data view, template-parity checks, and longest-row memory/update gate;
the built-in `lfm2` template is not assumed equivalent to SaySo serving.

Axolotl is not used. `training/scripts/train.py`, `train_lfm.py`, and
`configs/detect_gpu.py` are dead Axolotl launchers. `training/requirements.txt`
is the generator/view/test environment only (CPU, no torch, no trainer). The
GGUF export script prints instructions rather than executing a complete export.

## Quick start

Generator, data-view, and test environment (CPU). On the VM it lives at
`/srv/llm/sayso/training/.venv`:

```bash
cd training
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
HIP_VISIBLE_DEVICES= HF_HOME=/srv/llm/hf-cache .venv/bin/python -m pytest tests -q
```

## Historical checked-in adapter config

| | |
|---|---|
| Checked-in recipe | `training/configs/lfm25-230m-synthetic-v3-40k-trl.yml` |
| Historical host copy | `/srv/training-runs/sayso-lfm-v3-40k.yml` — absent at the 2026-09-22 inspection |
| Historical launcher | `/srv/training-runs/run-v3-40k-train.sh` — absent at the 2026-09-22 inspection |
| Settings | from Base, rsLoRA rank 32 / alpha 32, `all-linear`, FP16 (no BF16, no flash-attn), microbatch 1, accum 16, lr 2e-4 cosine, assistant-only loss, 2 epochs, `save_strategy: epoch` |

The observed later launchers guard the row count and run a 12-step smoke before
starting training. Their existence does not establish a safe full-SFT launch.

## Pipeline commands

These describe the existing generator. The production recipe is still v3,
with a 5,120-token budget, grounding disabled, 2% requested discrimination,
and synthetic-only data. It does not yet implement the new corpus plan.

The canonical generator CLI accepts only `--config` (YAML recipe under
`training/configs/generation/`) and `--dry-run` (generate in memory, no writes).
Mixing knobs — real-home rate, grounding/discrimination shares, area
distribution, STT noise — live in the recipe, not on the command line.

| Step | Command |
|------|---------|
| Generator smoke (recipe self-test) | `cd training && python -m generators.cli --config configs/generation/smoke.yaml` |
| Generator smoke (dry-run, no files) | `cd training && python -m generators.cli --config configs/generation/smoke.yaml --dry-run` |
| Build production corpus (40k recipe) | `cd training && python -m generators.cli --config configs/generation/production.yaml` |
| Build the v4 full-SFT corpus (40k recipe) | `cd training && python -m generators.cli --config configs/generation/full_sft_v4.yaml` |
| Generate balanced held-out test set | `python training/scripts/generate_balanced_test_data.py` |
| Split 80/10/10 | `python training/scripts/split_dataset.py INPUT.jsonl --out-dir training/datasets` |
| Fetch the real Home Assistant home | `HA_URL=... HA_TOKEN=... python training/scripts/fetch_ha_home.py --out training/fixtures/real_home.json` |
| Detect GPU | `python training/scripts/detect_gpu.py` |
| Export GGUF | `python training/scripts/export_gguf.py --checkpoint PATH --dry-run` |
| Verify llama.cpp | `python training/scripts/verify_llamacpp.py --dry-run` |
| Export LlamaFactory view (local) | `cd training && python scripts/export_llamafactory_view.py datasets/INPUT.jsonl -o datasets/OUT_rendered.jsonl --dataset-name sayso_v5_rendered --dataset-info-out datasets/dataset_info.fragment.json` |

## v5 full SFT on the ROCm host (`ssh llm`)

Host layout (outside git):

| Path | Role |
|---|---|
| `/srv/llm/lfm/models/LFM2.5-230M-Base` | Base weights (SSD) |
| `/srv/llm/data/lfm/datasets/sayso_full_sft_v5.jsonl` | Canonical corpus (HDD; not generated yet) |
| `/srv/llm/data/lfm/datasets/sayso_full_sft_v5_rendered.jsonl` | Derived prompt/completion view (HDD; not generated yet) |
| `/srv/llm/data/lfm/datasets/dataset_info.json` | LLaMA-Factory registry (history); holds `sayso_v5_rendered_smoke` |
| `training/configs/lfm25-230m-full-24gb-rocm-llamafactory*.yml`, `training/configs/history/` | LLaMA-Factory recipes (history; they name the pre-reorg paths) |
| `/srv/llm/lfm/runs/llamafactory-smoke/` | LLaMA-Factory HIP smoke trainer logs + `rocm-train.log` (weights deleted) |

Generate the corpus on the VM, not the Mac, from `/srv/llm/sayso/training`,
with the XTX hidden (`HIP_VISIBLE_DEVICES=`), using `training/.venv` (Quick start).

```bash
cd training
python -m generators.cli --config configs/generation/full_sft_v5.yaml
python scripts/export_llamafactory_view.py datasets/sayso_full_sft_v5.jsonl \
  -o datasets/sayso_full_sft_v5_rendered.jsonl \
  --dataset-name sayso_v5_rendered \
  --dataset-info-out datasets/dataset_info.sayso_v5_rendered.json
```

The rendered view is plain alpaca `instruction`/`output` JSONL, so it is not
tied to LLaMA-Factory. `scripts/train_unsloth_full.py` trains it inside the
`unsloth` container with the same contract (`docs/PLAN_LFM_HOST_V5_SETUP.md`):
full SFT from Base, completion-only loss, 8192 cutoff with zero truncation,
lr `2e-5`, batch 1 / accum 32, bf16, 2 epochs. `--max-steps N` is a smoke on
the longest rows:

```bash
ssh llm '/srv/llm/bin/gpu train lfm --serve-after python /workspace/host/sayso/training/scripts/train_unsloth_full.py --max-steps 20 --out /workspace/host/lfm/runs/smoke'
```

Every real training run goes through the GPU lock, so vLLM is stopped first
and the GPU is released when the run exits:

```bash
ssh llm '/srv/llm/bin/gpu train lfm --serve-after <command run inside the unsloth container>'
ssh llm /srv/llm/bin/gpu status
```

The Unsloth Studio UI can't be forced through the lock. Use it for exploring
only, and only while `gpu status` shows `idle`.

Repo verification:

```bash
cd training && .venv/bin/python -m pytest tests/test_llamafactory_view.py tests/test_lfm_config.py adapters/test_lfm.py -q
```

## Corpus v4 — the defect-driven full-SFT recipe

`configs/generation/full_sft_v4.yaml` is the recipe for the next full-parameter
run. It differs from `production.yaml` in four ways, each traced to an open
model defect rather than to a tuning preference.

| Knob | v3 production | v4 | Defect |
|---|---|---|---|
| `generation.bare_name_rate` | absent (0) | `0.35` | [#94](https://github.com/mossipcams/SaySo/issues/94) |
| `allocations.junk` | absent | `0.06` | [#97](https://github.com/mossipcams/SaySo/issues/97) |
| `coverage.min_positive_per_tool` | `1` | `200` | [#39](https://github.com/mossipcams/SaySo/issues/39) |
| `coverage.get_datetime_positive_min` | `1` | `400` | [#39](https://github.com/mossipcams/SaySo/issues/39) |
| `real_home.rate` | `0.0` | `0.10` | #94 |
| `generation.token_budget` | `5120` | `7168` | see below |

### Entity naming and `MatchFailedError`

Every name `generators.homes` produced was `f"{area} {role}"` — "Living Room TV",
"Kitchen Ceiling Lights". Real Home Assistant registries commonly hold a bare
name plus an area; `training/fixtures/real_home.json` names the failing entity
`TV` in area `Living Room`. Trained only on prefixed names, the model answers
"turn off the living room TV" with `name="Living room TV"`, which matches no
entity, and Home Assistant raises `MatchFailedError`.

`bare_name_rate` is the share of fixtures named without the area prefix. A bare
fixture also loses its `"{area} {noun}"` alias, including on the deliberate
collision entities — handing that string back in the static context would teach
the concatenation again. Deliberate alias collisions ("tv" on two devices) are
kept: they are what makes those rows hard.

### Non-command transcripts

The `junk` family carries false wakes, far-field TV audio, and half-heard speech:
zero tool calls, one short "Sorry, I didn't catch that." Junk rows skip STT
corruption (already corrupted) and the politeness finalizer (STT does not emit
"can you … for me?"), and are excluded from capability quotas because they teach
no capability. Near-miss commands like "turn on the living room TV" deliberately
stay in the ordinary families — putting them here would train the refusals of
issue #39.

### Token counting was dead, and the counts were fiction

`count_row_tokens` failed two ways at once, so no corpus this repo has shipped
was ever measured or length-gated:

1. Canonical rows keep `tool_calls[].function.arguments` as a JSON **string**.
   The LFM2 chat template calls `.items()` on it, so templating raised on every
   row that calls a tool. `validate_token_budget` caught the exception and
   substituted a `len(text) // 4` estimate.
2. Under transformers >= 5 (the training host runs 5.16.1),
   `apply_chat_template(tokenize=True)` returns a `BatchEncoding`, so `len()`
   counted its two keys. Rows without tool calls measured **2 tokens**.

`validate_serialized_row` — the only function that checks the budget — was also
never called by the pipeline. All three shipped 40k manifests report
`token_length: {min: 0, p50: 0, max: 0}` as a result.

Both bugs are fixed and the validator is wired into `generators.row_generation`.
Measured correctly, production-shaped rows run min 2,663 / p50 ~5,400 / max
~6,500 tokens against a ~30-tool catalog. The `7168` budget leaves headroom under
the 8,192 training context. The previous `4096` and `5120` settings never clipped
anything because nothing enforced them; do not read them as evidence that earlier
corpora were short. A row whose count had to fall back to the estimate is now
flagged `metadata._token_length_estimated`.

The v3 build writes the canonical JSONL and the TRL render in one pass and
records `render_rows` in the manifest; it must equal `accepted`. Never hand-filter
the render — dropping rows there shrinks the train set silently.

## Mixing in a real home

`training/scripts/fetch_ha_home.py` pulls a live Home Assistant instance into the
same dict `generators.homes.generate_home` returns: two REST calls, `/api/states`
for entities and `/api/template` for the area/floor map, plus two websocket
commands (`scripts/ha_websocket.py`) for the two facts REST does not carry —
which entities Assist actually exposes, and their aliases. It strips email
addresses from names and entity ids (voice bridges name entities after the linked
account); possessives stay, since they are the apostrophe failure class the
recipe-lock gate tests.

Home Assistant's exposure list is authoritative. An entity it does not expose to
the conversation agent never enters the corpus, whatever its domain, and the
export records which rule it was built under:

| `exposure_source` | Meaning |
|---|---|
| `assist_exposure` | Filtered by Home Assistant's own Assist exposure list. The only valid home-recipe input. |
| `domain_filter` | Websocket unreachable, `--allow-unexposed` used. Marked, and rejected by the recipe. |
| `synthetic_fixture` | `synthetic_reference_home.json`, for tests only. |

`generators.real_home.require_exposure_source` enforces that, so a stale snapshot
fails at build time rather than quietly training on entities the assistant cannot
see. `--allow-stale-home` overrides it deliberately.

Supported actions come from `supported_features`, not from the domain: an Echo
Dot reports no `TURN_ON`, and `media_player.living_room_tv` reports no
`VOLUME_SET`. Generation targets only entities that can actually perform the
operation (`capability_registry.entity_supports`), so the corpus never labels a
call Home Assistant would refuse.

```bash
# refresh the snapshot (required before a real-home recipe)
HA_URL=http://homeassistant.local:8123 HA_TOKEN=... \
  python training/scripts/fetch_ha_home.py --out training/fixtures/real_home.json \
  --require-entity media_player.living_room_tv

# point real_home.path and real_home.rate in the recipe YAML, then build
cd training && python -m generators.cli --config configs/generation/production.yaml
```

Production defaults to `generation.synthetic_only: true` and `real_home.rate: 0.0`.
Set `synthetic_only: false`, a snapshot path, and a nonzero `real_home.rate` in
the recipe to mix a live home in deliberately.

`generators.real_home` holds every fifth entity of each capability out of
training (`split="holdout"`, a capability with one entity stays in train). Those
names never enter the corpus, so scoring on them separates "learned this home"
from "learned homes" — the distinction the synthetic suites cannot make.

One home is a few dozen names, so real rows repeat a small vocabulary. Measured at 10% of
a 10k run: 487 real target labels over 162 distinct names, the most frequent at
1.8x its fair share. `real_home.entity_cap` in the recipe bounds how often one
entity may be the target; 0 derives it as four times the fair share
(`real_home.derive_entity_cap`), which is a safety net rather than an active
constraint at these settings. A capped row is rejected and retried, and the retry
re-rolls the real/synthetic draw, so capping redistributes rows instead of
shrinking the corpus. The manifest records the cap, the row count, and the five
most-targeted entities under `real_home`.

The cap matters most for a capability the real home has only one of — climate,
fan, scene — where every row of that capability lands on the same name. Set
`real_home.entity_cap` explicitly in the recipe to bind it.

The manifest separates requested from achieved mixing (`real_home.requested_rate`
vs `real_home.achieved_rate`) and counts rows, not targets: a row naming three
entities is one real-home row. `real_home.target_counts` holds the per-entity
label counts, and only real entities — an entity injected for a capability the
home lacks is synthetic and does not inflate them.

## Coverage gates

Quotas are accounted on what a row *teaches*, not on its metadata.
`generators/coverage.py` reads the rendered row and reports the outcome
(`action`, `status`, `clarify`, `absence`, `unsupported`), the tool, the domain
and the targeting mode. A row counts as positive supervision for its operation
only when it calls that operation's tool, on that capability's domain, against a
real target. A refusal tagged `media_players/turn_on` fills that bucket's
*negative* allowance and never its positive floor, and merely offering a tool in
the schema is not coverage.

Every bucket therefore carries three numbers: total rows, a positive floor
(`1 - negative_rate` of the bucket), and a negative allowance. Operations Home
Assistant supplies no tool for get an explicit refusal quota instead of a
positive one. Impossible configurations — a count below 1, a negative rate of 1,
tier proportions that do not sum to 1, or a bucket needing positive rows for an
operation with no tool mapping — raise before the generation loop starts rather
than after exhausting `max_attempts`.

`generators/audit.py` then re-derives all of it from the accepted rows and fails
generation when a required operation or tool has no positive row, or when absence
answers exceed `coverage.max_absence_rate` in the recipe (production default 10%).
The manifest records
`positive_by_tool`, `positive_by_operation`, `positive_by_domain_targeting`,
`by_outcome`, `negatives_by_reason` and `absence_rate`, so an absence count is a
distribution to look at rather than a verdict on its own.

`GetDateTime` is always offered in the production catalog and must appear in
`positive_by_tool` when `coverage.get_datetime_positive_min` is set. The
`datetime` scenario family labels clock queries with no entity graph; the audit
fails closed when the minimum is unmet. Tools listed in
`capability_registry.TRAINING_COVERAGE_EXCLUDED` are withheld from declared
coverage instead.

## Grounding families

`generators/grounding.py` holds paired scenarios where the request is fixed and
the entity graph decides the answer — one eligible media player in the area, the
same device renamed, the device moved out of the area, and the same device with
lights, switches and another player added as distractors. Nothing is hard-coded:
each variant is handed to `build_scenario`/`gold_from_scenario`, which derive the
label from the graph and the pinned contract before any wording is applied.
Families also vary aliases, domains and supported actions, and cover individual
and area targeting, genuine ambiguity, and presence/absence pairs.

`grounding.rate` in the recipe sets the share (production v3 uses `0.0` deliberately;
see `docs/SAYSO_LFM_TRAINING_PLAN.md`). A run large enough to fit every required family
fails closed if one is missing. Held-out grounding cases live
in `evals/cases/regressions.jsonl` (tag `grounding`) — the exact Living Room +
`media_player.living_room_tv` → `intent__HassTurnOn(name="TV", domain=["media_player"])`
regression plus variations with different names, ids, areas and distractors.
Their prompts join `evals.cases.excluded_train_utterances()`, so neither the
eval rows nor near-duplicate variants can be trained on.

## Eval sets and what each is for

Canonical cases live in `evals/cases/`. Suites are ID lists in `evals/suites/`.
See `evals/README.md`.

| Set | Rows | Role |
|---|---|---|
| `evals/suites/smoke.yaml` | 24 | Checkpoint selection. Subset of the locked 120. Never includes held-out grounding/gold/shadow/recipe-lock rows. |
| `evals/suites/promotion.yaml` | 120 | Locked realistic promotion gate, 10 per category. Supplemental regressions do not change this denominator. |
| `evals/cases/regressions.jsonl` | unique recipe-lock, quality gold/shadow, and grounding cases | Diagnostic regressions via `--tag`. Held out of training and of smoke. |
| `sayso_test_balanced.jsonl` | 2,500 | Broad held-out regression sample for dataset generation. Not the promotion gate. |

Promotion needs gold and shadow to move the right way together. Gold alone
improving means the benchmark is being overfit.

## Scoring

All model eval goes through `python -m evals.cli run` and the in-memory adapter
for training checkpoints. Serve the checkpoint with `llama-server --jinja`.

```bash
python -m evals.cli run --suite smoke --adapter endpoint --server http://127.0.0.1:8080
python -m evals.cli run --suite promotion --adapter endpoint --server http://127.0.0.1:8080
python -m evals.cli run --tag grounding --adapter endpoint --server http://127.0.0.1:8080
```

Training-time:

```python
from evals.adapters import InMemoryAdapter
from evals.cases import select_cases
from evals.runner import evaluate, write_run

results = evaluate(select_cases(suite="smoke"), InMemoryAdapter(predict))
```

## Dataset views

- **canonical**: OpenAI-compatible envelope with JSON-string `function.arguments`
- **historical TRL render**: dict `function.arguments` for `apply_chat_template` only
- **LlamaFactory view**: native LFM prompt/completion pairs (`instruction` /
  `output`), one per `train_on_turn: true` assistant message; built by
  `adapters/lfm.py` and `scripts/export_llamafactory_view.py` with
  `template: empty` on the host trainer

## Layout

```
training/
  adapters/          Schema validation and LFM helpers
  artifacts/         Checkpoints, eval outputs (gitignored)
  configs/           Axolotl history + `lfm25-230m-full-24gb-rocm-llamafactory*.yml`
  datasets/          Generated JSONL (gitignored)
  fixtures/          Test fixtures
  generators/        Canonical synthetic generation package and only CLI
  scripts/           Training, export, exposure, split, and verification operations
  tests/             Unit tests (no model downloads)
```

## GPU and tokenizer preflight

On the future AMD machine, record `rocminfo`/AMD SMI inventory and require both
nonempty `torch.version.hip` and `torch.cuda.is_available()`; ROCm uses those
PyTorch API names. Validate the chosen GPU/OS/ROCm combination and a real LFM
forward/backward pass. Audit the existing GPU detection script before treating
its NVIDIA-oriented recommendations as applicable to ROCm.
The historical 8 GiB setup needed the host patches above for long rows; its
earlier FP16/SDPA limits are not measurements for a 24 GB GPU.

Count tokens using the pinned tokenizer/template, including schemas and every
turn. Require identical expected/prepared row counts, complete supervised spans,
and zero truncation. LLaMA-Factory's `cutoff_len` can truncate examples; compare
its actual prepared tokens with the native render instead of relying on row
counts alone. Training must fail preflight if the tokenizer or assistant-loss
masks cannot be verified.

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
