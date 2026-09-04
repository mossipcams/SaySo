# First training

**Status:** recipes 1–8 locked (thermostat omitted). 10k train JSONL generated locally (gitignored). 38-row recipe-lock quality eval generated (`training/scripts/generate_recipe_lock_eval.py`). Base SFT running on `ubuntu@ai-inference` GTX 1070 — output `/srv/training-runs/SaySo-LFM2.5-230M-Base-First`, config `/srv/training-runs/sayso-lfm-base-first.yml`; rank 32 FP16, accum 16, 3 epochs, 1875 steps. EVA/rsLoRA not enabled on this TRL pin (Kaiming-A fallback).

This is the first supervised run on `LiquidAI/LFM2.5-230M-Base`. The contract in [TRAINING_PLAN.md](TRAINING_PLAN.md) still holds: OpenAI `messages` + `tools` on disk, v1 schema labels, no Home-LLM `<tool_call>` text, no `evals/cases/` IDs. This document only defines the first Base run.

Held-out baseline: `/srv/training-runs/eval_heldout_base.json` (2.9% exact). Promote only if this run beats that file on exact/name/schema without collapsing no-call behavior to the Instruct LoRA’s 0%.

## Why Base

`LFM2.5-230M` is instruction-tuned. `LFM2.5-230M-Base` is the pre-train checkpoint Liquid documents for domain SFT. Base already emits ChatML `<|tool_call_start|>` tokens and invents tool names. First training must teach the v1 catalog, canonical entity names, and when not to call.

## Format

```text
canonical JSONL (OpenAI messages, string arguments)
        ↓ tokenizer.apply_chat_template + tools= (dict arguments)
ChatML + <|tool_call_start|>[HassTurnOn(name='…')]
        ↓ rsLoRA SFT on Base
merged FP16 adapter
        ↓ llama.cpp convert + Q8_0
GGUF served with --jinja
        ↓ SaySo HTTP still sends OpenAI messages, not ChatML
```

Do not put ChatML in the dataset or in the Home Assistant integration. Parse `function.arguments` to objects only while applying the Base `chat_template.jinja`.

## Stack (GTX 1070 8 GiB)

```text
LFM2.5-230M-Base
  → FP16 (no BF16, no Flash Attention)
  → rsLoRA
  → EVA initialization on a calibration subset
  → rank 32 first; 64 only if the smoke step has VRAM headroom
  → gradient checkpointing if rank 32 still OOMs
  → effective batch via gradient accumulation (microbatch 1, accum 16 → 32)
  → merge adapter into Base
  → GGUF Q8_0 for llama.cpp --jinja
```

Host weights: `/srv/models/LFM2.5-230M-Base`. Output: `/srv/training-runs/SaySo-LFM2.5-230M-Base-First`.

Smoke before the full run: one forward/backward/update, EVA calibration completes, checkpoint reloads, one llama.cpp tool-call round-trip with `--jinja`. If EVA is unsupported on this PEFT/TRL pin, record the fallback (Kaiming A) and continue; do not block the run on EVA.

## Data gate

Labels are written before language. A human locks the recipes in the next section. Only then expand with the existing label-first generator (`training/scripts/build_synthetic_dataset.py`). The first curated train set is **10,000** rows: generate, judge, and curate to that size with the locked category mix below. Do not reuse stale candidate checkpoints from earlier oversized runs.

Mix (percent of the 10k curated train set):

| Category | Share | Job |
|---|---|---|
| `clean_direct` | 10% | Catalog: one named device, one v1 tool |
| `conversational` | 15% | Same labels, natural voice phrasing |
| `entity_identity` | 10% | Alias, casing, apostrophe, similar names |
| `multi_action_exclusion` | 20% | Two or three calls; named exclusion stays uncalled |
| `stt_corrupted` | 15% | Mild ASR; label is the canonical entity |
| `status` | 10% | `GetLiveContext` on a named device; not an action |
| `ambiguity` | 10% | Generic noun: default to the **SaySo entity’s area**; clarify only if that area (or a named area) has two+ matches |
| `unsupported_no_action` | 10% | Media, safety, incomplete; **not** thermostat |

Thermostat is not an “unsupported” recipe. v1 currently has no climate tool, so first training does not label “set the thermostat” at all until the pinned contract grows. Smoke/refuse/incomplete/media stay.

Cap `clean_direct` at 10%. Do not fill the set with easy commands to chase exact match. Keep contrastive trios (specific action / status / generic noun) in one split family. Contrastive generic labels follow recipe 7 in the SaySo entity area only: duplicate same-type devices for clarify must live in `sayso_entity_area`, never in another area.

Held-out eval stays `/srv/datasets/sayso_test_balanced.jsonl`. Train must not contain those prompts.

## Recipe lock

Line-by-line votes (2026-09-04):

| # | Vote | Note |
|---|---|---|
| 1 | yes, yes | locked |
| 2 | yes, yes | locked |
| 3 | round 2 all yes | locked (names in both columns) |
| 4 | round 2 all yes | locked |
| 5 | round 2 all yes | locked; every STT label names the entity |
| 6 | round 2 all yes | locked |
| 7 | round 3 a–i all yes | locked; default to the SaySo entity’s area; zero matches = no call + say none available |
| 8 | **no**, yes, yes, yes | locked; thermostat omitted from first training, not a refusal |

### 1. clean_direct — locked

| Utterance | Label |
|---|---|
| Turn on Office Main Light | `HassTurnOn` `{name: Office Main Light, domain: [light]}` |
| Close Kitchen North Garage Door | `HassTurnOff` `{name: Kitchen North Garage Door, device_class: [garage]}` |

### 2. conversational — locked

| Utterance | Label |
|---|---|
| Hey, when you get a chance, turn on the living room ceiling fan. | `HassTurnOn` `{name: Living Room Ceiling Fan, domain: [fan]}` |
| Could you set the kitchen herb garden cool light to 64 percent for me? | `HassLightSet` `{name: Kitchen Herb Garden Cool Light, brightness: 64, domain: [light]}` |

### 3. entity_identity — locked

Round 1 alias/casing yes. Round 2 a–h yes. Apostrophe rows keep the real `name` in both the utterance and the label (`Joe's Kitchen Light`, `O'Malley's Study Blinds`, `Kids' Room Light`, `Joe's Guest Room Door Lock`).

| Row | Utterance | Label |
|---|---|---|
| a | Open the patio blinds | `HassTurnOn` `{name: Patio South Blinds, device_class: [blind]}` |
| b | Lock the patio door | `HassTurnOn` `{name: Patio Side Door Lock, device_class: [door]}` |
| c | turn on office main light | `HassTurnOn` `{name: Office Main Light, domain: [light]}` |
| d | unlock joe's guest room door lock | `HassTurnOff` `{name: Joe's Guest Room Door Lock, device_class: [door]}` |
| e | Turn on Joe's Kitchen Light | `HassTurnOn` `{name: Joe's Kitchen Light, domain: [light]}` |
| f | Close O'Malley's Study Blinds | `HassTurnOff` `{name: O'Malley's Study Blinds, device_class: [blind]}` |
| g | Turn off Kids' Room Light | `HassTurnOff` `{name: Kids' Room Light, domain: [light]}` |
| h | Turn on Kitchen North Light | `HassTurnOn` `{name: Kitchen North Light, domain: [light]}` |

### 4. multi_action_exclusion — locked

Round 1 yes. Round 2 a–d yes. Named exclusion stays uncalled.

| Row | Utterance | Label |
|---|---|---|
| a | Set Kitchen Ceiling Cool Light to 40 percent and turn off Hallway East Outlet, but leave Office Main Light alone | `HassLightSet` `{name: Kitchen Ceiling Cool Light, domain: [light], brightness: 40}` + `HassTurnOff` `{name: Hallway East Outlet, domain: [switch]}`; Office Main Light uncalled |
| b | Open Patio South Blinds and lock Patio Side Door Lock, but leave Garage West Fan alone | `HassTurnOn` `{name: Patio South Blinds, device_class: [blind]}` + `HassTurnOn` `{name: Patio Side Door Lock, device_class: [door]}`; Garage West Fan uncalled |
| c | Turn on Nursery East Outlet and turn off Living Room Ceiling Fan, but leave Joe's Kitchen Light alone | `HassTurnOn` `{name: Nursery East Outlet, domain: [switch]}` + `HassTurnOff` `{name: Living Room Ceiling Fan, domain: [fan]}`; Joe's Kitchen Light uncalled |
| d | Open Joe's Workshop Blinds, close Primary Bedroom Corner Garage Door, and lock Patio Side Door Lock, but leave Garage Ceiling Fan alone | `HassTurnOn` `{name: Joe's Workshop Blinds, device_class: [blind]}` + `HassTurnOff` `{name: Primary Bedroom Corner Garage Door, device_class: [garage]}` + `HassTurnOn` `{name: Patio Side Door Lock, device_class: [door]}`; Garage Ceiling Fan uncalled |

### 5. stt_corrupted — locked

Round 1 “garage van” → **Garage West Fan** yes. Round 2 a–e yes. Every label names the canonical entity.

| Row | Utterance | Label |
|---|---|---|
| garage_van | Turn on the garage west van | `HassTurnOn` `{name: Garage West Fan, domain: [fan]}` |
| a | Uh unlock basement door lok please | `HassTurnOff` `{name: Basement South Door Lock, device_class: [door]}` |
| b | tern on office main lite | `HassTurnOn` `{name: Office Main Light, domain: [light]}` |
| c | close the patio south blends | `HassTurnOff` `{name: Patio South Blinds, device_class: [blind]}` |
| d | lok joe's guest room door | `HassTurnOn` `{name: Joe's Guest Room Door Lock, device_class: [door]}` |
| e | turn off basement van | `HassTurnOff` `{name: Basement South Fan, domain: [fan]}` |

### 6. status — locked

Round 1 yes. Round 2 a–d yes. `GetLiveContext` on the named device, never `HassTurnOn`.

| Row | Utterance | Label |
|---|---|---|
| a | Check the status of Patio South Blinds | `GetLiveContext` `{name: Patio South Blinds, device_class: [blind]}` |
| b | Is the Workshop West Fan running? | `GetLiveContext` `{name: Workshop West Fan, domain: [fan]}` |
| c | What's Joe's Guest Room Door Lock doing? | `GetLiveContext` `{name: Joe's Guest Room Door Lock, device_class: [door]}` |
| d | Is Kitchen North Light off? | `GetLiveContext` `{name: Kitchen North Light, domain: [light]}` |

### 7. area default — locked

Round 2 house-wide clarify examples all no. Round 3 a–i yes.

The default area is the Home Assistant area of **this SaySo conversation entity** (`conversation.sayso_*`), not the assist satellite device and not a house-wide guess. Training context states that area next to the exposed entities.

- No area in the utterance → use the SaySo entity’s area.
- Area named in the utterance → use that area, ignore the SaySo entity’s area.
- Clarify only when the **resolved area** has two or more matching entities of that type.
- Resolved area has **none** of that type → no call; spoken next action states that this area has none of that type available.

| | SaySo entity area | Utterance | Also in home | Label |
|---|---|---|---|---|
| a | Kitchen (one light: Kitchen Sink Cool Light) | Turn on the light | Office Main Light | `HassTurnOn` `{name: Kitchen Sink Cool Light, domain: [light]}` |
| b | Kitchen (two lights: Kitchen Sink Cool Light, Kitchen Ceiling Cool Light) | Turn on the light | Office Main Light | no call, clarify |
| c | Kitchen | Turn on the office light | Office Main Light only in Office | `HassTurnOn` `{name: Office Main Light, domain: [light]}` |
| d | Kitchen (two kitchen lights) | Turn on the kitchen light | — | no call, clarify |
| e | Living Room (one fan: Living Room Ceiling Fan) | Turn off the fan | Workshop West Fan | `HassTurnOff` `{name: Living Room Ceiling Fan, domain: [fan]}` |
| f | Hallway (two outlets: Hallway East Outlet, Hallway West Outlet) | Turn on the outlet | Nursery East Outlet | no call, clarify |
| g | Patio (one cover: Patio South Blinds) | Open the blinds | Joe's Workshop Blinds | `HassTurnOn` `{name: Patio South Blinds, device_class: [blind]}` |
| h | Kitchen (one lock: Kitchen Back Door Lock) | Lock the door | Patio Side Door Lock | `HassTurnOn` `{name: Kitchen Back Door Lock, device_class: [door]}` |
| i | Kitchen (no lights) | Turn on the light | Office Main Light | no call; next action: “The kitchen has no lights available.” |

### 8. unsupported_no_action — locked

| Utterance | Vote | Label |
|---|---|---|
| Set the thermostat to 72 degrees | **no** | not used; not a refusal |
| Disable the smoke alarm safety system | yes | no call, refuse |
| Set the light to | yes | no call, clarify |
| Play music in the garage | yes | no call, unsupported |


## Eval

After merge + GGUF, run the same held-out script as Base (`eval_heldout_base.py` against `sayso_test_balanced.jsonl`). Compare to `eval_heldout_base_summary.json`. Watch:

- normal household / conversational / multi-action exact (Base is 0%)
- schema-valid args (Base 47%)
- invented tool names must drop
- `no_call_when_expected` must not return to 0%

llama.cpp verification uses `--jinja` and the merged Q8_0, not Transformers-only scores.

## Out of scope

Instruct checkpoint, the deleted 2500 LoRA, Home-LLM labels, lowering judge floors to hit row count, ChatML in the integration.
