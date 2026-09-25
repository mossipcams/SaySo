# Aggressive SFT corpus — every knob

> Historical rejected proposal. Its allocations and cross-cut rates are not the
> active `full_sft_v5.yaml` recipe. Use the
> [SaySo training lifecycle](SAYSO_TRAINING_LIFECYCLE.md) for current commands.

First-turn-only v4 is rejected. The previous mix that left paraphrase off, grounding
at 20%, real-home at 10%, and “labels conservative” is also rejected. **Aggressive
means every axis:** families, graphs, noise, homes, history, tools, context
length, and how hard we hit *this* house’s 21 Assist names. We do not add HA
entities. We do not lower a share because generate is hard.

No training launch here. No eval/runtime edits.

## 1. Primary mix (40,000 rows, sum 1.00)

| Family | Share | Rows |
|---|---:|---:|
| `follow_up` | 0.20 | 8,000 |
| `ordinary` | 0.18 | 7,200 |
| `clarify` | 0.08 | 3,200 |
| `status` | 0.08 | 3,200 |
| `settings` | 0.07 | 2,800 |
| `junk` | 0.06 | 2,400 |
| `aliases` | 0.06 | 2,400 |
| `correction` | 0.06 | 2,400 |
| `multi_action` | 0.05 | 2,000 |
| `exclusion` | 0.05 | 2,000 |
| `unavailable` | 0.04 | 1,600 |
| `unsupported` | 0.04 | 1,600 |
| `absence` | 0.03 | 1,200 |

Datetime ≥800 positives **inside** ordinary/status. Per-tool floor **400**.
Per operation **80**. Script tools: every exposed script name gets positives.

### Follow-up 8,000 (3-turn default, not 2)

- 3,500 clarify → “the shelf lamp” → call
- 1,500 user correction → “no, the TV” → call
- 1,500 additive → “brighter” / “and the kitchen” with one live target
- 800 cancel → “stop” / “never mind” → no call
- 700 still-ambiguous second turn → **another named question**, not a guess

Stack STT and bare-name on these. History is in the prompt; only the last
assistant turn is `train_on_turn` unless the first-turn clarify also needs
loss — then export **two views** of one conversation (plan §3).

### Correction 2,400 (pre-execution only)

- 1,000 `Living room TV` → `name="TV"`
- 800 wrong tool / missing domain
- 600 missing required setting

Bad call is never a loss target. No post-execution `MatchFailedError` retry.

## 2. Cross-cuts — push until the catalogue breaks, then grow the catalogue

| Knob | Aggressive value | What “grow the catalogue” means |
|---|---|---|
| `grounding.rate` | **0.25** (10,000 rows) | Add sites, domains (media, fans, switches, climate), occupancy pairs. Raise `near_duplicate_limit` to **16** so capacity ≥ 10k. |
| `discrimination.rate` | **0.05** | Extend `DISCRIMINATION_CAPABILITIES` to media/fans/climate; kill the 0.02 ceiling or replace it with measured capacity after the expansion. |
| `bare_name_rate` | **0.50** | Half of synthetic names are bare. Gold never concatenates. |
| `stt_log_rate` | **0.10** | Dedicated slice from live faster-whisper: `living room`→`ribbon room`/`librarian`, dropped articles/fillers, homophones, punct/casing. Gold unchanged. |
| `stt_noise_rate` | **0.15** | Additional recoverable noise (numbers, `T V`) on rows that did not take the log slice. Relabel when off/target/intent dies. |
| `real_home.rate` | **0.20** | 8,000 rows on 17 train entities. Hit TV, living-room lamps, Echo-no-power, ugly names (`Porch_Light`, trailing space) with follow_up+STT+correction **stacked**. Entity cap multiplier **8**, not 4. Holdout stride stays 5. Synthetic still fills leftover ordinary. |
| `paraphrase_enabled` | **true** | OHF + paraphrase. Drop the row if gold would change. |
| Area distribution | **2×** every `per_1000` | Implicit satellite, duplicate-name clarify, conflict-with-satellite, exclusion-in-area. |
| Home size | **bias large** | Enough 32–64 entity homes that ~10% of rendered rows land 6145–8192 tokens. Full catalog every row. |
| `similar_name` | **wired, not a no-op** | Near-miss names in-graph → named clarify or follow_up, never auto-pick. |

## 3. What every row must teach (stack, don’t pick one)

On a living-room TV slot, the same semantic should appear as: bare `TV`, spoken
“living room TV”, STT `T V`, alias `television` if we add it only in **synthetic**
homes, concatenated-name **correction**, and follow_up “the TV”. Real-home gold
name stays `TV`. Lamps: “the lamp” → named clarify → “the corner one” → call.

Status: tool result payload is real-shaped; speech quotes it. Never “Done.”
Settings: missing number is clarify. Unsupported: Echo/Party Time never get
`HassTurnOn`. Junk: destroyed STT is junk, not the original on/off.

## 4. Generator work (all of it, before generate)

- New families `follow_up`, `correction` in `planning.py` / scenarios / rendering
- Grounding catalogue expansion + `near_duplicate_limit` 16
- Discrimination beyond lights/covers/switches; remove stale 0.02 ceiling
- STT relabel + `refers_to` normalize
- `similar_name` graph handler
- Paraphrase gate
- Area YAML doubled
- Real-home cap multiplier 8; leftover ordinary → synthetic retry (already)
- `full_sft_v5.yaml` with this recipe; v4 YAML stays as the rejected baseline
- Tests for: stacked TV views, 3-turn follow_up, relabeled STT, Echo no power,
  paraphrase-does-not-flip-gold, grounding 25% on an 800-row probe

## 5. Feasibility rule

If 40k cannot fill, **expand identity** (more grounding sites, more synthetic
homes, higher near-dup) or retry synthetic. **Forbidden:** cutting follow_up,
grounding, correction, STT, real-home rate, or tool floors.

## 6. Verification

1. `verify_feasible` on v5.
2. 2,000-row probe: follow_up ~400, grounding ~500, real_home ~400, STT ~600,
   named clarify in text, zero action on junk, `name="TV"` on TV gold, Echo
   turn_on is unsupported/clarify not a call.
3. Schema round-trip every supervised call.
4. 40k → `training/datasets/sayso_full_sft_v5_<date>.jsonl` + manifest.

Kill if no jsonl. Audit before train.
