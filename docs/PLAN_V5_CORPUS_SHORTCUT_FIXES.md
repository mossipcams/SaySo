# v5 corpus: remove the catalog-shape shortcut and the coverage gaps it exposed

Date: 2026-09-24. Scope: generator changes plus a v5 regeneration. The recipe,
eval, and runtime are unchanged. The v5 full-SFT run was stopped at step ~2650
(checkpoint-2400 scored 11/24 on smoke).

## Findings (40k v5 corpus audit + checkpoint-2400 probes)

1. **Catalog shape predicts refusal.** Only `unavailable` rows withhold a tool,
   and homes carry 0, 1 or 3 scripts, never 2. Result: 17, 23 or 29 offered tools
   → 100% refusal, 14 → 72%, 27 → 79%. The eval's production catalog is 29 tools
   (27 + 2 scripts). Probes: removing script tools, or adding `"required": []` to one,
   flips a call into a refusal and back. Target names and casing had no effect.
2. **Status is templated only.** Every status request is "status/state of X". There are
   no yes/no questions ("Is the front door locked?", "Did I leave X on?"), which are
   what both smoke status cases ask. The model answered one with `HassTurnOn`.
3. **Exclusion never scopes a group.** All 1,868 exclusion rows name each target
   ("turn off A and turn off B, but leave C alone", C often in another room). The eval
   asks "turn off the kitchen lights, but leave X alone".
4. **Vacuum start phrasing** is only "start X" (~55 rows). The eval's "have X start
   cleaning" drew `HassVacuumCleanArea`.
5. **Text bugs:** 400 utterances end in "?.". STT `log_trailing_period` appends
   "." after "?".

## Changes

- `generators/homes.py`: scripts per home drawn from 1–4 (was `max(1, size//20)`).
- `generators/rendering.py`: one `_offered_catalog` helper for the three render sites.
  A row with no deliberate removal withholds 1–2 tools it does not need with p=0.3.
  Needed means called anywhere in the row, or the tool of its requested operation.
  The candidate set spans all tools, like `unavailable` removals do.
- `generators/utterances.py`: status seeds use a domain-appropriate yes/no question
  half the time; `HassVacuumStart` gets varied phrasing; `vary_training_utterance`
  leaves questions ending in "?" alone.
- `generators/validation.py`: a status utterance may be a question (starts with
  is/are/did/does/do/what/how or ends with "?").
- `generators/scenarios/core.py` + `utterances.py`: 60% of exclusion rows scope an
  area: targets are every other entity of the capability in one area, the excluded
  entity is in the same area, spoken as "the {area} {plural}".
- `generators/stt_noise.py`: no trailing period after "?" or "!".

### Second audit pass (label correctness)

6. **Clarify when a device was named exactly:** ~10% of clarify/follow-up rows
   (831 of 7,846) ask "did you mean" even though the request says one candidate's
   exact multi-word name ("turn on the kitchen tv" with "Kitchen tv" present).
   `validation.py` now rejects these as `clarify_target_named`; a shared alias
   still counts as ambiguous.
7. **Real-home whitespace:** `switch.living_room_lights` is named `"Living room lights "`.
   The label kept the trailing space, but the prompt doesn't show it. `real_home._load`
   now normalizes whitespace.
8. **Status replies echoed raw HA states:** "X is 2026-09-22T23:21:49+00:00." and
   "X is 2.". They now read "was last activated at HH:MM" and "has N items".
   Absence replies use one area casing.
9. **Timers were always 5–30 minutes;** the eval uses hours and seconds. Start and
   adjust timers now vary the unit, with singular units ("1 hour").

Flagged, not changed: lock calls carry `device_class: ["door"]` (the eval cases
disagree among themselves; needs a check against live HA). Timers, climate, scripts
and vacuums are each 1–2% of rows vs ~8% per eval category (a recipe decision).

## Verification

- `pytest training/tests -q` green (plus new tests for each change).
- Smoke recipe dry-run, then the full v5 regeneration on the VM (CPU).
- Re-run the corpus audit: no `n_tools` value above 90% refusal, no "?.", status
  yes/no share ~50%, exclusion rows with area scope present.
- Retrain, evaluate a mid-run checkpoint on CPU (`eval_checkpoint_cpu.py`), and
  repeat the catalog-shape probes.
