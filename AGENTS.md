# Agent notes

Prioritize completing the end-to-end voice path over future-proofing. Do not cut
pre-execution tool validation, fail-closed safety/ambiguity/capability barriers,
HA tool execution, state verification, metric scoring, or the existing basic eval
path.

## How to apply that

The MVP succeeds when a standard Home Assistant voice pipeline can reach SaySo,
call user-managed llama.cpp, execute real device actions, and return speech
through Home Assistant TTS. Prefer the next increment that unblocks a real Home
Assistant device demo.

End-to-end path (Home Assistant owns everything except the llama.cpp HTTP call):

```text
wake word -> STT -> SaySo ConversationEntity -> llama.cpp -> HA LLM tools ->
HA intent/action execution -> ConversationResult -> TTS -> satellite
```

SaySo is a Home Assistant conversation agent (`custom_components/sayso`). It
does not own audio transport, wake word, STT, TTS, satellites, or model
hosting. llama.cpp is user-managed inference only.

Keep (do not skip to “save time”):

- Pre-execution tool validation and the existing fail-closed
  safety/ambiguity/capability barriers (schema routing, argument validation,
  boundary diagnostics, correction retries)
- HA LLM tool execution and post-action verification where the integration
  already checks outcomes
- Metric scoring and the runnable offline eval path under `evals/`
- Core/safety/follow-up coverage already represented in `evals/cases/`
- The 38 recipe-lock golden cases and the shadow eval in `training/`

Defer if they threaten the voice path:

- Extra model bake-offs
- Expanding corpora past the reviewed `evals/cases/` and training gold sets
- Generalized multi-satellite support beyond standard HA voice pipelines
- Streaming optimizations and polished diagnostics

## Workflow

- Read `ARCHITECTURE.md` at the repo root before changing runtime wiring or
  assuming topology. It documents boundaries and the integration shape.
- Offline eval cases and runners live in `evals/` (`evals/cases/`,
  `evals/runner.py`, `evals/scorer.py`, `evals/metrics.py`). Do not train on
  Home-LLM tool-call labels or on eval case IDs from `evals/cases/`.
- Training design lives only in `docs/TRAINING_PLAN.md`. The target is
  `LFM2.5-230M-Base` with schema-conditioned function calling. `ALLOWED_HASS_TOOLS`
  validates the pinned training contract only — it does not define runtime support.
- Python throughout. Extend the existing test suite: `tests/` and colocated
  `custom_components/sayso/test_*.py`. Do not add another `tests/` tree or a new
  test framework.
- One numbered TDD unit at a time unless the user already authorized
  continuing.
- Do not commit `context.json`.

## Architecture alignment

- Read `ARCHITECTURE.md` at the repo root before changing runtime wiring or assuming topology.
- Treat Home Assistant as authoritative for entities, exposure, context, tools, and action execution.
- Keep llama.cpp inference-only; model-generated tool calls must validate against current Home Assistant schemas before execution.
- Keep the SaySo integration as the only bridge between model output and Home Assistant tools.
- Keep the satellite as an optional thin overlay for local wake detection and edge audio; it must not perform language understanding, model inference, or actions.
- Do not introduce a central SaySo server, broker, custom action protocol, or direct satellite-to-llama.cpp connection.
- Update `ARCHITECTURE.md` only when ownership, runtime communication, trust boundaries, or an architectural invariant changes.
