# Agent notes

Prioritize completing the end-to-end voice path over future-proofing. Do not
cut ControlPlan validation, ambiguity handling, integration execution,
verification, metrics, or basic evals. Defer extensive model benchmarking,
large eval datasets, generalized satellite support, fine-tuning hooks,
streaming optimizations, and polished diagnostics if they threaten completion.

## How to apply that

The MVP succeeds when a Mac can act as a temporary smart speaker and control
real Home Assistant devices. Prefer the next increment that unblocks
wake → capture → STT → ControlPlan → resolve/validate → HA execute → verify →
response.

Keep (do not skip to “save time”):

- ControlPlan validation and the existing safety/ambiguity/capability barriers
- Integration execution and state verification
- Metric scoring and a runnable basic eval path
- Core/safety/follow-up eval coverage already in the plan

Defer until the voice path works end to end:

- Extra model bake-offs (LFM vs Home-FunctionGemma, Home-LLM, Alexa+)
- Expanding corpora past the basic reviewed sets
- Generalized multi-satellite support beyond the Mac living-room satellite
- Fine-tuning hooks, streaming optimizations, and polished diagnostics

## Workflow

- Read `docs/ARCHITECTURE.md` before changing runtime wiring or assuming the
  MVP topology is already assembled. It documents what is implemented,
  partial, planned, conflicting, and unresolved.
- Source of truth for remaining numbered units: `docs/MVP_PLAN.md`. Phase 4
  eval tasks: `docs/EVALUATION_PLAN.md`. Prefer the next increment that
  `docs/ARCHITECTURE.md` lists as required for the first physical-device demo.
- Python throughout. Colocated `test_*.py`. Never create a `tests/` directory.
- One numbered TDD unit at a time unless the user already authorized continuing.
- Do not start a `tests/` directory. Do not commit `context.json`.
