# Historical VM layout migration (2026-09-23)

This plan records the completed move of SaySo generation, model training, and
serving assets on the `llm` VM. Its old host paths and migration commands are
historical; use [training/README.md](../training/README.md) for the observed
host layout and [the training lifecycle](SAYSO_TRAINING_LIFECYCLE.md) for the
supported local workflow.

The active services use `/srv/llm/serve/vllm/` for vLLM and
`/srv/llm/services/unsloth/` for Unsloth Studio. The checkout and generated
datasets live on the data disk; model weights, run outputs, and service state
live on the SSD. A shared Hugging Face cache is kept at `/srv/llm/hf-cache/`.
Wake-word training remains a separate LiveKit pipeline.

The migration verified service health, dataset visibility, filesystem
placement, and the generation environment. Detailed host measurements and
run history belong in [training/README.md](../training/README.md) and
[training/TRAINING_LOG.md](../training/TRAINING_LOG.md).
