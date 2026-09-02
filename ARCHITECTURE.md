# SaySo Architecture

SaySo is a fully local Home Assistant conversation agent. Home Assistant remains the
only application runtime; llama.cpp is an external user-managed inference process.

## Runtime topology

```text
Standard HA satellite
        |
        v
Home Assistant voice pipeline
wake word -> STT -> SaySo ConversationEntity
                         |
                         v
                 llama.cpp HTTP API
                         |
                         v
              HA LLM tool selection
                         |
                         v
             HA intent/action execution
                         |
                         v
             SaySo ConversationResult
                         |
                         v
                  HA TTS -> satellite
```

## Boundaries

SaySo does **not** own:

- Audio transport
- Wake-word detection
- Speech-to-text (STT)
- Text-to-speech (TTS)
- Satellite software
- Model hosting (llama-server lifecycle)
- A separate backend, daemon, add-on, sidecar, broker, worker, or custom API

Home Assistant owns voice pipeline orchestration, satellite integration, conversation
routing, entity exposure, smart-home actions, conversation context, and returning audio
to the satellite. llama.cpp performs inference only via its OpenAI-compatible HTTP API.

## Integration shape

- Domain: `sayso`
- Package: `custom_components.sayso`
- Config-entry based; conversation entity registers as a Home Assistant conversation agent
- Direct HTTP from Home Assistant to llama.cpp; no SaySo-specific wire protocol

See `docs/PLAN.md` for the full product specification and build sequence.
