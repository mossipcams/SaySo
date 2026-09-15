# Plan: Context Area and Action Acknowledgement

## Scope
- Resolve the request area from Home Assistant `LLMContext.device_id`, falling back to `ConversationInput.satellite_id` through the entity registry when needed.
- Use the same resolved satellite device for area-aware schema routing preferences.
- Inject `area=<name>` into the model system context when an area resolves.
- After one successful HA action tool call, return a deterministic acknowledgement without a second LFM completion; preserve follow-up inference for batches and non-action tools.
- Classify action tools from Home Assistant tool types, including namespaced and integration action tools.
- Preserve validation, correction, fail-closed execution, tracing, and failure handling.

## Files to touch
- `custom_components/sayso/conversation.py`: context enrichment and successful-action response path.
- `custom_components/sayso/routing.py`: satellite-aware routing preferences.
- `tests/test_conversation.py`: call-count and batch regression expectations.
- `custom_components/sayso/test_schema_identity.py`: focused action acknowledgement coverage.

## Verification
- Run syntax/compile checks because Home Assistant is unavailable in this shell.
- Run focused tests when Home Assistant is available.
- Inspect the diff and confirm validation/fail-closed behavior is unchanged.
- Run formatting/static checks available in the repository if needed.
