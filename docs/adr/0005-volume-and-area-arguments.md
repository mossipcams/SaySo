# 0005. Volume step and area handling for tool arguments

## Status

Accepted.

## Context

"Turn down the TV" produced the correct call, `HassSetVolumeRelative` with `volume_step: down`. Home Assistant's handler matches targets with `VOLUME_SET`, so a Roku TV that supports only `VOLUME_STEP` never matched (defect #34). The fix belongs in SaySo until Home Assistant changes upstream.

The satellite's area leaked into named-device calls. "Mute the TV" from the Kitchen satellite produced `name=Sofa TV, area=Kitchen`. Grounding accepted the call because `is_grounded` checked only the name. Home Assistant then found no Sofa TV in the Kitchen and the turn failed, with no retry. Hiding the area from the prompt would move the model off its training layout, so the fix belongs before execution.

Separately, executed tools used the validator's converted output. Home Assistant re-ran its slot converters on it, so "volume up ten" became 10, then 0.1, then 0.

## Decision

- `StepVolumeRelativeHandler` subclasses Home Assistant's relative volume handler and keeps its slot schema. When matching fails with reason `FEATURE`, it retries with `VOLUME_STEP`. An "up" or positive step is one volume-up call per target, a "down" or negative step is one volume-down call, and zero makes no call. Numeric changes on players with `VOLUME_SET` still go through Home Assistant's path. Without a name, several matches keep only the playing players, as Home Assistant does.
- The override is registered from `async_setup`. The manifest lists `media_player` in `after_dependencies`, so Home Assistant registers its handler first and this one replaces it.
- Registration is guarded. If the override cannot be imported or registered, SaySo logs an error and Home Assistant's default stays active. Setup still succeeds.
- After validation passes, tools run with the model's original arguments (`ce8c119`). Validation still rejects bad calls before execution.
- `drop_unspoken_area` runs in the post-model path, after validation and before grounding. When a string `name` resolves to exactly one entity in a different area, and the utterance did not say that area, the `area` argument is removed. Area-only calls, ambiguous or unknown names, matching areas, and spoken areas are unchanged. Dropped areas are recorded in the parse span as `area_dropped`.

Rejected:

- Hiding the satellite area in the prompt. It moves the model off the training layout.
- Patching Home Assistant upstream. SaySo cannot ship that, and the override is temporary.
- Retrying a failed call without its area. Executed actions are never retried (ADR 0001).

## Consequences

- A numeric volume change on a step-only player is one step, not the requested amount. Home Assistant cannot set an exact level on these players.
- If Home Assistant changes the intent or its imports, relative volume falls back to Home Assistant's default and logs an error. Other voice commands are unaffected.
- A named device in another room now resolves by name. Area-only calls keep their area, so area-only targeting is unchanged.
- The area rule is in `core/grounding.py`, which moved there in `33e1169`.

## Shipped in

- `e7aaf93` fix(integration): step the volume of players without VOLUME_SET
- `189c971` fix(integration): keep SaySo loading when the volume-step override fails to import
- `ce8c119` fix(integration): execute tools with the model's original arguments
- `71efc71` fix(grounding): drop unspoken areas from named-device calls
