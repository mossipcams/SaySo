# 0004. Media player ducking while a satellite listens

## Status

Accepted.

## Context

While an `assist_satellite` is listening, speech-to-text needs the room quiet. Media players in the satellite's area, such as TVs and speakers, could drown out the command. Listening ends before the conversation agent runs, so ducking lives in the integration, not the agent. Only Home Assistant can reach the players.

Live use on 2026-11-05 showed a Samsung TV in state `on` that reported a stale `volume_level` of 0.19 and never changed. Every wake ducked it to 0.02 and then restored the stale 0.19, which undid volume changes made with the remote or by voice. Roku players report no `volume_level`, so their original level cannot be read back, and a step-down followed by a step-up cannot be undone.

## Decision

- Trigger on `assist_satellite` state changes. Entering `listening` ducks; leaving it releases. Targets are `media_player` entities in the satellite's area (entity area, else device area) that are not `off`, `standby`, `unavailable`, or `unknown`, and that are not on the satellite's own device.
- Each player takes the first tier that applies:
  1. It supports `VOLUME_SET` and is `playing`. Save the level and set it to the duck level. A player already at or below the duck level is left alone.
  2. It supports `PAUSE` and is `playing` (Roku apps). Pause, then play on restore.
  3. It supports `VOLUME_MUTE` and is not muted (Roku live TV and HDMI, which report `on`). Mute, then unmute on restore.
  4. Otherwise skip.
- The volume tier applies only to `playing` players, so a TV reporting `on` with a stale level is muted and never set.
- Overlapping satellites share one map of original states with a reference count per entity. The original is recorded only on the first hold, and an entity is restored when its count reaches zero.
- Duck calls are non-blocking. Restore calls are blocking, tracked tasks, each capped at 10 s, and a failing call is logged without skipping the other restores.
- Volume intents and the conversation agent wait up to 2 s for pending restores before they run, so relative volume commands step from the real level.
- A 30 s timer per satellite releases a hold that never leaves `listening`. Unload and Home Assistant stop restore everything.
- Each config entry owns its ducker, stored on `runtime_data` and torn down on unload. A set of active duckers in `hass.data` lets volume intents wait for restores across entries. This replaces the single shared ducker in the review-fixes plan.
- The `duck_volume` option is 0.0 to 1.0 and defaults to 0.02.

Rejected:

- Ducking in the conversation agent. Listening is over by the time it runs.
- Restoring a remembered volume on a player that reports `on`. The remembered level can be stale.
- Stepping a Roku down to zero and back up. The original level is not recoverable.

## Consequences

- Only players in the satellite's area are touched. Players in other rooms are never affected.
- A player with no readable volume is paused or muted, not stepped.
- A hold that is never released clears within 30 s, or on unload or Home Assistant stop. Until then, a player can stay quiet.
- A hung player can delay a volume command by up to 2 s, the wait cap. The per-call cap of 10 s bounds the restore itself.

## Shipped in

- `a7e4ce2` feat(integration): duck media players while a satellite listens
- `a139728` fix(integration): mute non-playing TVs instead of restoring stale volume
- `b59f649` fix(integration): harden media ducking restores
- `ca660de` fix(integration): make volume intents wait for ducking restores
- `a6b8f47` refactor: keep one ducker per entry on runtime_data
