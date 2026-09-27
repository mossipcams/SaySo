# Plan: exclude missing wake fixtures from promotion eval

## Scope

Keep the five unrecorded wake cases available for diagnostics, but exclude them from the automated promotion run. Retain all 30 `silent30` clips as the available candidate evaluation set. Run each candidate at its LiveKit `optimal_threshold` and record that threshold without changing the deployed baseline. Keep the known shared speaker/session lineage visible in reports and preserve the separate baseline/calibration guard.

Deploy pass 2 as the user-designated champion **SaySo Voicev1** on the Pi. Preserve the previous model/config as a rollback, install Voicev1 at a distinct path, set its threshold to `0.22`, switch the Pi config, and restart the satellite service. **Completed:** Voicev1 is active and the service is healthy.

## Files to touch

- `satellite/eval/cases.json` — mark the five unrecorded cases as not required for promotion.
- `satellite/sayso/wake/eval.py` and `satellite/eval/run.py` — add a promotion-only case filter while leaving normal strict evaluation unchanged, and include the applied threshold in reports.
- `scripts/wake_livekit_experiment.py` — invoke the promotion-only strict eval at each candidate's LiveKit optimal threshold.
- `satellite/eval/manifest.json`, `satellite/eval/splits.json`, `satellite/eval/baseline.json`, `satellite/eval/README.md` — document the diagnostic-only cases, promotion selection, and remaining baseline blockers.
- `docs/WAKE_WORD_DATA.md`, `docs/SAYSO_WAKE_WORD_TRAINING_PLAN.md`, `docs/PLAN_expand_wake_generator.md`, `training/wake/README.md`, and `docs/HANDOFF_WAKE_TRAINING_MIX.md` — align current pipeline and handoff guidance.
- `satellite/models/README.md` — record Voicev1 champion selection, Pi deployment, measured eval, and activation state.
- On the Pi: back up `/opt/sayso-satellite/models/sayso.onnx` and `/etc/sayso-satellite/config.yaml`, install `/opt/sayso-satellite/models/sayso-voicev1.onnx`, set the active config to that model and threshold, then restart `sayso-satellite.service`.

## Verification

- Parse all changed JSON files.
- Check CLI help exposes the promotion-only option.
- Run pass 1 and pass 2 against the 30-case eval set at their model thresholds; do not modify or run tests.
- Verified the Pi model SHA-256 matches the pass-2 artifact; config points to Voicev1 at `0.22`; `sayso-satellite.service` is active after restart. Rollback files are retained.
