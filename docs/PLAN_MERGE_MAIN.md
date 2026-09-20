# Plan: Merge origin/main into ajax/nanowakeword

Status: resolving. PR 85 is CONFLICTING with main (PR 86 satellite review
plus wake mining). User instruction: keep the previous/merge (main) code
where hunks conflict.

## Keep from main (theirs)

PR 86 wake-review satellite code, mining pipeline, eval, inference GGUF
template-arg parse, lockfile/release metadata.

## Keep from this branch (ours)

- living2 ONNX, `living2.yaml`, `sayso-verifier.npz`, verifier wiring
- NanoWakeWord **removed** (do not resurrect provider, yaml, ONNX, or
  `nanowakeword` runtime dep from main’s v3 bump)
- living2 docs (`HANDOFF_WAKE`, `PLAN_LIVEKIT_VERIFIER`, models README
  operating point)

## Verification

Satellite wake/config/launcher tests. Grep: no `NanoWakeWordProvider`.
