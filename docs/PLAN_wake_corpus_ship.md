# PLAN: Ship long-form sessions Pi → train VM with rsync, then delete on Pi

## Goal

Hours-long ingest on the Pi is staging. After a successful **rsync** to the
training VM (`ubuntu@192.168.1.140`), remove the session from the Pi.
rsync-over-SSH (no sftp). Do not use tar.

## Flow

```text
Pi: wake_corpus.py ingest room.wav --corpus /var/lib/sayso-satellite/wake-sessions --session-id room_a
Pi: wake_corpus.py ship room_a --corpus /var/lib/sayso-satellite/wake-sessions
  -> rsync -a sessions/room_a/ ubuntu@192.168.1.140:.../sessions/room_a/
  -> verify remote audio sha256 matches session.json
  -> delete local session directory only on success
```

Do not delete on copy/verify failure. Do not stop LFM2 on the train host.
`--dry-run` neither copies nor deletes. `--remove-source-files` is not used;
delete is explicit after verify.

## Scope

- `satellite/sayso/wake/sessions.py` — rsync ship helper
- `scripts/wake_corpus.py` — `ship` subcommand
- `satellite/sayso/wake/test_sessions.py` — mocked rsync tests
- `scripts/test_wake_corpus.py` — CLI smoke if needed
- `docs/SATELLITE_DATA_COLLECTION.md`
- `docs/WAKE_TRAINING_DATA_ARCHITECTURE.md`
- this plan

Defaults: remote `ubuntu@192.168.1.140`, dest
`/home/ubuntu/sayso-wake-data/corpus`. SSH options overridable (`-e ssh`).

## Verification

```text
python3 -m pytest satellite/sayso/wake/test_sessions.py scripts/test_wake_corpus.py -q
```

Success deletes local; verify fail keeps local; dry-run no-op.
