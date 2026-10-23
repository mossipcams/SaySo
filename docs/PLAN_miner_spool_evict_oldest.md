# Plan: wake miner spool evicts the oldest record when full

## Scope

The satellite hard-negative miner (`satellite/sayso/wake/mining.py`) drops new
records once a class cap or the spool limit is reached, counting only the drops.
Change it to evict the oldest record and save the new one:

- Class at its cap: evict the oldest record of that class (by `published_utc`).
- Spool at its record or byte limit: evict the oldest record overall.

Constraints:

- Eviction removes `records/<id>/` and `outcomes/<id>.*.json`, the same cleanup
  as `drain_acks()`.
- Counters stay consistent (class count, published records, published bytes,
  `_spool_full`), by decrementing or by reconciling.
- Never evict the record being published.
- Count evictions in the losses stats (`evicted`).
- An ack for an already-evicted ID is harmless.
- The ingest/ack contract with the host (`ingest_record()`, `write_ack()`,
  `drain_acks()`) is unchanged.

Out of scope: deploying to the Pi (a separate step that needs sudo), host-side
ingest tooling.

## Files to touch

- `satellite/sayso/wake/mining.py`
- `satellite/sayso/wake/test_mining.py`
- Docs that describe "stop collection and count drops" (under `docs/` or the
  satellite docs), if any

## Verification

- New tests in `test_mining.py`:
  - Class cap: at the cap, a new record is saved; the oldest record of that
    class and its outcomes are gone; other classes are untouched; the count
    stays at the cap.
  - Global limit: at the record or byte limit, the oldest record overall is
    evicted.
  - Draining: `drain_acks()` and reconciliation still work after evictions,
    including an ack for an evicted ID.
- Existing drop-when-full assertions (for example `losses.spool_full >= 1`) are
  replaced with eviction assertions, not deleted.
- `ruff check .`, `ruff format --check .`, `pytest -q satellite/sayso`
