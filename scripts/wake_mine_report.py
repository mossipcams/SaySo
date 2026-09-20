#!/usr/bin/env python3
"""Summarise, ingest, label, and inventory wake-training assets.

Records land unlabelled: the satellite knows a window scored high, not whether
anyone actually said "SaySo". Labelling is a listening job, and this script is
the thin wrapper around it.

    # what's in the spool
    python scripts/wake_mine_report.py /var/lib/sayso-satellite/wake-mining

    # verify hashes and write ack files for transfer back to the satellite
    python scripts/wake_mine_report.py SPOOL --ingest

    # highest-scoring unreviewed clips first, with a play command per row
    python scripts/wake_mine_report.py SPOOL --unreviewed --play

    # record a verdict on a legacy flat clip or a record directory
    python scripts/wake_mine_report.py SPOOL --label 20260910T001432_412_s0.3120 negative

    # inventory wake-only roots (mine spool, eval audio, models data/output)
    python scripts/wake_mine_report.py SPOOL --inventory

    # quarantine corrupt assets and remove verified duplicates (retained copy kept)
    python scripts/wake_mine_report.py SPOOL --inventory --cleanup
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from collections import defaultdict
from pathlib import Path

# Import mining helpers from the satellite package when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SATELLITE_ROOT = _REPO_ROOT / "satellite"
if str(_SATELLITE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SATELLITE_ROOT))

from sayso.wake.mining import ingest_record, write_ack  # noqa: E402

LABELS = ("positive", "negative", "unsure")
MANIFEST_NAME = "wake_cleanup_manifest.json"
LFM_ROOT = "training"


def pinned_wake_roots() -> list[Path]:
    return [
        _REPO_ROOT / "satellite" / "eval" / "audio",
        _REPO_ROOT / "satellite" / "models" / "data",
    ]


def _is_pinned_path(path: Path) -> bool:
    resolved = path.resolve()
    for root in pinned_wake_roots():
        if not root.is_dir():
            continue
        try:
            if resolved.is_relative_to(root.resolve()):
                return True
        except ValueError:
            continue
    return False


def _record_dirs(spool: Path) -> list[Path]:
    records = spool / "records"
    if records.is_dir():
        return sorted(p for p in records.iterdir() if p.is_dir())
    return []


def _legacy_rows(spool: Path) -> list[dict]:
    rows = []
    for wav in sorted(spool.glob("*.wav")):
        sidecar = wav.with_suffix(".json")
        if not sidecar.is_file():
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"skipping malformed sidecar: {sidecar}", file=sys.stderr)
            continue
        meta["_wav"] = wav
        meta["_json"] = sidecar
        rows.append(meta)
    return rows


def load(spool: Path) -> list[dict]:
    rows: list[dict] = []
    for record_dir in _record_dirs(spool):
        meta_path = record_dir / "record.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"skipping malformed record: {meta_path}", file=sys.stderr)
            continue
        meta["_record_dir"] = record_dir
        meta["_wav"] = record_dir / "window.wav"
        meta["_json"] = meta_path
        rows.append(meta)
    rows.extend(_legacy_rows(spool))
    return rows


def ingest(spool: Path) -> int:
    """Verify published records and write ack files. Returns ack count."""
    acked = 0
    for record_dir in _record_dirs(spool):
        capture_id = record_dir.name
        ok, message = ingest_record(record_dir)
        if not ok:
            print(f"reject {capture_id}: {message}", file=sys.stderr)
            continue
        write_ack(spool, capture_id)
        print(f"acked {capture_id}")
        acked += 1
    return acked


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_under_lfm(path: Path) -> bool:
    parts = {part.lower() for part in path.resolve().parts}
    return LFM_ROOT in parts


def default_wake_roots(spool: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if spool is not None:
        roots.append(spool)
    roots.extend(
        [
            _REPO_ROOT / "satellite" / "eval" / "audio",
            _REPO_ROOT / "satellite" / "models" / "data",
            _REPO_ROOT / "satellite" / "models" / "output",
        ]
    )
    return [root for root in roots if root.exists()]


def _verify_wav(path: Path) -> tuple[bool, str]:
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.getnframes()
        if sample_width != 2:
            return False, f"expected 16-bit PCM, got width {sample_width}"
        if channels < 1:
            return False, "no audio channels"
        if frames == 0:
            return False, "truncated or empty wav"
        if rate not in (8000, 16000, 22050, 44100, 48000):
            return False, f"unexpected sample rate {rate}"
        return True, "ok"
    except (OSError, wave.Error) as exc:
        return False, f"corrupt wav: {exc}"


def _record_label(meta_path: Path) -> str | None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    label = meta.get("label")
    return str(label) if label else None


def _scan_wake_roots(roots: list[Path]) -> list[dict]:
    entries: list[dict] = []
    for root in roots:
        if _is_under_lfm(root):
            continue
        if not root.is_dir():
            continue

        records_dir = root / "records"
        if records_dir.is_dir():
            for record_dir in sorted(records_dir.iterdir()):
                if not record_dir.is_dir():
                    continue
                window = record_dir / "window.wav"
                meta_path = record_dir / "record.json"
                if not meta_path.is_file():
                    entries.append(
                        {
                            "path": str(record_dir),
                            "sha256": None,
                            "reason": "incomplete_record",
                            "detail": "missing record.json",
                            "action": "quarantine",
                        }
                    )
                    continue
                ok, message = ingest_record(record_dir)
                if not ok:
                    entries.append(
                        {
                            "path": str(record_dir),
                            "sha256": _sha256_file(window) if window.is_file() else None,
                            "reason": "incomplete_record",
                            "detail": message,
                            "action": "quarantine",
                        }
                    )
                    continue
                label = _record_label(meta_path)
                wav_ok, wav_reason = _verify_wav(window)
                if not wav_ok:
                    entries.append(
                        {
                            "path": str(window),
                            "sha256": _sha256_file(window),
                            "reason": "corrupt_recording",
                            "detail": wav_reason,
                            "label": label,
                            "action": "quarantine",
                        }
                    )
                    continue
                entries.append(
                    {
                        "path": str(window),
                        "sha256": _sha256_file(window),
                        "reason": "ok",
                        "label": label,
                        "capture_id": record_dir.name,
                        "action": "keep",
                    }
                )

        for wav in sorted(root.rglob("*.wav")):
            if _is_under_lfm(wav):
                continue
            # records/<id>/window.wav is indexed above; parent.parent is records/, not records/records/.
            if wav.name == "window.wav" and wav.parent.parent.name == "records":
                continue
            if any(part == "quarantine" for part in wav.parts):
                continue
            wav_ok, wav_reason = _verify_wav(wav)
            sidecar = wav.with_suffix(".json")
            label = _record_label(sidecar) if sidecar.is_file() else None
            if not wav_ok:
                entries.append(
                    {
                        "path": str(wav),
                        "sha256": None,
                        "reason": "corrupt_recording",
                        "detail": wav_reason,
                        "label": label,
                        "action": "quarantine",
                    }
                )
                continue
            digest = _sha256_file(wav)
            entries.append(
                {
                    "path": str(wav),
                    "sha256": digest,
                    "reason": "ok",
                    "label": label,
                    "action": "keep",
                }
            )
    return entries


def _audit_labels(entries: list[dict]) -> list[dict]:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        digest = entry.get("sha256")
        if digest:
            by_hash[digest].append(entry)
    findings: list[dict] = []
    for digest, group in by_hash.items():
        labels = {entry.get("label") for entry in group if entry.get("label")}
        if len(labels) > 1:
            for entry in group:
                findings.append(
                    {
                        "path": entry["path"],
                        "sha256": digest,
                        "reason": "conflicting_labels",
                        "detail": f"labels={sorted(labels)}",
                        "action": "quarantine",
                    }
                )
    return findings


def _duplicate_keep_priority(entry: dict) -> tuple[int, str]:
    pinned = 0 if _is_pinned_path(Path(entry["path"])) else 1
    return (pinned, entry["path"])


def _mark_duplicates(entries: list[dict]) -> list[dict]:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        if entry.get("reason") != "ok" or not entry.get("sha256"):
            continue
        by_hash[entry["sha256"]].append(entry)
    updated = list(entries)
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        keep = sorted(group, key=_duplicate_keep_priority)[0]
        for entry in sorted(group, key=_duplicate_keep_priority)[1:]:
            if _is_pinned_path(Path(entry["path"])):
                continue
            updated.append(
                {
                    "path": entry["path"],
                    "sha256": digest,
                    "reason": "verified_duplicate",
                    "detail": f"retained_copy={keep['path']}",
                    "action": "remove",
                }
            )
    return updated


def _quarantine_path(root: Path, asset: Path) -> Path:
    rel = asset.relative_to(root) if asset.is_relative_to(root) else Path(asset.name)
    return root / "quarantine" / rel


def _apply_cleanup(roots: list[Path], manifest: list[dict]) -> int:
    actions = 0
    for root in roots:
        if _is_under_lfm(root) or not root.is_dir():
            continue
        for entry in manifest:
            if entry.get("action") not in {"quarantine", "remove"}:
                continue
            asset = Path(entry["path"])
            if not asset.exists():
                continue
            if _is_pinned_path(asset):
                continue
            if not any(asset.is_relative_to(r) for r in roots if r.is_dir()):
                continue
            owning_root = next(r for r in roots if asset.is_relative_to(r))
            if entry["action"] == "remove":
                asset.unlink()
                actions += 1
                continue
            target = _quarantine_path(owning_root, asset)
            target.parent.mkdir(parents=True, exist_ok=True)
            asset.replace(target)
            actions += 1
    return actions


def inventory_wake_assets(
    roots: list[Path],
    *,
    cleanup: bool = False,
    manifest_path: Path | None = None,
) -> list[dict]:
    entries = _scan_wake_roots(roots)
    entries.extend(_audit_labels(entries))
    entries = _mark_duplicates(entries)
    unknown = sum(1 for entry in entries if entry.get("label") is None and entry.get("reason") == "ok")
    manifest = {
        "version": 1,
        "roots": [str(root) for root in roots],
        "unknown_evidence_preserved": unknown,
        "entries": entries,
    }
    if manifest_path is None and roots:
        manifest_path = roots[0] / MANIFEST_NAME
    if manifest_path is not None:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if cleanup:
        _apply_cleanup(roots, entries)
    return entries


def summarise(rows: list[dict]) -> None:
    if not rows:
        print(
            "Spool is empty. Mining writes a record only when a window clears "
            "wake_word.mine_threshold or an independent below-threshold sample fires; "
            "a quiet room can go hours."
        )
        return
    fired = sum(1 for r in rows if r.get("fired"))
    labelled = [r for r in rows if r.get("label")]
    scores = sorted(float(r["score"]) for r in rows)
    print(f"records         {len(rows)}")
    print(f"  fired         {fired}   (>= detect threshold; the rest are near-misses)")
    print(f"  near-miss     {len(rows) - fired}")
    print(f"  labelled      {len(labelled)} / {len(rows)}")
    for label in LABELS:
        n = sum(1 for r in labelled if r.get("label") == label)
        if n:
            print(f"    {label:<10}  {n}")
    print(f"score  min {scores[0]:.4f}  p50 {scores[len(scores) // 2]:.4f}  max {scores[-1]:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spool", type=Path)
    ap.add_argument("--unreviewed", action="store_true", help="List only records with no label")
    ap.add_argument("--play", action="store_true", help="Print a play command per row")
    ap.add_argument("--ingest", action="store_true", help="Verify hashes and write ack files")
    ap.add_argument("--label", nargs=2, metavar=("STEM", "LABEL"), help=f"Set label; one of {LABELS}")
    ap.add_argument("--inventory", action="store_true", help="Scan wake-only roots and write cleanup manifest")
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="With --inventory: quarantine corrupt assets and remove verified duplicates",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"Cleanup manifest path (default: <spool>/{MANIFEST_NAME})",
    )
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    if args.inventory:
        roots = default_wake_roots(args.spool if args.spool.is_dir() else None)
        if args.spool.is_dir() and args.spool not in roots:
            roots.insert(0, args.spool)
        if not roots:
            print("no wake asset roots found", file=sys.stderr)
            return 1
        manifest_path = args.manifest or (args.spool / MANIFEST_NAME if args.spool.is_dir() else None)
        entries = inventory_wake_assets(roots, cleanup=args.cleanup, manifest_path=manifest_path)
        quarantine = sum(1 for entry in entries if entry.get("action") == "quarantine")
        remove = sum(1 for entry in entries if entry.get("action") == "remove")
        keep = sum(1 for entry in entries if entry.get("action") == "keep")
        print(f"roots           {len(roots)}")
        print(f"entries         {len(entries)}")
        print(f"  keep          {keep}")
        print(f"  quarantine    {quarantine}")
        print(f"  remove        {remove}")
        if manifest_path is not None:
            print(f"manifest        {manifest_path}")
        return 0

    if not args.spool.is_dir():
        print(f"No such spool directory: {args.spool}", file=sys.stderr)
        return 1

    if args.ingest:
        acked = ingest(args.spool)
        print(f"ingested {acked} record(s)")
        return 0

    if args.label:
        stem, label = args.label
        if label not in LABELS:
            print(f"Label must be one of {LABELS}", file=sys.stderr)
            return 1
        record_json = args.spool / "records" / stem / "record.json"
        sidecar = record_json if record_json.is_file() else args.spool / f"{stem}.json"
        if not sidecar.is_file():
            print(f"No sidecar for stem {stem}", file=sys.stderr)
            return 1
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["label"] = label
        sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        print(f"{stem} -> {label}")
        return 0

    rows = load(args.spool)
    summarise(rows)

    listing = [r for r in rows if not r.get("label")] if args.unreviewed else rows
    if not listing:
        return 0
    listing.sort(key=lambda r: float(r["score"]), reverse=True)
    print(f"\n{'score':>7}  {'fired':<5}  {'label':<9}  clip")
    for row in listing[: args.limit]:
        wav = row["_wav"]
        print(
            f"{float(row['score']):>7.4f}  {str(bool(row.get('fired'))):<5}  "
            f"{str(row.get('label') or '-'):<9}  {wav.name if wav.is_file() else row.get('capture_id', wav)}"
        )
        if args.play and wav.is_file():
            print(f"         aplay {wav}")
    if len(listing) > args.limit:
        print(f"... {len(listing) - args.limit} more (--limit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
