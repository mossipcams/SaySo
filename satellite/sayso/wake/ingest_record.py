from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest_record(record_dir: Path) -> tuple[bool, str]:
    meta_path = record_dir / "record.json"
    if not meta_path.is_file():
        return False, "missing record.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"malformed record.json: {exc}"
    hashes = meta.get("hashes") or {}
    window = record_dir / "window.wav"
    if not window.is_file():
        return False, "missing window.wav"
    window_hash = _sha256_file(window)
    if hashes.get("window_sha256") and hashes["window_sha256"] != window_hash:
        return False, "window hash mismatch"
    pre = record_dir / "pre.wav"
    if pre.is_file():
        pre_hash = _sha256_file(pre)
        if hashes.get("pre_sha256") and hashes["pre_sha256"] != pre_hash:
            return False, "pre hash mismatch"
    post = record_dir / "post.wav"
    if post.is_file():
        post_hash = _sha256_file(post)
        if hashes.get("post_sha256") and hashes["post_sha256"] != post_hash:
            return False, "post hash mismatch"
    raw_stream = (meta.get("streams") or {}).get("raw")
    if raw_stream is not None:
        raw = record_dir / "raw_pre.wav"
        raw_hash = raw_stream.get("sha256")
        if raw_hash:
            if not raw.is_file():
                return False, "missing raw_pre.wav"
            if _sha256_file(raw) != raw_hash:
                return False, "raw hash mismatch"
        elif not raw_stream.get("raw_missing"):
            return False, "missing raw sha256"
    return True, "ok"
