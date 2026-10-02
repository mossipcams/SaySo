import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import wake_koda_seeds as k


def _rir():
    rir = np.zeros(4000, dtype=np.float32)
    rir[10] = 1.0
    rir[1500:3000] = np.random.default_rng(0).normal(0, 0.01, 1500)
    return rir


def test_direct_scales_tail_kept():
    rir = _rir()
    far = k.distance_rir(rir, 3.0)
    assert abs(far[10] - 0.1) < 1e-6
    assert np.array_equal(far[1500:], rir[1500:])


def test_level_drops_with_distance():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.1, k.FRAME).astype(np.float32)
    rms = [np.sqrt((k.apply_rir(x, _rir(), d) ** 2).mean()) for d in k.DISTANCES_M]
    assert rms[0] > rms[1] > rms[2]


def test_word_ends_in_last_half_second():
    rng = np.random.default_rng(2)
    speech = np.ones(8000, dtype=np.float32)
    for _ in range(50):
        frame = k.place(speech, rng)
        assert len(frame) == k.FRAME
        assert np.flatnonzero(frame)[-1] >= k.FRAME - int(0.5 * k.SAMPLE_RATE) - 1


def test_holdout_never_emitted(tmp_path):
    spool = tmp_path / "s"
    rec = spool / "records" / "abcdef012345"
    rec.mkdir(parents=True)
    x = np.random.default_rng(3).normal(0, 0.1, k.FRAME).astype(np.float32)
    k.write_wav(rec / "window.wav", x)
    import hashlib
    sha = hashlib.sha256((rec / "window.wav").read_bytes()).hexdigest()
    bg = tmp_path / "bg.wav"
    k.write_wav(bg, x)
    rir = tmp_path / "rir.wav"
    k.write_wav(rir, _rir())
    seed = {"capture_id": "abcdef012345", "sha256": sha, "split": "holdout", "verified": True}
    assert k.build([seed], spool, [rir], [bg], tmp_path / "o1", 2, 0) == 0
    seed.update(split="train", verified=False)
    assert k.build([seed], spool, [rir], [bg], tmp_path / "o2", 2, 0) == 0
    seed["verified"] = True
    assert k.build([seed], spool, [rir], [bg], tmp_path / "o3", 2, 0) == 2 * 4


def test_negatives_skip_koda_like_and_holdout(tmp_path):
    import hashlib
    import json
    spool = tmp_path / "s"
    x = np.random.default_rng(4).normal(0, 0.1, k.FRAME).astype(np.float32)
    for cid, text in (("a", "Thank you."), ("b", "CODA!"), ("c", "Toda."), ("d", "Facebook.")):
        rec = spool / "records" / cid
        rec.mkdir(parents=True)
        k.write_wav(rec / "window.wav", x + len(cid) * 0)
        sha = hashlib.sha256((rec / "window.wav").read_bytes()).hexdigest()
        (rec / "record.json").write_text(json.dumps({"hashes": {"window_sha256": sha}}))
        (rec / "check.json").write_text(json.dumps({"transcript": text}))
    rows = k.select_negatives(spool)
    assert {r["capture_id"] for r in rows} == {"a", "d"}
    for r in rows:
        r["split"] = "holdout"
    assert k.build_negatives(rows, spool, tmp_path / "o", 3, 0) == 0
    rows[0]["split"] = "train"
    assert k.build_negatives(rows, spool, tmp_path / "o", 3, 0) == 3
