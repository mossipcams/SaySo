import hashlib
import json
import os
from pathlib import Path

import pytest

import custom_components.sayso.model_store as model_store
from custom_components.sayso.core.exceptions import SaySoModelLoadError

FILENAME = "model.gguf"
CONTENT = b"verified model bytes"
SHA256 = hashlib.sha256(CONTENT).hexdigest()


class _FakeConfig:
    def __init__(self, root: Path) -> None:
        self._root = root

    def path(self, *parts: str) -> str:
        return str(self._root.joinpath(*parts))


class _FakeHass:
    def __init__(self, root: Path) -> None:
        self.config = _FakeConfig(root)

    async def async_add_executor_job(self, func, *args):
        return func(*args)


@pytest.fixture
def fake_hass(tmp_path: Path) -> _FakeHass:
    fake = _FakeHass(tmp_path)
    models_dir = model_store.models_dir(fake)
    models_dir.mkdir(parents=True)
    (models_dir / FILENAME).write_bytes(CONTENT)
    return fake


@pytest.fixture
def hash_calls(monkeypatch) -> list[Path]:
    calls: list[Path] = []
    real_sha256 = model_store._sha256

    def counting_sha256(path: Path) -> str:
        calls.append(path)
        return real_sha256(path)

    monkeypatch.setattr(model_store, "_sha256", counting_sha256)
    return calls


def _model_path(fake_hass: _FakeHass) -> Path:
    return model_store.models_dir(fake_hass) / FILENAME


def _sidecar(fake_hass: _FakeHass) -> Path:
    return model_store.models_dir(fake_hass) / f"{FILENAME}.verified"


def _expected_record(path: Path) -> dict:
    stat = path.stat()
    return {"sha256": SHA256, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


async def test_first_verify_hashes_and_writes_sidecar(fake_hass, hash_calls):
    await model_store.async_verify_model(fake_hass, filename=FILENAME, sha256=SHA256)

    assert len(hash_calls) == 1
    assert json.loads(_sidecar(fake_hass).read_text()) == _expected_record(
        _model_path(fake_hass)
    )


async def test_second_verify_skips_hashing(fake_hass, hash_calls):
    await model_store.async_verify_model(fake_hass, filename=FILENAME, sha256=SHA256)
    await model_store.async_verify_model(fake_hass, filename=FILENAME, sha256=SHA256)

    assert len(hash_calls) == 1


async def test_changed_mtime_forces_rehash(fake_hass, hash_calls):
    await model_store.async_verify_model(fake_hass, filename=FILENAME, sha256=SHA256)
    path = _model_path(fake_hass)
    later = path.stat().st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(later, later))

    await model_store.async_verify_model(fake_hass, filename=FILENAME, sha256=SHA256)

    assert len(hash_calls) == 2


async def test_changed_size_forces_rehash(fake_hass, hash_calls):
    await model_store.async_verify_model(fake_hass, filename=FILENAME, sha256=SHA256)
    with _model_path(fake_hass).open("ab") as handle:
        handle.write(b"x")

    with pytest.raises(SaySoModelLoadError):
        await model_store.async_verify_model(
            fake_hass, filename=FILENAME, sha256=SHA256
        )

    assert len(hash_calls) == 2


async def test_sidecar_with_other_sha256_forces_rehash(fake_hass, hash_calls):
    await model_store.async_verify_model(fake_hass, filename=FILENAME, sha256=SHA256)
    record = _expected_record(_model_path(fake_hass))
    record["sha256"] = "0" * 64
    _sidecar(fake_hass).write_text(json.dumps(record))

    await model_store.async_verify_model(fake_hass, filename=FILENAME, sha256=SHA256)

    assert len(hash_calls) == 2
    assert json.loads(_sidecar(fake_hass).read_text()) == _expected_record(
        _model_path(fake_hass)
    )


async def test_corrupt_sidecar_forces_rehash(fake_hass, hash_calls):
    _sidecar(fake_hass).write_text("{not json")

    await model_store.async_verify_model(fake_hass, filename=FILENAME, sha256=SHA256)

    assert len(hash_calls) == 1
    assert json.loads(_sidecar(fake_hass).read_text()) == _expected_record(
        _model_path(fake_hass)
    )


async def test_wrong_hash_raises_and_leaves_no_sidecar(fake_hass, hash_calls):
    with pytest.raises(SaySoModelLoadError, match="Checksum mismatch"):
        await model_store.async_verify_model(
            fake_hass, filename=FILENAME, sha256="0" * 64
        )

    assert len(hash_calls) == 1
    assert not _sidecar(fake_hass).exists()
