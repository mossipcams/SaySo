"""Provision the embedded backend: the native wheel and the GGUF weights.

Both live outside the integration on purpose. ``llama-cpp-python`` cannot be a
manifest requirement (PyPI ships sdist only and the Home Assistant container has
no compiler), and a multi-hundred-megabyte GGUF does not belong in a HACS
repository. See docs/PLAN_EMBEDDED_INFERENCE.md §1.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import package as pkg_util

from .const import (
    LLAMA_CPP_MIN_VERSION,
    LLAMA_CPP_PACKAGE,
    LLAMA_CPP_WHEEL_INDEX,
    MODEL_STORAGE_SUBDIR,
)
from .exceptions import SaySoDependencyError, SaySoModelLoadError

_LOGGER = logging.getLogger(__name__)

_DOWNLOAD_CHUNK = 1024 * 1024
_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_read=60)


def models_dir(hass: HomeAssistant) -> Path:
    """Return the persistent model directory under /config."""
    return Path(hass.config.path(MODEL_STORAGE_SUBDIR))


def is_llama_cpp_installed() -> bool:
    """Return whether llama_cpp can be imported in this interpreter."""
    return pkg_util.is_installed(f"{LLAMA_CPP_PACKAGE}>={LLAMA_CPP_MIN_VERSION}")


def _install_llama_cpp() -> bool:
    """Install the prebuilt llama-cpp-python wheel. Runs in an executor.

    ``--no-deps`` is not optional: a plain install resolves a newer numpy over
    the one Home Assistant pins, which would affect every other integration.
    ``diskcache`` is llama_cpp's only other import-time dependency.
    """
    # install_package copies os.environ, so the extra index reaches uv this way.
    # Home Assistant already sets UV_EXTRA_INDEX_URL to its own wheel mirror;
    # append rather than replace so that mirror keeps working.
    # ponytail: process-global env mutation, restored in the finally. Two entries
    # setting up at the same moment could interleave here; uv tolerates it
    # because the second install is a no-op. Use a module-level lock if SaySo
    # ever provisions more than one backend concurrently.
    previous = os.environ.get("UV_EXTRA_INDEX_URL")
    merged = f"{previous} {LLAMA_CPP_WHEEL_INDEX}" if previous else LLAMA_CPP_WHEEL_INDEX
    os.environ["UV_EXTRA_INDEX_URL"] = merged
    try:
        for requirement in ("diskcache", f"{LLAMA_CPP_PACKAGE}>={LLAMA_CPP_MIN_VERSION}"):
            if not pkg_util.install_package(requirement, upgrade=False):
                return False
    finally:
        if previous is None:
            os.environ.pop("UV_EXTRA_INDEX_URL", None)
        else:
            os.environ["UV_EXTRA_INDEX_URL"] = previous
    return True


async def async_ensure_llama_cpp(hass: HomeAssistant) -> None:
    """Make llama_cpp importable, installing the prebuilt wheel if needed.

    The container filesystem is reset by Home Assistant updates, so this runs on
    every setup and is a cheap no-op once installed.
    """
    if is_llama_cpp_installed():
        return

    _LOGGER.info(
        "Installing %s from the prebuilt wheel index; this happens once per "
        "Home Assistant update",
        LLAMA_CPP_PACKAGE,
    )
    if not await hass.async_add_executor_job(_install_llama_cpp):
        raise SaySoDependencyError(
            f"Could not install {LLAMA_CPP_PACKAGE}. No prebuilt wheel is "
            f"available for this platform ({sys.platform}/{os.uname().machine}) "
            f"at {LLAMA_CPP_WHEEL_INDEX}, and the Home Assistant container "
            "cannot build it from source."
        )
    # Drop any negative import cache from before the install.
    import importlib

    importlib.invalidate_caches()
    _LOGGER.info("Installed %s", LLAMA_CPP_PACKAGE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_DOWNLOAD_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


async def async_ensure_model(
    hass: HomeAssistant,
    *,
    url: str,
    filename: str,
    sha256: str | None = None,
) -> Path:
    """Return the local GGUF path, downloading it once if missing.

    The download lands on a ``.part`` file and is renamed only after it
    verifies, so an interrupted download can never be loaded as a model.
    """
    directory = models_dir(hass)
    await hass.async_add_executor_job(
        lambda: directory.mkdir(parents=True, exist_ok=True)
    )
    target = directory / filename

    if await hass.async_add_executor_job(target.is_file):
        if sha256 is not None:
            actual = await hass.async_add_executor_job(_sha256, target)
            if actual != sha256:
                await hass.async_add_executor_job(target.unlink)
                raise SaySoModelLoadError(
                    f"Checksum mismatch for {filename}: expected {sha256}, got "
                    f"{actual}. The corrupt file was removed; retry setup."
                )
        return target

    _LOGGER.info("Downloading SaySo model %s from %s", filename, url)
    partial = target.with_suffix(target.suffix + ".part")
    session = async_get_clientsession(hass)

    try:
        async with session.get(url, timeout=_DOWNLOAD_TIMEOUT) as response:
            if response.status != 200:
                raise SaySoModelLoadError(
                    f"Model download failed with HTTP {response.status}: {url}"
                )
            handle = await hass.async_add_executor_job(partial.open, "wb")
            try:
                async for chunk in response.content.iter_chunked(_DOWNLOAD_CHUNK):
                    await hass.async_add_executor_job(handle.write, chunk)
            finally:
                await hass.async_add_executor_job(handle.close)
    except (TimeoutError, aiohttp.ClientError) as err:
        await hass.async_add_executor_job(partial.unlink, True)
        raise SaySoModelLoadError(f"Model download failed: {err}") from err
    except asyncio.CancelledError:
        await hass.async_add_executor_job(partial.unlink, True)
        raise

    if sha256 is not None:
        actual = await hass.async_add_executor_job(_sha256, partial)
        if actual != sha256:
            await hass.async_add_executor_job(partial.unlink, True)
            raise SaySoModelLoadError(
                f"Downloaded {filename} has checksum {actual}, expected {sha256}"
            )

    await hass.async_add_executor_job(partial.replace, target)
    _LOGGER.info("SaySo model ready at %s", target)
    return target
