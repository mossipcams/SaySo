from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import site
import subprocess
import sys
from pathlib import Path

import aiohttp
import homeassistant
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import package as pkg_util

from .const import (
    LLAMA_CPP_PACKAGE,
    LLAMA_CPP_VERSION,
    LLAMA_CPP_WHEEL_INDEX,
    MODEL_STORAGE_SUBDIR,
)
from .exceptions import SaySoDependencyError, SaySoModelLoadError

_LOGGER = logging.getLogger(__name__)

_DOWNLOAD_CHUNK = 1024 * 1024
_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_read=60)


def models_dir(hass: HomeAssistant) -> Path:
    return Path(hass.config.path(MODEL_STORAGE_SUBDIR))


def list_local_models(hass: HomeAssistant) -> list[str]:
    directory = models_dir(hass)
    if not directory.is_dir():
        return []
    return sorted(str(path) for path in directory.glob("*.gguf"))


def is_llama_cpp_installed() -> bool:
    return pkg_util.is_installed(f"{LLAMA_CPP_PACKAGE}=={LLAMA_CPP_VERSION}")


def _install_llama_cpp() -> bool:
    env = os.environ.copy()
    extra_index = env.get("UV_EXTRA_INDEX_URL")
    env["UV_EXTRA_INDEX_URL"] = (
        f"{extra_index} {LLAMA_CPP_WHEEL_INDEX}"
        if extra_index
        else LLAMA_CPP_WHEEL_INDEX
    )
    constraints = os.path.join(
        os.path.dirname(homeassistant.__file__), "package_constraints.txt"
    )
    for requirement in ("diskcache", f"{LLAMA_CPP_PACKAGE}=={LLAMA_CPP_VERSION}"):
        args = [sys.executable, "-m", "uv", "pip", "install", "--quiet"]
        args += [requirement, "--index-strategy", "unsafe-first-match"]
        args += ["--constraint", constraints]
        if (
            not pkg_util.is_virtual_env()
            and not any(
                var in os.environ for var in ("UV_SYSTEM_PYTHON", "UV_PYTHON")
            )
            and (user_site := site.getusersitepackages())
        ):
            args += ["--python", sys.executable, "--target", user_site]
        result = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        if result.returncode != 0:
            _LOGGER.error(
                "Failed to install %s: %s",
                requirement,
                result.stderr.decode("utf-8", "replace").strip(),
            )
            return False
    return True


async def async_ensure_llama_cpp(hass: HomeAssistant) -> None:
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
