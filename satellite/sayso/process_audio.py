"""Register SaySo wake detection with upstream LVA external wake hooks."""

from __future__ import annotations

import logging
import os
from typing import Any

from .wake.hook import SaySoExternalWakeHook, install_external_wake_hook

_LOGGER = logging.getLogger(__name__)


def install_wake_audio_path(lva_main: Any, hook: SaySoExternalWakeHook) -> None:
    """Install the external wake hook and preserve upstream SystemExit handling."""
    install_external_wake_hook(lva_main, hook)
    original_run = lva_main.run

    def run() -> None:
        hook.start()
        try:
            original_run()
        except SystemExit as err:
            code = err.code if isinstance(err.code, int) and err.code > 0 else 1
            os._exit(code)
        finally:
            hook.shutdown()

    lva_main.run = run
