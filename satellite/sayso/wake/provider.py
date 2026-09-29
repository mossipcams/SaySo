from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from .detection import Detection


class WakeWordProvider(Protocol):

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def suspend(self) -> None:
        pass

    def resume(self) -> None:
        ...

    def reset(self) -> None:
        pass

    def process_pcm(self, pcm_s16le: bytes, sample_rate: int = 16000) -> Optional[Detection]:
        pass

    def predict_window(
        self,
        window: np.ndarray,
        sample_index: int | None = None,
    ) -> Optional[Detection]:
        pass

    def shutdown(self) -> None:
        ...

    @property
    def available(self) -> bool:
        pass
