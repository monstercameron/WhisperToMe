from __future__ import annotations

from collections.abc import Iterator
from queue import Full, Queue
from time import monotonic

import numpy as np

from whispertome.audio.types import AudioChunk
from whispertome.config import AudioConfig
from whispertome.errors import AudioError


class MicrophoneInput:
    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._frames_per_block = max(1, int(config.sample_rate * config.block_ms / 1000))

    def chunks(self) -> Iterator[AudioChunk]:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioError("sounddevice is required for microphone capture") from exc

        queue: Queue[np.ndarray] = Queue(maxsize=32)

        def callback(indata, frames, time, status) -> None:  # type: ignore[no-untyped-def]
            if status:
                return
            mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
            try:
                queue.put_nowait(mono)
            except Full:
                _ = queue.get_nowait()
                queue.put_nowait(mono)

        try:
            with sd.InputStream(
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                blocksize=self._frames_per_block,
                dtype="float32",
                callback=callback,
            ):
                started = monotonic()
                while True:
                    samples = queue.get()
                    timestamp_ms = int((monotonic() - started) * 1000)
                    yield AudioChunk(
                        samples=samples,
                        sample_rate=self._config.sample_rate,
                        timestamp_ms=timestamp_ms,
                    )
        except Exception as exc:  # pragma: no cover - device dependent
            raise AudioError(f"Microphone capture failed: {exc}") from exc

