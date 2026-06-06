from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FloatArray = np.ndarray


@dataclass(frozen=True)
class AudioChunk:
    samples: FloatArray
    sample_rate: int
    timestamp_ms: int

    @property
    def duration_ms(self) -> int:
        return int((len(self.samples) / self.sample_rate) * 1000)


@dataclass(frozen=True)
class AudioBuffer:
    samples: FloatArray
    sample_rate: int

    @property
    def duration_ms(self) -> int:
        return int((len(self.samples) / self.sample_rate) * 1000)


@dataclass(frozen=True)
class SynthesizedSpeech:
    samples: FloatArray
    sample_rate: int

