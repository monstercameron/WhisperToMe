from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from whispertome.audio.types import AudioBuffer, AudioChunk
from whispertome.config import AudioConfig


@dataclass(frozen=True)
class VadDecision:
    is_speech: bool
    rms: float


class EnergyVad:
    """Small signal-processing VAD.

    This is not an AI model, so it is not part of the NPU-only inference policy.
    """

    def __init__(self, rms_threshold: float) -> None:
        self._rms_threshold = rms_threshold

    def classify(self, chunk: AudioChunk) -> VadDecision:
        samples = chunk.samples.astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(np.square(samples))) if samples.size else 0.0)
        return VadDecision(is_speech=rms >= self._rms_threshold, rms=rms)


class UtteranceSegmenter:
    def __init__(self, config: AudioConfig, vad: EnergyVad) -> None:
        self._config = config
        self._vad = vad
        self._pre_roll_chunks = max(1, config.pre_roll_ms // config.block_ms)
        self._speech_start_chunks = max(1, config.speech_start_ms // config.block_ms)
        self._speech_end_chunks = max(1, config.speech_end_ms // config.block_ms)
        self._max_chunks = max(1, config.max_utterance_ms // config.block_ms)

    def utterances(self, chunks: Iterable[AudioChunk]) -> Iterable[AudioBuffer]:
        pre_roll: deque[AudioChunk] = deque(maxlen=self._pre_roll_chunks)
        active: list[AudioChunk] = []
        speech_run = 0
        silence_run = 0
        in_speech = False

        for chunk in chunks:
            decision = self._vad.classify(chunk)

            if not in_speech:
                pre_roll.append(chunk)
                speech_run = speech_run + 1 if decision.is_speech else 0
                if speech_run >= self._speech_start_chunks:
                    in_speech = True
                    active = list(pre_roll)
                    silence_run = 0
                continue

            active.append(chunk)
            silence_run = silence_run + 1 if not decision.is_speech else 0
            if silence_run >= self._speech_end_chunks or len(active) >= self._max_chunks:
                yield self._merge(active)
                active = []
                pre_roll.clear()
                speech_run = 0
                silence_run = 0
                in_speech = False

    @staticmethod
    def _merge(chunks: list[AudioChunk]) -> AudioBuffer:
        if not chunks:
            raise ValueError("Cannot merge empty chunk list")
        sample_rate = chunks[0].sample_rate
        samples = np.concatenate([chunk.samples for chunk in chunks]).astype(np.float32, copy=False)
        return AudioBuffer(samples=samples, sample_rate=sample_rate)

