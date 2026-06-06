from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from whispertome.audio.types import AudioBuffer


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    latency_ms: float
    model: str
    provider: str


class SpeechToTextModel(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, audio: AudioBuffer) -> Transcript:
        raise NotImplementedError

