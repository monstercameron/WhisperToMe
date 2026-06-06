from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from whispertome.audio.types import SynthesizedSpeech


@dataclass(frozen=True)
class SpeechSynthesisResult:
    speech: SynthesizedSpeech
    latency_ms: float
    model: str
    provider: str


class TextToSpeechModel(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def synthesize(self, text: str) -> SpeechSynthesisResult:
        raise NotImplementedError

