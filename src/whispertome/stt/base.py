from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from whispertome.audio.types import AudioBuffer
from whispertome.models.loading import ModelLoadReporter
from whispertome.runtime.events import EventBus


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

    def set_load_reporter(self, reporter: ModelLoadReporter) -> None:
        """Receive a reporter to emit load-progress events. Default: ignore."""

    def set_event_bus(self, bus: EventBus) -> None:
        """Receive a runtime event bus to publish live STT events. Default: ignore."""

