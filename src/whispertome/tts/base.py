from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from whispertome.audio.types import SynthesizedSpeech
from whispertome.models.loading import ModelLoadReporter


@dataclass(frozen=True)
class SpeechSynthesisResult:
    speech: SynthesizedSpeech
    latency_ms: float
    model: str
    provider: str


class VoiceNotSupportedError(RuntimeError):
    """Raised when a backend cannot change voices at runtime."""


class TextToSpeechModel(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def synthesize(self, text: str) -> SpeechSynthesisResult:
        raise NotImplementedError

    # --- optional capabilities (default no-op / unsupported) --------------

    def set_load_reporter(self, reporter: ModelLoadReporter) -> None:
        """Receive a reporter to emit load-progress events. Default: ignore."""

    def list_voices(self) -> list[str]:
        """Return selectable voice names. Default: none advertised."""
        return []

    def current_voice(self) -> str | None:
        """Return the active voice name, if the backend has one."""
        return None

    def set_voice(self, voice: str) -> None:
        """Switch the active voice at runtime. Default: unsupported."""
        raise VoiceNotSupportedError(
            f"{type(self).__name__} does not support runtime voice switching"
        )

